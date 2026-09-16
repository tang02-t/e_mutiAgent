#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P5-4：对封存 test 集做 Planner 推理，输出统一格式的预测文件（供 scripts/eval/eval_planner_offline.py 评分）。

两种后端：
  --backend swift   （默认，训练环境）ms-swift PtEngine 本地推理：--model 基座 [--adapters LoRA ckpt]
  --backend openai  通过 OpenAI 兼容接口推理（百炼部署模型 / 其他云端模型作为 M0 对照）：
                    --base-url --api-key-env --model-name
  A-1 基线（M0 = 未微调 qwen3.7-flash，分层抽 100 条）：
    DASHSCOPE_API_KEY=... python3 training/planner_sft/predict.py --backend openai \
        --base-url <config.yaml llms.planner.base_url> --model-name qwen3.7-flash \
        --stratified 100 --seed 42 --run-name M0_qwen3.7-flash_s42

输入：data/planner/sft/swift_test.jsonl + swift_multiturn_test.jsonl（messages 去掉最后一条 assistant 作为 prompt）
输出：<out-dir>/<run-name>.jsonl，每行：
  {"seed_id", "category", "sub_type", "decision_index", "raw": 模型原始文本,
   "pred": {"content": str, "tool_calls": [{"name", "arguments"(dict|str)}]} | null(解析失败),
   "gold": {"content": str, "tool_calls": [...]}, "latency_ms": int}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data/planner/sft"
TEST_FILES = ["swift_test.jsonl", "swift_multiturn_test.jsonl"]

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def load_test(limit: Optional[int] = None, files: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for fn in files or TEST_FILES:
        p = DATA_DIR / fn
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def stratified_sample(rows: List[Dict[str, Any]], n: int, seed: int = 42, min_per_cat: int = 5) -> List[Dict[str, Any]]:
    """按 meta.category 比例分层抽 n 条；每类保底 min_per_cat（不超过该类总数），余额按比例分配，确定性可复现。"""
    import random
    from collections import defaultdict
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r.get("meta", {}).get("category") or "unknown"].append(r)
    cats = sorted(by_cat)
    quota = {c: min(min_per_cat, len(by_cat[c])) for c in cats}
    rest = max(n - sum(quota.values()), 0)
    total = sum(len(by_cat[c]) for c in cats)
    for c in cats:
        quota[c] += int(rest * len(by_cat[c]) / total)
    # 补齐取整损失，优先给剩余样本最多的类别
    while sum(quota.values()) < n:
        c = max(cats, key=lambda k: len(by_cat[k]) - quota[k])
        if len(by_cat[c]) - quota[c] <= 0:
            break
        quota[c] += 1
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    for c in cats:
        pool = list(by_cat[c])
        rng.shuffle(pool)
        out.extend(pool[: min(quota[c], len(pool))])
    return out


def gold_of(sample: Dict[str, Any]) -> Dict[str, Any]:
    last = sample["messages"][-1]
    assert last["role"] == "assistant", "样本最后一条必须是 assistant"
    calls = []
    for tc in last.get("tool_calls") or []:
        fn = tc.get("function", tc)
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        calls.append({"name": fn.get("name"), "arguments": args})
    content = last.get("content") or ""
    # 单轮 JSON 文本规划：steps 中的 tool/arguments 也视为金标调用
    if not calls and content.strip().startswith("{"):
        try:
            plan = json.loads(content)
            for st in plan.get("steps") or []:
                if st.get("tool"):
                    calls.append({"name": st["tool"], "arguments": st.get("arguments") or {}})
        except json.JSONDecodeError:
            pass
    return {"content": content, "tool_calls": calls}


def parse_response(text: str, native_calls: Optional[List[Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    """把模型输出规范化为 {content, tool_calls}；无法解析出合法结构时返回 None（格式不合法）。"""
    calls: List[Dict[str, Any]] = []
    if native_calls:
        for tc in native_calls:
            fn = tc.get("function", tc)
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return None
            calls.append({"name": fn.get("name"), "arguments": args})
        return {"content": text or "", "tool_calls": calls}
    text = text or ""
    body = text
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        calls.append({"name": obj["name"], "arguments": args})
    if calls:
        body = _TOOL_CALL_RE.sub("", text).strip()
        return {"content": body, "tool_calls": calls}
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.S)
    if stripped.startswith("{"):
        try:
            plan = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(plan, dict) or "steps" not in plan:
            return None
        for st in plan.get("steps") or []:
            if isinstance(st, dict) and st.get("tool"):
                calls.append({"name": st["tool"], "arguments": st.get("arguments") or {}})
        return {"content": stripped, "tool_calls": calls}
    if "<tool_call>" in text:  # 有标签但未闭合/不可解析
        return None
    return {"content": text.strip(), "tool_calls": []}


# ── 后端 ────────────────────────────────────────────────────────────────────────
class SwiftBackend:
    def __init__(self, model: str, adapters: Optional[str], max_new_tokens: int) -> None:
        from swift.llm import PtEngine, RequestConfig  # type: ignore
        self.engine = PtEngine(model, adapters=[adapters] if adapters else None)
        self.cfg = RequestConfig(max_tokens=max_new_tokens, temperature=0.0)
        self._InferRequest = __import__("swift.llm", fromlist=["InferRequest"]).InferRequest

    def __call__(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> tuple[str, Optional[list]]:
        req = self._InferRequest(messages=messages, tools=tools)
        resp = self.engine.infer([req], self.cfg)[0]
        msg = resp.choices[0].message
        calls = getattr(msg, "tool_calls", None)
        native = [tc.model_dump() if hasattr(tc, "model_dump") else dict(tc) for tc in calls] if calls else None
        return msg.content or "", native


class OpenAIBackend:
    def __init__(self, base_url: str, api_key_env: str, model_name: str, max_new_tokens: int,
                 use_tools: bool = True) -> None:
        from openai import OpenAI  # type: ignore
        key = os.environ.get(api_key_env, "")
        if not key:
            raise SystemExit(f"环境变量 {api_key_env} 为空")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_tools = use_tools

    def __call__(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> tuple[str, Optional[list]]:
        kwargs: Dict[str, Any] = dict(model=self.model_name, messages=messages, temperature=0.0,
                                      max_tokens=self.max_new_tokens, extra_body={"enable_thinking": False})
        if self.use_tools and tools:
            kwargs["tools"] = tools
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        native = None
        if getattr(msg, "tool_calls", None):
            native = [{"function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in msg.tool_calls]
        return msg.content or "", native


def to_prompt_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """去掉最后一条 assistant；历史 assistant 的 tool_calls 与 tool 消息按 OpenAI 格式保留。"""
    msgs = []
    for m in sample["messages"][:-1]:
        mm = {"role": m["role"], "content": m.get("content") or ""}
        if m["role"] == "assistant" and m.get("tool_calls"):
            mm["tool_calls"] = m["tool_calls"]
        if m["role"] == "tool":
            mm["tool_call_id"] = m.get("tool_call_id", "call_1")
        msgs.append(mm)
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["swift", "openai"], default="swift")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct", help="swift 后端基座")
    ap.add_argument("--adapters", default=None, help="swift 后端 LoRA checkpoint；M0 留空")
    ap.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    ap.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    ap.add_argument("--model-name", default=None, help="openai 后端模型名")
    ap.add_argument("--no-tools", action="store_true", help="openai 后端不传 tools（考察纯文本 JSON 规划格式）")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "data/planner/predictions"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stratified", type=int, default=None,
                    help="按 category 分层抽样 N 条（每类保底 5，余额按比例；与 --limit 互斥，A-1 基线用 100）")
    ap.add_argument("--seed", type=int, default=42, help="分层抽样随机种子")
    ap.add_argument("--files", nargs="*", default=None, help="覆盖默认 test 文件列表")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    args = ap.parse_args()

    if args.backend == "swift":
        backend = SwiftBackend(args.model, args.adapters, args.max_new_tokens)
    else:
        if not args.model_name:
            raise SystemExit("--backend openai 需要 --model-name")
        backend = OpenAIBackend(args.base_url, args.api_key_env, args.model_name, args.max_new_tokens,
                                use_tools=not args.no_tools)

    samples = load_test(args.limit, args.files)
    if args.stratified:
        if args.limit:
            raise SystemExit("--stratified 与 --limit 互斥")
        samples = stratified_sample(samples, args.stratified, seed=args.seed)
        from collections import Counter
        print(f"分层抽样 {len(samples)} 条：{dict(Counter(s.get('meta', {}).get('category') for s in samples))}",
              file=sys.stderr)
    out_path = Path(args.out_dir) / f"{args.run_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():  # 断点续跑
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    done.add((r["seed_id"], r.get("decision_index")))
    n_fail = 0
    with open(out_path, "a", encoding="utf-8") as fo:
        for i, s in enumerate(samples, 1):
            meta = s.get("meta", {})
            key = (meta.get("seed_id"), meta.get("decision_index"))
            if key in done:
                continue
            t0 = time.time()
            try:
                raw, native = backend(to_prompt_messages(s), s.get("tools") or [])
                pred = parse_response(raw, native)
            except Exception as exc:  # noqa: BLE001
                raw, pred = f"__error__: {exc}", None
                n_fail += 1
            row = {"seed_id": meta.get("seed_id"), "category": meta.get("category"), "sub_type": meta.get("sub_type"),
                   "decision_index": meta.get("decision_index"), "raw": raw, "pred": pred, "gold": gold_of(s),
                   "latency_ms": int((time.time() - t0) * 1000)}
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 50 == 0:
                print(f"[{args.run_name}] {i}/{len(samples)} 失败 {n_fail}", file=sys.stderr)
    print(f"写出 {out_path}（{len(samples)} 条，新增 {len(samples) - len(done)}，调用失败 {n_fail}）")


if __name__ == "__main__":
    main()
