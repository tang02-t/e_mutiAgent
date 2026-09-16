#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P4-2「模拟用户」口语化改写（需调用 LLM；默认 --dry-run 只估算规模与费用，不发请求）。

流程：
  1. 读取 data/planner/seeds/task_seeds.jsonl 中 needs_llm_rewrite=True 的种子（test 切分默认不改写，保持封存评测口径；
     可用 --include-test 打开）
  2. 每条种子请求 LLM 生成 K 个改写（默认 2）：口语化、省略主语、夹杂现场描述、术语与俗称混用（乙炔涨了 / 有响声 / 瓦斯动了）
  3. 守卫（不通过则丢弃该改写）：
     - 数值不变：原问题中的所有数字（DGA 浓度、日期、步长、kVA）必须原样出现在改写中
     - 关键实体不变：原问题中命中领域词典（training/planner_sft/domain_terms.json）的词至少保留 60%，或出现其同义词
     - 长度 8–200 字，不与原句或其他改写重复
     - insufficient 类：不得补出原本缺失的参数（不得新增数字）
     - 判断词守卫：改写不得引入原句没有的结论 / 故障类型词（超标、放电、过热…）；numeric_tool / composite / insufficient
       另拦截软判断（看着挺高、不太对、指向…），避免问句替 Planner 预判（2026-09-16 全量实跑：硬拦 160、软拦 140）
  4. 写出 data/planner/seeds/task_seeds_rewritten.jsonl：原种子 + 改写种子（seed_id 加后缀 -rN，query 替换，其余字段沿用，
     group_key 不变以保证切分不泄漏），并把 needs_llm_rewrite 置 False、记录 rewrite_of
  5. 之后重跑：python3 scripts/planner_data/export_sft.py --seeds data/planner/seeds/task_seeds_rewritten.jsonl

用法：
  python3 scripts/planner_data/rewrite_queries.py --dry-run                 # 估算条数 / token / 费用
  python3 scripts/planner_data/rewrite_queries.py --limit 20                # 小样试跑（会调用 LLM）
  python3 scripts/planner_data/rewrite_queries.py --k 2 --workers 4         # 全量
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import get_llm_config, load_config  # noqa: E402
from src.utils.llm import USAGE, LLMClient, LLMConfig      # noqa: E402

SEEDS = ROOT / "data/planner/seeds/task_seeds.jsonl"
OUT = ROOT / "data/planner/seeds/task_seeds_rewritten.jsonl"
TERMS = ROOT / "training/planner_sft/domain_terms.json"

SYSTEM = (
    "你是变电站一线运维人员，负责把一句「标准书面问法」改写成现场人员会说的口语问法。要求：\n"
    "1. 意思完全不变，问的目标不变；\n"
    "2. 原句里所有数字、单位、日期、数据集名（如 ETTh1）必须原样保留，不得增删或改写数字；\n"
    "3. 可以省略主语、加现场描述（如「刚巡视发现」「值班的时候」）、混用俗称（乙炔→C2H2 或「乙炔涨了」、总烃→TDCG、"
    "轻瓦斯动作→「瓦斯动了」、噪声异常→「有响声」）；\n"
    "4. 如果原句缺少某些数据，改写后也必须同样缺少，不能补上；\n"
    "5. 每条 8–120 字，K 条之间差异明显；\n"
    "6. 不得加入原句没有的判断或结论：不要说「超标 / 偏高 / 异常」，不要猜测故障类型（放电、过热、受潮等），"
    "不要暗示答案，只描述现场和数据本身；\n"
    "只输出 JSON：{\"rewrites\": [\"...\", \"...\"]}，不要解释。"
)

_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_SYMBOL_RE = re.compile(r"\b(?:C2H2|C2H4|C2H6|CH4|H2|CO2|CO|ETT[hm][12]|DL/T\s*\d+|GB/T\s*\d+|Q/GDW\s*\d+)\b", re.I)
_JUDGEMENT_RE = re.compile(
    r"超标|偏高|偏低|过高|过低|异常|不正常|不对劲|超限|放电|过热|受潮|短路|悬浮|局放|电弧|老化|绝缘劣化|绝缘故障|匝间|"
    r"火花|电晕|热点|裸金属|固体绝缘|是不是.{0,4}故障|可能是.{0,6}故障")
# 软判断词：仅对带数据 / 需工具判断的类别（numeric_tool / composite / insufficient）拦截，避免改写替用户「预判」数据高低
_SOFT_JUDGEMENT_RE = re.compile(r"看着.{0,3}[高大]|挺大|太高|很高|偏大|明显高|不太对|有点高|有点大|涨得|飙|不太正常|超了|不放心|怀疑|像是|指向")
_SOFT_GUARD_CATEGORIES = {"numeric_tool", "composite", "insufficient"}
_PRICE_PER_1K = {"input": 0.0003, "output": 0.003}  # 估算用，元 / 千 token（flash 档位量级，实际以百炼计费为准）


def _numbers(text: str) -> List[str]:
    """提取数值，先屏蔽气体符号 / 数据集名 / 标准编号中的数字。"""
    return _NUM_RE.findall(_SYMBOL_RE.sub(" ", text))


def _load_terms() -> List[str]:
    if not TERMS.exists():
        return []
    d = json.loads(TERMS.read_text(encoding="utf-8"))
    return sorted(set(d.get("terms") or []), key=lambda s: (-len(s), s))


def _syn_groups() -> Dict[str, set]:
    """图谱同义词：词 → 同组全部词，用于「同义词替换也算保留」。"""
    p = ROOT / "data/kg/synonyms.json"
    groups: Dict[str, set] = {}
    if not p.exists():
        return groups
    for _t, mapping in (json.loads(p.read_text(encoding="utf-8")).get("entities") or {}).items():
        for canon, alias in mapping.items():
            grp = {canon, *alias}
            for w in grp:
                groups.setdefault(w, set()).update(grp)
    # 常用俗称
    for grp in [{"乙炔", "C2H2"}, {"总烃", "TDCG"}, {"轻瓦斯动作", "瓦斯动了", "瓦斯动作"}, {"噪声异常", "有响声", "异响"},
                {"氢气", "H2"}, {"甲烷", "CH4"}, {"乙烯", "C2H4"}, {"乙烷", "C2H6"}]:
        for w in grp:
            groups.setdefault(w, set()).update(grp)
    return groups


class Guard:
    def __init__(self) -> None:
        self.terms = _load_terms()
        self.syn = _syn_groups()
        self._pat = re.compile("|".join(re.escape(t) for t in self.terms)) if self.terms else None

    def entities(self, q: str) -> List[str]:
        return list(dict.fromkeys(self._pat.findall(q))) if self._pat else []

    def check(self, seed: Dict[str, Any], rw: str, others: List[str]) -> Optional[str]:
        q = seed["query"]
        rw = rw.strip()
        if not (8 <= len(rw) <= 200):
            return "length"
        if rw == q or rw in others:
            return "duplicate"
        nums_q, nums_r = _numbers(q), _numbers(rw)
        if sorted(nums_q) != sorted(nums_r):
            if seed["category"] == "insufficient" and len(nums_r) > len(nums_q):
                return "added_numbers"
            if set(nums_q) - set(nums_r):
                return "lost_numbers"
            if set(nums_r) - set(nums_q):
                return "added_numbers"
        # 判断词守卫：改写中出现原句没有的结论 / 故障类型词 → 标签泄漏，丢弃
        for m in _JUDGEMENT_RE.finditer(rw):
            if m.group(0) not in q:
                return "added_judgement"
        if seed["category"] in _SOFT_GUARD_CATEGORIES:
            for m in _SOFT_JUDGEMENT_RE.finditer(rw):
                if m.group(0) not in q:
                    return "added_soft_judgement"
        ents = self.entities(q)
        if ents:
            kept = sum(1 for e in ents if e in rw or any(s in rw for s in self.syn.get(e, ())))
            if kept / len(ents) < 0.6:
                return "lost_entities"
        return None


def build_client(config: Dict[str, Any]) -> LLMClient:
    cfg = get_llm_config(config, "planner")
    return LLMClient(LLMConfig(
        provider=cfg.get("provider", "openai"), model_name=cfg.get("model_name", ""),
        temperature=float(cfg.get("rewrite_temperature", 0.8)), max_tokens=512,
        base_url=cfg.get("base_url", ""), api_key=cfg.get("api_key", ""), api_key_env=cfg.get("api_key_env", "DASHSCOPE_API_KEY"),
        enable_thinking=False, timeout=float(cfg.get("timeout", 60.0)), max_retries=2))


def rewrite_one(client: LLMClient, seed: Dict[str, Any], k: int) -> List[str]:
    ctx = ""
    if seed.get("context"):
        ctx = "\n（注意：该问题附带前端表单数据，问句本身可以不含数字，但不要把表单数字写进问句。）"
    user = f"K={k}\n标准问法：{seed['query']}{ctx}"
    msg = client.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
    text = (msg.get("content") or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        obj = json.loads(text)
        rws = obj.get("rewrites") if isinstance(obj, dict) else obj
        return [str(x) for x in (rws or []) if isinstance(x, (str, int, float))]
    except json.JSONDecodeError:
        return [ln.strip("-• \t") for ln in text.splitlines() if ln.strip()][:k]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default=str(SEEDS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--k", type=int, default=2, help="每条种子改写数")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include-test", action="store_true", help="也改写 test 切分（默认封存不动）")
    ap.add_argument("--categories", default="", help="只改写这些类别，逗号分隔")
    ap.add_argument("--dry-run", action="store_true", help="不调用 LLM，只统计规模与估算费用")
    args = ap.parse_args()

    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    cats = {c for c in args.categories.split(",") if c}
    todo = [s for s in seeds if s.get("needs_llm_rewrite") and (args.include_test or s["split"] != "test")
            and (not cats or s["category"] in cats)]
    if args.limit:
        todo = todo[:args.limit]
    est_in = len(todo) * (len(SYSTEM) + 60) // 1.5  # 粗估：中文约 1.5 字/token
    est_out = len(todo) * args.k * 60
    print(f"待改写种子 {len(todo)} 条（总 {len(seeds)}，test 封存 {'包含' if args.include_test else '排除'}），每条 {args.k} 个改写")
    print(f"估算 token：输入约 {int(est_in):,}，输出约 {est_out:,}；费用约 "
          f"{est_in / 1000 * _PRICE_PER_1K['input'] + est_out / 1000 * _PRICE_PER_1K['output']:.2f} 元（按 flash 档位粗估）")
    if args.dry_run:
        print("dry-run 结束，未调用模型。去掉 --dry-run 即开始改写。")
        return

    config = load_config()
    client = build_client(config)
    if not client.enabled:
        raise SystemExit("LLM 未启用：请检查 config.yaml 的 llms.planner / default 配置")
    guard = Guard()
    USAGE.reset()

    done: Dict[str, List[str]] = {}
    out_path = Path(args.out)
    if out_path.exists():  # 断点续跑：已有改写的 seed 跳过
        for l in open(out_path, encoding="utf-8"):
            r = json.loads(l)
            if r.get("rewrite_of"):
                done.setdefault(r["rewrite_of"], []).append(r["query"])
    todo = [s for s in todo if s["seed_id"] not in done]
    print(f"已有改写 {len(done)} 条种子，本次需处理 {len(todo)} 条")

    rejected: Dict[str, int] = {}
    new_rows: Dict[str, List[Dict[str, Any]]] = {}
    t0 = time.time()

    def work(seed: Dict[str, Any]):
        try:
            return seed, rewrite_one(client, seed, args.k), None
        except Exception as exc:  # noqa: BLE001
            return seed, [], str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, s) for s in todo]
        for i, f in enumerate(as_completed(futs), 1):
            seed, rws, err = f.result()
            if err:
                rejected["llm_error"] = rejected.get("llm_error", 0) + 1
                continue
            kept: List[str] = []
            for rw in rws:
                why = guard.check(seed, rw, kept)
                if why:
                    rejected[why] = rejected.get(why, 0) + 1
                    continue
                kept.append(rw)
            rows = []
            for j, rw in enumerate(kept, 1):
                r = dict(seed)
                r.update({"seed_id": f"{seed['seed_id']}-r{j}", "query": rw, "needs_llm_rewrite": False,
                          "rewrite_of": seed["seed_id"], "seed_source": seed.get("seed_source"),
                          "note": (seed.get("note") or "") + " | LLM 口语化改写"})
                rows.append(r)
            new_rows[seed["seed_id"]] = rows
            if i % 50 == 0 or i == len(todo):
                u = USAGE.snapshot()
                print(f"[{i}/{len(todo)}] 保留 {sum(len(v) for v in new_rows.values())} 条改写，"
                      f"拒绝 {rejected}，tokens in/out {u['prompt_tokens']:,}/{u['completion_tokens']:,}，{time.time() - t0:.0f}s")

    # 写出：原种子（needs_llm_rewrite 置 False 表示已处理）+ 已有改写 + 新改写
    processed = set(done) | set(new_rows)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in seeds:
            s2 = dict(s)
            if s["seed_id"] in processed:
                s2["needs_llm_rewrite"] = False
            f.write(json.dumps(s2, ensure_ascii=False) + "\n")
        for sid, qs in done.items():
            base = next(s for s in seeds if s["seed_id"] == sid)
            for j, q in enumerate(qs, 1):
                r = dict(base)
                r.update({"seed_id": f"{sid}-r{j}", "query": q, "needs_llm_rewrite": False, "rewrite_of": sid})
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for rows in new_rows.values():
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total_new = sum(len(v) for v in new_rows.values())
    print(f"写出 {out_path.relative_to(ROOT)}：原种子 {len(seeds)} + 改写 {total_new + sum(len(v) for v in done.values())}；拒绝统计 {rejected}")
    print("下一步：python3 scripts/planner_data/export_sft.py --seeds", out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
