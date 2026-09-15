#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼（阿里云 Model Studio）模型调优 SFT 数据格式：转换与校验。

百炼 SFT（对话 / 工具调用）单行 JSON 结构（ChatML 变体）：
    {
      "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "...",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "...", "arguments": "<JSON 字符串>"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "<JSON 字符串>"},
        {"role": "assistant", "content": "..."}
      ],
      "tools": [{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}]
    }

与 ms-swift 格式的差异（本模块负责抹平）：
1. 不允许自定义顶层字段（`meta` 剥离）；
2. `tool` 消息不支持 OpenAI 的 `name` 字段（剥离）、不支持 `weight`；
3. `tool_calls[].id` 必填，且 `tool` 消息的 `tool_call_id` 必须与其前一条 assistant 的某个 `tool_calls[].id` 对应；
4. `function.arguments` 必须是 JSON 字符串而非对象；
5. 所有 assistant 消息都会参与训练（不支持逐条 loss 权重），因此错误恢复轨迹的「刻意错误首调」
   只应出现在上下文，不作为最后一条 assistant（export_sft.multi_turn_samples 已保证）。

单文件上限 200 MB（控制台 300 MB），此处按 200 MB 提示。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ALLOWED_ROLES = {"system", "user", "assistant", "tool"}
ALLOWED_MSG_KEYS = {
    "system": {"role", "content"},
    "user": {"role", "content"},
    "assistant": {"role", "content", "tool_calls"},
    "tool": {"role", "content", "tool_call_id"},
}
ALLOWED_TOP_KEYS = {"messages", "tools"}
MAX_FILE_BYTES = 200 * 1024 * 1024


def _strip_tool_message(msg: Dict[str, Any]) -> Dict[str, Any]:
    out = {"role": "tool", "tool_call_id": msg["tool_call_id"], "content": msg["content"]}
    if not isinstance(out["content"], str):
        out["content"] = json.dumps(out["content"], ensure_ascii=False)
    return out


def to_bailian_record(sample: Dict[str, Any]) -> Dict[str, Any]:
    """把 export_sft 产出的 ms-swift 样本（messages + tools [+ meta]）转成百炼 SFT 记录。

    - 剥离 `meta` 与 tool 消息的 `name`；
    - 为缺少 `id` 的 tool_calls 顺序补 `call_{k}`（k 全局从 1 递增，与已有 `call_{k}` 编号不冲突时直接沿用）；
    - `function.arguments` 若为对象则序列化为字符串。
    """
    msgs: List[Dict[str, Any]] = []
    next_id = 1
    for m in sample["messages"]:
        role = m["role"]
        if role == "tool":
            msgs.append(_strip_tool_message(m))
            continue
        nm: Dict[str, Any] = {"role": role, "content": m.get("content", "") or ""}
        if role == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                fn = c["function"]
                args = fn["arguments"]
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False)
                cid = c.get("id") or f"call_{next_id}"
                next_id += 1
                calls.append({"id": cid, "type": "function", "function": {"name": fn["name"], "arguments": args}})
            nm["tool_calls"] = calls
        msgs.append(nm)
    rec: Dict[str, Any] = {"messages": msgs}
    if sample.get("tools"):
        rec["tools"] = sample["tools"]
    return rec


def validate_bailian_record(rec: Dict[str, Any], tool_names: Iterable[str] | None = None) -> List[str]:
    """返回错误列表；空列表表示合法。tool_names 给出时同时校验 tool_calls 的函数名在 tools 内。"""
    errs: List[str] = []
    extra = set(rec.keys()) - ALLOWED_TOP_KEYS
    if extra:
        errs.append(f"顶层多余字段: {sorted(extra)}")
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return errs + ["messages 缺失或少于 2 条"]
    names = set(tool_names) if tool_names is not None else (
        {t["function"]["name"] for t in rec.get("tools", []) if t.get("type") == "function"} or None)

    if msgs[0]["role"] not in ("system", "user"):
        errs.append("首条消息必须为 system 或 user")
    if msgs[-1]["role"] != "assistant":
        errs.append("末条消息必须为 assistant（训练目标）")
    if not any(m.get("role") == "user" for m in msgs):
        errs.append("缺少 user 消息")

    pending_ids: set = set()
    for i, m in enumerate(msgs):
        role = m.get("role")
        if role not in ALLOWED_ROLES:
            errs.append(f"msg[{i}] 非法 role: {role!r}")
            continue
        bad = set(m.keys()) - ALLOWED_MSG_KEYS[role]
        if bad:
            errs.append(f"msg[{i}]({role}) 多余字段: {sorted(bad)}")
        if not isinstance(m.get("content", ""), str):
            errs.append(f"msg[{i}]({role}) content 必须为字符串")
        if role in ("system", "user") and not m.get("content"):
            errs.append(f"msg[{i}]({role}) content 为空")
        if role == "assistant":
            calls = m.get("tool_calls")
            if calls is None and not m.get("content"):
                errs.append(f"msg[{i}] assistant 既无 content 也无 tool_calls")
            pending_ids = set()
            if calls is not None:
                if not isinstance(calls, list) or not calls:
                    errs.append(f"msg[{i}] tool_calls 必须为非空列表")
                    continue
                for j, c in enumerate(calls):
                    cid = c.get("id")
                    if not isinstance(cid, str) or not cid:
                        errs.append(f"msg[{i}].tool_calls[{j}] 缺少 id")
                    elif cid in pending_ids:
                        errs.append(f"msg[{i}].tool_calls[{j}] id 重复: {cid}")
                    else:
                        pending_ids.add(cid)
                    if c.get("type") != "function":
                        errs.append(f"msg[{i}].tool_calls[{j}] type 必须为 function")
                    fn = c.get("function") or {}
                    if names is not None and fn.get("name") not in names:
                        errs.append(f"msg[{i}].tool_calls[{j}] 函数名不在 tools 中: {fn.get('name')!r}")
                    args = fn.get("arguments")
                    if not isinstance(args, str):
                        errs.append(f"msg[{i}].tool_calls[{j}] arguments 必须为 JSON 字符串")
                    else:
                        try:
                            json.loads(args)
                        except json.JSONDecodeError:
                            errs.append(f"msg[{i}].tool_calls[{j}] arguments 不是合法 JSON")
        elif role == "tool":
            tcid = m.get("tool_call_id")
            if not tcid:
                errs.append(f"msg[{i}] tool 消息缺少 tool_call_id")
            elif tcid not in pending_ids:
                errs.append(f"msg[{i}] tool_call_id={tcid!r} 未对应前一条 assistant 的 tool_calls")
            else:
                pending_ids.discard(tcid)
            if i > 0 and msgs[i - 1]["role"] not in ("assistant", "tool"):
                errs.append(f"msg[{i}] tool 消息之前必须是 assistant 或 tool")
    return errs


def validate_bailian_file(path: Path, max_errors: int = 20) -> Tuple[int, List[str]]:
    """逐行校验 jsonl 文件。返回 (行数, 错误列表[最多 max_errors 条])。"""
    path = Path(path)
    errs: List[str] = []
    n = 0
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        errs.append(f"文件 {size / 1024 / 1024:.1f} MB 超过百炼单文件上限 200 MB")
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            n += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errs.append(f"L{ln}: JSON 解析失败 {e}")
            else:
                for e in validate_bailian_record(obj):
                    errs.append(f"L{ln}: {e}")
            if len(errs) >= max_errors:
                errs.append("... 错误过多，截断")
                break
    return n, errs


if __name__ == "__main__":  # 命令行：python bailian_format.py file1.jsonl [file2 ...]
    import sys
    rc = 0
    for p in sys.argv[1:]:
        n, errs = validate_bailian_file(Path(p))
        print(f"{p}: {n} 行, {'通过' if not errs else f'{len(errs)} 处错误'}")
        for e in errs:
            print("  ", e)
        rc |= bool(errs)
    sys.exit(rc)
