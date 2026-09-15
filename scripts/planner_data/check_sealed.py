#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验 D8 test 切分是否仍与 data/planner/sft/SEALED.md 登记的种子级哈希一致。

只比对种子级（seeds/*.jsonl 的 test 行）哈希；导出文件哈希随系统提示变化，仅供参考不强制。
退出码 0 = 一致；1 = 不一致或 SEALED.md 缺失。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEALED = ROOT / "data/planner/sft/SEALED.md"
SEEDS_DIR = ROOT / "data/planner/seeds"


def seed_hash(name: str) -> tuple[int, str]:
    lines = [l for l in open(SEEDS_DIR / name, encoding="utf-8") if l.strip() and json.loads(l)["split"] == "test"]
    return len(lines), hashlib.sha256("".join(lines).encode()).hexdigest()


def main() -> int:
    if not SEALED.exists():
        print("SEALED.md 不存在")
        return 1
    text = SEALED.read_text(encoding="utf-8")
    ok = True
    for name in ["task_seeds.jsonl", "multi_turn_seeds.jsonl"]:
        n, h = seed_hash(name)
        m = re.search(rf"seeds/{re.escape(name)}（test 切分 (\d+) 条[^`]*` \| `([0-9a-f]{{64}})`", text)
        if not m:
            print(f"[MISSING] SEALED.md 中无 {name} 记录")
            ok = False
            continue
        exp_n, exp_h = int(m.group(1)), m.group(2)
        same = (n == exp_n and h == exp_h)
        print(f"[{'OK' if same else 'CHANGED'}] {name}: test {n} 条 sha256 {h[:12]}… (登记 {exp_n} 条 {exp_h[:12]}…)")
        ok &= same
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
