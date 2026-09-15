"""B-3 部分观测模拟器（D12）回归测试：可复现、分层、接口语义。"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))

import partial_obs_sim as sim  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok   {name}")
    else:
        FAILED += 1
        print(f"  FAIL {name} {detail}")


def main() -> int:
    print("[1] 构建可复现")
    a = sim.build(sim.SEED, 50)
    b = sim.build(sim.SEED, 50)
    check("同 seed 两次构建完全一致", json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True))
    c = sim.build(sim.SEED + 1, 50)
    check("不同 seed 结果不同", json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True))
    check("规模 = 3 档 × per_level", len(a) == 150, str(len(a)))

    print("[2] 遮蔽档位与字段")
    for level, k in sim.LEVELS.items():
        rows = [r for r in a if r["level"] == level]
        check(f"{level} 每条遮蔽 {k} 种气体", all(len(r["masked_gases"]) == k for r in rows))
        check(
            f"{level} 被遮蔽气体在 visible_gases 中为 None",
            all(all(r["visible_gases"][g] is None for g in r["masked_gases"]) for r in rows),
        )
        check(
            f"{level} 可见气体与 full_gases 一致",
            all(
                all(r["visible_gases"][g] == r["full_gases"][g] for g in r["full_gases"] if g not in r["masked_gases"])
                for r in rows
            ),
        )
    check("initial 与 hidden 征兆无交集", all(not (set(r["initial_evidence"]) & set(r["hidden_evidence"])) for r in a))
    check("不含 normal 标签", all(r["fault_label"] != "normal" for r in a))
    check("unavailable_symptoms 覆盖全部非气体征兆", all(set(r["unavailable_symptoms"]) == set(sim.NON_GAS_SYMPTOMS) for r in a))

    print("[3] 正式数据文件")
    path = sim.OUT_DIR / sim.OUT
    stats_path = sim.OUT_DIR / "build_stats.json"
    if path.exists() and stats_path.exists():
        rows = sim.load_d12(path)
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        check("正式文件 1500 条", len(rows) == 1500, str(len(rows)))
        check("sha256 与 build_stats 一致", sim.sha256_of(rows) == stats["sha256"])
        src = sim.load_real()
        src_dist = Counter(r["fault_label"] for r in src)
        d12_dist = Counter(r["fault_label"] for r in rows)
        max_dev = max(abs(src_dist[f] / len(src) - d12_dist[f] / len(rows)) for f in src_dist)
        check("标签分布最大偏差 ≤ 2%", max_dev <= 0.02, f"{max_dev:.4f}")
        for level in sim.LEVELS:
            check(f"{level} 档 500 条", sum(1 for r in rows if r["level"] == level) == 500)
        check("load_d12(level, limit) 过滤有效", len(sim.load_d12(path, level="heavy", limit=7)) == 7)
    else:
        check("正式文件存在（请先 --build）", False, str(path))

    print("[4] PartialObsEnv 接口")
    r0 = next(r for r in a if r["level"] == "medium")
    env = sim.PartialObsEnv(r0)
    ctx0 = env.context()
    check("初始 context 气体含 None", any(v is None for v in ctx0.values()))
    check("初始 evidence == initial_evidence", env.evidence == r0["initial_evidence"])
    hidden_sym = next(iter(r0["hidden_evidence"]))
    v = env.reveal(hidden_sym)
    check("reveal 隐藏征兆返回 bool", isinstance(v, bool) and v == r0["hidden_evidence"][hidden_sym], repr(v))
    check("reveal 后 evidence 含该征兆", hidden_sym in env.evidence)
    check("reveal 后对应气体变为可见", all(env.context()[g] is not None for g in sim._SYMPTOM_GASES[hidden_sym]))
    check("reveal 不可用征兆返回 None", env.reveal("oil_temp_elevated") is None)
    check("成本累计 > 0 且 n_queries 记录", env.total_cost > 0 and env.n_queries == 2, f"{env.total_cost} {env.n_queries}")
    check("history 长度 2", len(env.history) == 2)
    check("masked_remaining 递减", len(env.masked_remaining) < len(r0["masked_gases"]))
    before = env.total_cost
    env.reveal(hidden_sym)
    check("重复 reveal 同一征兆不重复计费", env.total_cost == before, f"{before}->{env.total_cost}")
    env2 = sim.PartialObsEnv(r0)
    g = r0["masked_gases"][0]
    val = env2.reveal_gas(g)
    check("reveal_gas 返回真实浓度", val == r0["full_gases"][g], repr(val))

    print(f"结果：{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
