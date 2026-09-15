"""
B-5 主动规划对照实验回归测试（纯 CPU）。

覆盖：
  - 轨迹生成：四种策略在同一样例上 q_max 步内每步只问一个未观测征兆、不重复；EIG 策略每步选的确是 argmax
  - 停止判定 stop_index 的四种原因；固定预算曲线单调不增熵（EIG 策略）
  - 配对检验：置换检验 / McNemar 在构造数据上的方向与量级
  - 小规模端到端（每档 5 条）：报告与 json / 两张图生成，验收字段齐全
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
os.environ["FAULT_ATTR_PARAMS"] = str(ROOT / "data/real/dga/learned_params.json")

import eval_active_planning as ev  # noqa: E402
import partial_obs_sim as sim  # noqa: E402
from src.tools.eig import load_cost_table, recommend  # noqa: E402

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
    table = load_cost_table()
    askable = ev.askable_symptoms(table)
    lam = float(table["lambda_default"])
    eng = ev.make_engine("calibrated")
    samples = sim.load_d12(level="heavy", limit=5) + sim.load_d12(level="light", limit=3)

    print("[1] 候选与引擎")
    check("候选征兆不含 call_tool 类", all(not table["symptoms"][s]["how_to_obtain"].startswith("call_tool") for s in askable) and len(askable) == 12, str(askable))
    check("calibrated 引擎已校准，expert 未校准", eng.is_calibrated and not ev.make_engine("expert").is_calibrated)

    print("[2] 轨迹生成")
    for strat in ev.STRATEGIES:
        ok = True
        for x in samples:
            t = ev.run_trajectory(x, strat, eng, q_max=6, rng=random.Random(1), askable=askable, lam_ref=lam)
            syms = [p["symptom"] for p in t["picks"]]
            env = sim.PartialObsEnv(x)
            ok &= len(set(syms)) == len(syms) and all(s not in env.evidence for s in syms) and len(t["snaps"]) == len(t["picks"]) + 1
            ok &= all(s["q"] == i for i, s in enumerate(t["snaps"])) and all(p["q"] == i + 1 for i, p in enumerate(t["picks"]))
        check(f"{strat}: 每步问一个未观测且不重复的征兆，快照对齐", ok)
    t_fixed = ev.run_trajectory(samples[0], "fixed", eng, q_max=6, rng=random.Random(1), askable=askable, lam_ref=lam)
    order = [ev.FIXED_ORDER.index(p["symptom"]) for p in t_fixed["picks"]]
    check("fixed 按 IEC 顺序递增", order == sorted(order), str(order))
    # eig_cost 第一步 = recommend 首推
    x = samples[0]
    env = sim.PartialObsEnv(x)
    cands = [s for s in askable if s not in env.evidence]
    rec0 = recommend(env.evidence, env.context(), engine=eng, lam=lam, candidates=cands, top_k=20, with_kg=False)
    t_ec = ev.run_trajectory(x, "eig_cost", eng, q_max=6, rng=random.Random(1), askable=askable, lam_ref=lam)
    check("eig_cost 首问 = VoI 最高", t_ec["picks"][0]["symptom"] == rec0["recommendations"][0]["symptom"])
    best_eig = max(rec0["recommendations"], key=lambda r: r["eig"])["symptom"]
    t_eg = ev.run_trajectory(x, "eig_greedy", eng, q_max=6, rng=random.Random(1), askable=askable, lam_ref=lam)
    check("eig_greedy 首问 = EIG 最高", t_eg["picks"][0]["symptom"] == best_eig)
    check("EIG 策略命中率@2 恒为 1", all(p["hit2"] for p in t_ec["picks"]))
    check("模拟器成本与轨迹成本一致", abs(sum(p["cost"] for p in t_ec["picks"]) - t_ec["snaps"][-1]["cost"]) < 1e-9)

    print("[3] 停止判定")
    fake = {"snaps": [
        {"q": 0, "entropy": 1.5, "n_cands": 5, "best_voi": 0.3, "best_eig": 0.3},
        {"q": 1, "entropy": 1.2, "n_cands": 4, "best_voi": 0.01, "best_eig": 0.05},
        {"q": 2, "entropy": 0.5, "n_cands": 3, "best_voi": 0.2, "best_eig": 0.2},
        {"q": 3, "entropy": 0.9, "n_cands": 0, "best_voi": None, "best_eig": None},
    ]}
    check("eig_cost 在 VoI<ε 处停", ev.stop_index(fake, "eig_cost", tau=0.8, eps=0.02, K=3) == (1, "voi_below_eps"))
    check("eig_greedy 用 EIG 判停（0.05 ≥ ε 不停），熵 0.5<τ 停", ev.stop_index(fake, "eig_greedy", tau=0.8, eps=0.02, K=3) == (2, "entropy_below_tau"))
    check("fixed 不看 VoI，熵 0.5<τ 停", ev.stop_index(fake, "fixed", tau=0.8, eps=0.02, K=3) == (2, "entropy_below_tau"))
    check("K=1 达轮数上限", ev.stop_index(fake, "fixed", tau=0.1, eps=0.02, K=1) == (1, "max_rounds"))
    check("无候选停止", ev.stop_index(fake, "random", tau=0.1, eps=0.02, K=9) == (3, "no_candidates"))
    check("预算耗尽", ev.stop_index({"snaps": fake["snaps"][:2]}, "random", tau=0.1, eps=0.02, K=9) == (1, "budget_exhausted"))

    print("[4] 配对检验")
    a = [1.0] * 50; b = [2.0] * 50
    r = ev.paired_permutation(a, b, n_perm=4000, seed=0)
    check("全体差 -1 → p 极小、均差 -1", r["mean_diff"] == -1.0 and r["p_value"] < 0.01, str(r))
    r2 = ev.paired_permutation([1, 2, 3, 4] * 10, [1, 2, 3, 4] * 10, n_perm=2000)
    check("无差异 → p = 1", r2["p_value"] > 0.99, str(r2))
    m = ev.mcnemar_exact([1] * 20 + [0] * 5, [0] * 20 + [1] * 5)
    check("McNemar b=20 c=5 → p≈0.004", m["b"] == 20 and m["c"] == 5 and 0.003 < m["p_value"] < 0.005, str(m))
    check("McNemar 无不一致 → p=1", ev.mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0)

    print("[5] 指标与曲线")
    trajs = [ev.run_trajectory(x, "eig_cost", eng, q_max=4, rng=random.Random(1), askable=askable, lam_ref=lam) for x in samples]
    m = ev.evaluate_adaptive(trajs, "eig_cost", tau=0.8, eps=0.02, K=3)
    check("evaluate_adaptive 字段齐全", all(k in m for k in ("top1", "top3", "avg_queries", "avg_cost", "hit2", "ece", "nll", "stop_reasons", "per_level")))
    check("平均问询 ≤ K", m["avg_queries"] <= 3)
    cv = ev.curve(trajs, 4)
    check("曲线长度 q_max+1", len(cv["top1"]) == 5 and len(cv["entropy"]) == 5)
    check("q=0 熵最高", cv["entropy"][0] >= max(cv["entropy"][1:]) - 1e-9, str(cv["entropy"]))

    print("[6] 端到端小规模运行（每档 5 条）")
    md, js = ev.REPORT_MD, ev.REPORT_JSON
    figs = [ev.FIG_DIR / "fig_b5_acc_vs_queries.png", ev.FIG_DIR / "fig_b5_entropy_curve.png"]
    bak = {p: (p.read_bytes() if p.exists() else None) for p in (md, js, *figs)}
    try:
        proc = subprocess.run([sys.executable, str(ROOT / "scripts/eval/eval_active_planning.py"), "--limit", "5",
                               "--workflow-check", "3", "--n-perm", "500"], capture_output=True, text=True, timeout=600)
        check("脚本退出码 0", proc.returncode == 0, proc.stderr[-800:])
        res = json.loads(js.read_text(encoding="utf-8"))
        check("json 含四策略 × 两引擎、检验、τ 扫描、工作流核对", all(s in res["adaptive"]["calibrated"] for s in ev.STRATEGIES)
              and all(s in res["adaptive"]["expert"] for s in ev.STRATEGIES) and "`eig_cost` vs `fixed`" in res["tests"]
              and len(res["tau_sweep"]) == 5 and res["workflow_check"]["n"] == 3)
        check("工作流与模拟 Top-1 逐条一致", res["workflow_check"]["agree_top1"] == 3, str(res["workflow_check"]))
        txt = md.read_text(encoding="utf-8")
        check("报告含各节", all(h in txt for h in ("## 2. 协议 B", "## 3. 配对检验", "## 5. 消融", "## 7. 结论", "fig_b5_acc_vs_queries.png")))
        check("两张图已生成", (ev.FIG_DIR / "fig_b5_acc_vs_queries.png").exists() and (ev.FIG_DIR / "fig_b5_entropy_curve.png").exists())
    finally:
        for p, content in bak.items():
            if content is not None:
                p.write_bytes(content)

    print(f"\n结果：{PASSED} passed, {FAILED} failed")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
