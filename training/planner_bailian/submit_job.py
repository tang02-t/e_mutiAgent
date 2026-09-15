#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
百炼（阿里云 Model Studio）模型调优任务提交器：文件上传 → 创建任务 → 轮询 / 日志 → 部署。

只依赖标准库（urllib），不依赖 dashscope SDK。所有子命令默认 `--dry-run`：只打印将要发送的请求
（方法、URL、请求头脱敏、请求体），不产生任何网络请求与费用；加 `--execute` 才真正发送。

用法（示例，均在项目根目录执行）：
  # 1) 上传训练 / 验证集（purpose=fine-tune），打印/返回 file_id
  python3 training/planner_bailian/submit_job.py upload data/planner/sft/bailian_train.jsonl data/planner/sft/bailian_dev.jsonl
  # 2) 创建 SFT 任务（默认 qwen3-8b + efficient_sft + PLAN D-1 超参）
  python3 training/planner_bailian/submit_job.py create --training-file <train_file_id> --validation-file <dev_file_id> \
      --job-name planner_sft_m1 --model-name planner-sft-m1
  # 2') 创建 DPO 任务（D-3：以 M1 的 finetuned_output 为 model）
  python3 training/planner_bailian/submit_job.py create --stage dpo --model <M1 finetuned_output> \
      --training-file <dpo_train_file_id> --validation-file <dpo_dev_file_id> --job-name planner_dpo_m4_exec
  # 3) 查询状态 / 拉日志（--wait 轮询直到终态）
  python3 training/planner_bailian/submit_job.py status --job-id <job_id> --wait
  python3 training/planner_bailian/submit_job.py logs --job-id <job_id>
  # 4) 部署（capacity=1）并查询部署状态
  python3 training/planner_bailian/submit_job.py deploy --model-name <finetuned_output> --capacity 1
  python3 training/planner_bailian/submit_job.py deploy-status --deployed-model <deployed_model>

鉴权：环境变量 DASHSCOPE_API_KEY（北京地域；DPO / OSS 导入仅北京地域支持）。
计费：API 创建的调优任务按训练 Token 计费；部署后按部署时长计费，评测结束务必下线（`--execute` 慎用）。
限制：参数快照不可下载（平台约束，见 README），因此评测必须通过部署后的 API 端点完成。
每次真实执行的响应都会追加写入 training/planner_bailian/jobs.jsonl 供 README 记录 job_id / file_id。
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request

BASE = "https://dashscope.aliyuncs.com/api/v1"
HERE = Path(__file__).resolve().parent
LEDGER = HERE / "jobs.jsonl"

# PLAN D-1 / D-3 超参（与清单一致；learning_rate 不传则用平台默认）
HP_SFT = {"n_epochs": 3, "batch_size": 16, "max_length": 4096}
HP_DPO = {"n_epochs": 2, "batch_size": 16, "max_length": 4096, "dpo_beta": 0.1}
TRAINING_TYPE = {"sft": "efficient_sft", "dpo": "dpo_lora"}
BASE_MODEL = "qwen3-8b"
TERMINAL = {"SUCCEEDED", "FAILED", "CANCELED", "CANCELLED"}


# ── HTTP 封装 ──────────────────────────────────────────────
def _api_key() -> str:
    key = os.getenv("DASHSCOPE_API_KEY", "")
    return key


def _mask(key: str) -> str:
    return (key[:6] + "…" + key[-4:]) if len(key) > 12 else ("<未设置>" if not key else "***")


def _send(method: str, url: str, body: Optional[bytes], headers: Dict[str, str],
          execute: bool, preview: Any = None) -> Dict[str, Any]:
    shown = dict(headers)
    if "Authorization" in shown:
        shown["Authorization"] = "Bearer " + _mask(_api_key())
    print(f"\n>>> {method} {url}")
    print("    headers:", json.dumps(shown, ensure_ascii=False))
    if preview is not None:
        print("    body:", json.dumps(preview, ensure_ascii=False, indent=2))
    if not execute:
        print("    [dry-run] 未发送。加 --execute 才会真正调用并产生费用。")
        return {"dry_run": True}
    if not _api_key():
        sys.exit("DASHSCOPE_API_KEY 未设置，无法执行。")
    req = request.Request(url, data=body, method=method, headers=headers)
    try:
        with request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8") or "{}")
    except error.HTTPError as e:
        payload = e.read().decode("utf-8", "ignore")
        print(f"    HTTP {e.code}: {payload}")
        sys.exit(1)
    print("<<<", json.dumps(data, ensure_ascii=False, indent=2))
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "method": method, "url": url,
                            "request": preview, "response": data}, ensure_ascii=False) + "\n")
    return data


def _json_call(method: str, path: str, body: Optional[Dict[str, Any]], execute: bool) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    return _send(method, BASE + path, raw, headers, execute, preview=body)


# ── 子命令 ─────────────────────────────────────────────────
def cmd_upload(args: argparse.Namespace) -> None:
    """POST /api/v1/files  multipart: files=<file>, descriptions=<text>（purpose 固定 fine-tune）。"""
    for p in args.files:
        path = Path(p)
        if not path.exists():
            sys.exit(f"文件不存在: {path}")
        size_mb = path.stat().st_size / 1024 / 1024
        if size_mb > 200:
            sys.exit(f"{path.name} {size_mb:.1f} MB 超过单文件 200 MB 上限")
        boundary = "----bailian" + uuid.uuid4().hex
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        desc = args.description or f"planner {args.stage} {path.stem}"
        parts = [
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"{path.name}\"\r\n"
            f"Content-Type: {ctype}\r\n\r\n".encode(),
            path.read_bytes() if args.execute else b"<file bytes omitted in dry-run>",
            f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"descriptions\"\r\n\r\n{desc}\r\n".encode(),
            f"--{boundary}--\r\n".encode(),
        ]
        body = b"".join(parts)
        headers = {"Authorization": f"Bearer {_api_key()}",
                   "Content-Type": f"multipart/form-data; boundary={boundary}"}
        preview = {"files": f"{path} ({size_mb:.1f} MB)", "descriptions": desc, "purpose": "fine-tune"}
        data = _send("POST", BASE + "/files", body, headers, args.execute, preview=preview)
        if not data.get("dry_run"):
            for item in (data.get("data", {}) or {}).get("uploaded_files", []):
                print(f"file_id={item.get('file_id')}  name={item.get('name')}")


def cmd_create(args: argparse.Namespace) -> None:
    """POST /api/v1/fine-tunes。"""
    hp = dict(HP_SFT if args.stage == "sft" else HP_DPO)
    if args.learning_rate:
        hp["learning_rate"] = args.learning_rate
    if args.n_epochs:
        hp["n_epochs"] = args.n_epochs
    model = args.model or BASE_MODEL
    if args.stage == "dpo" and model == BASE_MODEL:
        print("提示：D-3 要求以 D-1 产出的 M1（finetuned_output）为 model；当前为基座 qwen3-8b。")
    body: Dict[str, Any] = {
        "model": model,
        "training_file_ids": [args.training_file],
        "hyper_parameters": hp,
        "training_type": TRAINING_TYPE[args.stage],
    }
    if args.validation_file:
        body["validation_file_ids"] = [args.validation_file]
    if args.job_name:
        body["job_name"] = args.job_name
    if args.model_name:
        body["model_name"] = args.model_name
    data = _json_call("POST", "/fine-tunes", body, args.execute)
    if not data.get("dry_run"):
        out = data.get("output", {})
        print(f"job_id={out.get('job_id')}  status={out.get('status')}  finetuned_output={out.get('finetuned_output')}")


def cmd_status(args: argparse.Namespace) -> None:
    """GET /api/v1/fine-tunes/{job_id}；--wait 每 interval 秒轮询直到终态。"""
    while True:
        data = _json_call("GET", f"/fine-tunes/{args.job_id}", None, args.execute)
        if data.get("dry_run"):
            return
        out = data.get("output", {})
        status = str(out.get("status", "")).upper()
        print(f"[{time.strftime('%H:%M:%S')}] status={status} finetuned_output={out.get('finetuned_output')}")
        if not args.wait or status in TERMINAL:
            return
        time.sleep(args.interval)


def cmd_logs(args: argparse.Namespace) -> None:
    _json_call("GET", f"/fine-tunes/{args.job_id}/logs?offset={args.offset}&line={args.lines}", None, args.execute)


def cmd_deploy(args: argparse.Namespace) -> None:
    """POST /api/v1/deployments {model_name: <finetuned_output>, capacity}。"""
    body = {"model_name": args.model_name, "capacity": args.capacity}
    data = _json_call("POST", "/deployments", body, args.execute)
    if not data.get("dry_run"):
        out = data.get("output", {})
        print(f"deployed_model={out.get('deployed_model')}  status={out.get('status')}")
        print("部署完成后：把 deployed_model 写入 config.yaml 的 llms.planner_finetuned.model_name，"
              "并运行 scripts/probe_llm_endpoint.py 验证 tool_calls 解析。")


def cmd_deploy_status(args: argparse.Namespace) -> None:
    _json_call("GET", f"/deployments/{args.deployed_model}", None, args.execute)


def cmd_undeploy(args: argparse.Namespace) -> None:
    _json_call("DELETE", f"/deployments/{args.deployed_model}", None, args.execute)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true", help="真正发送请求（默认 dry-run，只打印请求）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("upload"); p.add_argument("files", nargs="+")
    p.add_argument("--stage", choices=["sft", "dpo"], default="sft"); p.add_argument("--description", default="")
    p.set_defaults(fn=cmd_upload)

    p = sub.add_parser("create")
    p.add_argument("--stage", choices=["sft", "dpo"], default="sft")
    p.add_argument("--model", default="", help="基座或 M1 的 finetuned_output；默认 qwen3-8b")
    p.add_argument("--training-file", required=True); p.add_argument("--validation-file", default="")
    p.add_argument("--job-name", default=""); p.add_argument("--model-name", default="")
    p.add_argument("--learning-rate", type=float, default=None); p.add_argument("--n-epochs", type=int, default=None)
    p.set_defaults(fn=cmd_create)

    p = sub.add_parser("status"); p.add_argument("--job-id", required=True)
    p.add_argument("--wait", action="store_true"); p.add_argument("--interval", type=int, default=120)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("logs"); p.add_argument("--job-id", required=True)
    p.add_argument("--offset", type=int, default=0); p.add_argument("--lines", type=int, default=200)
    p.set_defaults(fn=cmd_logs)

    p = sub.add_parser("deploy"); p.add_argument("--model-name", required=True, help="fine-tunes 返回的 finetuned_output")
    p.add_argument("--capacity", type=int, default=1); p.set_defaults(fn=cmd_deploy)

    p = sub.add_parser("deploy-status"); p.add_argument("--deployed-model", required=True); p.set_defaults(fn=cmd_deploy_status)
    p = sub.add_parser("undeploy"); p.add_argument("--deployed-model", required=True); p.set_defaults(fn=cmd_undeploy)

    args = ap.parse_args()
    if not args.execute:
        print("[dry-run 模式] 不发送任何请求。")
    args.fn(args)


if __name__ == "__main__":
    main()
