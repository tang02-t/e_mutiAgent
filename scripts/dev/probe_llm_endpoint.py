"""探测 LLM 端点：检查普通对话与 function calling 是否可用。

用法：
    python3 scripts/dev/probe_llm_endpoint.py            # 读取 config.yaml 中 llms.default
    DASHSCOPE_API_KEY=sk-xxx python3 scripts/dev/probe_llm_endpoint.py --key-env DASHSCOPE_API_KEY
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "fault_attribution",
        "description": "根据 DGA 数据进行故障归因",
        "parameters": {
            "type": "object",
            "properties": {"h2": {"type": "number"}, "c2h2": {"type": "number"}},
            "required": ["h2", "c2h2"],
        },
    },
}]


def load_default_llm() -> dict:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    return cfg["llms"]["default"]


def probe(base_url: str, key: str, model: str) -> bool:
    client = OpenAI(api_key=key, base_url=base_url, timeout=60)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "只回复两个字：成功"}],
            extra_body={"enable_thinking": False},
            max_tokens=16,
        )
        print(f"chat OK -> {r.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001
        print(f"chat FAIL -> {type(e).__name__}: {str(e)[:200]}")
        return False
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "H2=150, C2H2=12，请调用工具做故障归因"}],
            tools=TOOLS,
            tool_choice="auto",
            extra_body={"enable_thinking": False},
            max_tokens=256,
        )
        tc = r.choices[0].message.tool_calls
        print(f"tools OK -> {json.dumps([t.function.model_dump() for t in tc] if tc else None, ensure_ascii=False)}")
    except Exception as e:  # noqa: BLE001
        print(f"tools FAIL -> {type(e).__name__}: {str(e)[:200]}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--key-env", default=None, help="从指定环境变量读取 API Key；默认读 config.yaml")
    a = ap.parse_args()
    d = load_default_llm()
    key = os.getenv(a.key_env) if a.key_env else d["api_key"]
    if not key:
        print("未找到 API Key"); sys.exit(1)
    probe(a.base_url or d["base_url"], key, a.model or d["model_name"])
