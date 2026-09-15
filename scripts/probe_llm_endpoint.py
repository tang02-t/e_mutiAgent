"""探测 LLM 端点：依次尝试多种凭证，检查普通对话与 function calling 是否可用。"""
import json
import sys

from openai import OpenAI

BASE_URL = "https://llm-8hwhlldwix34tnyh.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
MODEL = "qwen3.7-flash"

CANDIDATES = {
    "ak_secret": "REDACTED_SECRET",
    "ak_id": "REDACTED_AK_ID",
    "ak_id_colon_secret": "REDACTED_AK_ID:REDACTED_SECRET",
    "old_sk": "sk-4c11f48458704864a9e62b6841d2235d",
}

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


def probe(name: str, key: str) -> bool:
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=60)
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "只回复两个字：成功"}],
            extra_body={"enable_thinking": False},
            max_tokens=16,
        )
        print(f"[{name}] chat OK -> {r.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] chat FAIL -> {type(e).__name__}: {str(e)[:200]}")
        return False
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "H2=150, C2H2=12，请调用工具做故障归因"}],
            tools=TOOLS,
            tool_choice="auto",
            extra_body={"enable_thinking": False},
            max_tokens=256,
        )
        tc = r.choices[0].message.tool_calls
        print(f"[{name}] tools OK -> {json.dumps([t.function.model_dump() for t in tc] if tc else None, ensure_ascii=False)}")
    except Exception as e:  # noqa: BLE001
        print(f"[{name}] tools FAIL -> {type(e).__name__}: {str(e)[:200]}")
    return True


if __name__ == "__main__":
    only = sys.argv[1:] or list(CANDIDATES)
    for n in only:
        if probe(n, CANDIDATES[n]):
            break
