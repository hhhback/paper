import json
from pathlib import Path

import tiktoken

from experiments.config import ExperimentConfig
from experiments.constructors.base import AbstractConstructor

_SYSTEM_PROMPT = (
    "사용자 interaction 데이터에서 핵심 사실과 선호를 추출하는 분석가입니다. "
    "JSON으로만 응답하세요."
)

_USER_PROMPT_TEMPLATE = (
    "다음 사용자 interaction에서 주요 사실과 선호를 key-value 쌍으로 추출하세요.\n\n"
    "[interaction data]\n{merged_text}\n\n"
    'JSON 형식: {{"facts": [{{"key": "카테고리", "value": "설명"}}, ...]}}\n'
    "최대 30개까지 추출하세요."
)


class FlatMemoryConstructor(AbstractConstructor):
    def __init__(self, config: ExperimentConfig, llm_client):
        self.config = config
        self._llm = llm_client
        self._enc = tiktoken.get_encoding("cl100k_base")

    def build_memory(self, user_id: str, daily_texts: dict[str, str], memory_dir: Path) -> None:
        merged = "\n".join(daily_texts[d] for d in sorted(daily_texts))
        user_prompt = _USER_PROMPT_TEMPLATE.format(merged_text=merged)

        raw = self._llm._chat_json(_SYSTEM_PROMPT, user_prompt)
        parsed = json.loads(raw)
        facts = parsed.get("facts", [])

        user_dir = memory_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        (user_dir / "flat_memory.json").write_text(
            json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get_memory_context(self, user_id: str, memory_dir: Path) -> str:
        path = memory_dir / user_id / "flat_memory.json"
        if not path.exists():
            return ""

        facts = json.loads(path.read_text(encoding="utf-8"))
        lines = [f"- {item['key']}: {item['value']}" for item in facts]
        text = "\n".join(lines)

        tokens = self._enc.encode(text)
        if len(tokens) > self.config.context_budget:
            text = self._enc.decode(tokens[: self.config.context_budget])

        return text
