import json
import re
from pathlib import Path

# POI categories from production config
POI_CATEGORIES = [
    "음식점", "카페/디저트", "교통/모빌리티", "여행/숙박", "레저/스포츠",
    "문화/예술", "자기계발", "반려동물", "쇼핑", "의료", "미용",
]

PREDICTION_SYSTEM = "너는 사용자 프로필을 분석하여 향후 방문할 POI 카테고리를 예측하는 분석가야. 반드시 JSON으로만 응답해."

PREDICTION_PROMPT = """주어진 사용자 메모리/프로필을 분석하여, 이 사용자가 향후 14일간 방문할 가능성이 높은
POI 카테고리를 상위 10개까지 순위를 매겨 예측하세요.

[사용자 메모리]
{memory_context}

[POI 카테고리 목록]
{category_list}

정확히 10개의 카테고리를 예측하세요. 10개 미만이면 가능성 순으로 채우고, 10개를 초과하지 마세요.
JSON 형식으로 응답: {{"predictions": ["카테고리1", "카테고리2", ...]}}"""


def _parse_predictions(raw: str) -> list[str] | None:
    """Extract predictions list from LLM response string.

    Tries json.loads first, then regex fallback for malformed responses.
    Returns None if parsing fails entirely.
    """
    cleaned = raw.strip()

    # Try direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "predictions" in data:
            return data["predictions"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting JSON object with regex
    obj_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if obj_match:
        try:
            data = json.loads(obj_match.group(0))
            if isinstance(data, dict) and "predictions" in data:
                return data["predictions"]
        except (json.JSONDecodeError, TypeError):
            pass

    # Try extracting bare JSON array
    arr_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if arr_match:
        try:
            items = json.loads(arr_match.group(0))
            if isinstance(items, list):
                return items
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _validate_and_pad(predictions: list[str]) -> list[str]:
    """Filter to valid categories, truncate to 10, pad if needed."""
    valid_set = set(POI_CATEGORIES)
    seen = set()
    result = []

    for item in predictions:
        if isinstance(item, str) and item in valid_set and item not in seen:
            seen.add(item)
            result.append(item)
        if len(result) == 10:
            return result

    # Pad with remaining categories in default order
    for cat in POI_CATEGORIES:
        if cat not in seen:
            result.append(cat)
            if len(result) == 10:
                break

    return result


def predict_categories(llm_client, memory_context: str) -> list[str]:
    """Call LLM to predict POI categories from memory context.

    Uses the unified prediction prompt with temperature=0.0 for reproducibility.
    Accesses the underlying OpenAI client directly to set temperature.

    Args:
        llm_client: Production LLMClient instance (uses llm_client.client for direct API calls).
        memory_context: Formatted memory text from constructor's get_memory_context().

    Returns:
        Ordered list of exactly 10 valid POI category strings (rank 1 first).
    """
    category_list = ", ".join(POI_CATEGORIES)
    user_prompt = PREDICTION_PROMPT.format(
        memory_context=memory_context,
        category_list=category_list,
    )

    response = llm_client.client.chat.completions.create(
        model=llm_client.model,
        messages=[
            {"role": "system", "content": PREDICTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw_response = response.choices[0].message.content.strip()
    predictions = _parse_predictions(raw_response)

    if predictions is None:
        return POI_CATEGORIES[:10]

    return _validate_and_pad(predictions)
