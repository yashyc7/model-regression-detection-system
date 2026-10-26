"""The llm feature under test"""

import json
import time

from openai import AsyncOpenAI
from pydantic import ValidationError

from regression_guard.config import get_settings
from regression_guard.schemas import (
    ClassificationOutcome,
    ClassificationResult,
    PromptConfig,
)


def make_client() -> AsyncOpenAI:
    s = get_settings()
    return AsyncOpenAI(api_key=s.openai_api_key, base_url=s.openai_base_url)


def build_messages(email: str, prompt: PromptConfig) -> list[dict]:
    """configure structured messages which we have to sent to the llm so that classification works fine"""
    messages = []
    messages.append({"role": "system", "content": prompt.system_prompt})

    # now also giving some examples to llm

    for ex in prompt.few_shot_examples:
        messages.append({"role": "user", "content": ex.email})
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"category": ex.category.value, "summary": ex.summary}
                ),
            }
        )
    messages.append({"role": "user", "content": email})
    return messages


async def classify_email(
    email: str, prompt: PromptConfig, client: AsyncOpenAI
) -> ClassificationOutcome:
    """Run one email through the LLM. Never raises: failures become data."""
    start = time.perf_counter()
    try:
        response = await client.chat.completions.create(
            model=prompt.model,
            temperature=prompt.temperature,
            messages=build_messages(email, prompt),
            response_format={"type": "json_object"},
        )
    except Exception as e:  # network error, rate limit, bad key, etc.
        return ClassificationOutcome(
            error=f"API error: {e}",
            latency_ms=(time.perf_counter() - start) * 1000,
        )

    latency_ms = (time.perf_counter() - start) * 1000
    raw = response.choices[0].message.content or ""
    usage = response.usage
    outcome = ClassificationOutcome(
        raw_output=raw,
        latency_ms=latency_ms,
        input_tokens=usage.prompt_tokens if usage else 0,
        output_tokens=usage.completion_tokens if usage else 0,
    )
    try:
        outcome.result = ClassificationResult.model_validate_json(raw)
    except ValidationError as e:
        outcome.error = f"Bad output format: {e.errors()[0]['msg']}"
    return outcome
