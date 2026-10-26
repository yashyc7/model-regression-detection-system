from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassificationResult(BaseModel):
    category: Category
    summary: str = Field(min_length=1, description="One sentenced summary")


class ClassificationOutcome(BaseModel):
    result: ClassificationResult | None = None
    error: str | None = None
    raw_output: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class FewShotExample(BaseModel):
    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    """One versioned prompt . This is the code our CI will test"""

    version: str
    created_at: datetime
    description: str = ""
    model: str = "deepseek-ai/DeepSeek-V4-Flash"
    temperature: float = 0.0
    system_prompt: str
    few_shot_examples: list[FewShotExample] = []
