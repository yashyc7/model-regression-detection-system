import yaml
from regression_guard.config import PROJECT_ROOT
from regression_guard.schemas import PromptConfig

PROMPTS_DIR = PROJECT_ROOT / "prompts"


def load_prompt(name: str) -> PromptConfig:
    """Load e.g. 'classifier_v1' from prompts/classifier_v1.yaml."""
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PromptConfig.model_validate(data)  # safely validate data here and


def list_prompts() -> list[str]:
    return sorted(p.stem for p in PROMPTS_DIR.glob("*.yaml"))
