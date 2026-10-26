from regression_guard.prompts import list_prompts, load_prompt
from regression_guard.schemas import Category


def test_v1_loads_and_is_valid():
    p = load_prompt("classifier_v1")
    assert p.version == "v1"
    assert p.temperature == 0
    assert len(p.few_shot_examples) == 2
    assert p.few_shot_examples[0].category == Category.BILLING


def test_list_prompts_finds_v1():
    assert "classifier_v1" in list_prompts()
