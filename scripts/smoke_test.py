import asyncio

from regression_guard.classifier import classify_email, make_client
from regression_guard.prompts import load_prompt

EMAILS = [
    "I can't log in, it says my password is wrong even after I reset it.",
    "Why was my card charged 4999 rupees when my plan costs 999?",
    "Love the new dashboard! Do you have a referral programme?",
]


async def main() -> None:
    prompt = load_prompt("classifier_v1")
    client = make_client()
    print(f"Prompt {prompt.version} | model {prompt.model}\n")
    for email in EMAILS:
        out = await classify_email(email, prompt, client)
        print("EMAIL  :", email)
        if out.result:
            print("RESULT :", out.result.category.value, "|", out.result.summary)
        else:
            print("ERROR  :", out.error)
        print(
            f"STATS  : {out.latency_ms:.0f} ms, {out.input_tokens} in / {out.output_tokens} out\n"
        )


asyncio.run(main())
