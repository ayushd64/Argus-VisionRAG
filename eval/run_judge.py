"""
run_judge.py
------------
The LLM-AS-JUDGE. Scores each predicted answer against its gold answer
using a strong, independent model (NVIDIA-hosted 70B) — deliberately
bigger than the 3B that produced the answers, so the grading isn't
circular.

For each answer the judge returns one of three verdicts:
  CORRECT  - matches the gold answer's key facts.
  PARTIAL  - partially right (some facts right, some missing/wrong).
  WRONG    - incorrect, or fabricated, or missing the point.

Special handling for 'unanswerable' questions: the gold answer is a
refusal, so "CORRECT" means the prediction ALSO refused. This is how we
measure whether Argus correctly declines instead of hallucinating.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from config import JUDGE_MODEL_NAME, NVIDIA_BASE_URL

load_dotenv()   # read NVIDIA_API_KEY from .env

RESULTS_DIR = Path(__file__).parent / "results"

# One shared client pointed at NVIDIA's OpenAI-compatible endpoint.
_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=os.environ["NVIDIA_API_KEY"],
    timeout=90.0,        # big model can be slow on the free tier
    max_retries=2,
)

_JUDGE_PROMPT = """You are grading a RAG system's answer against a known \
correct answer.

Question: {question}

Correct answer (gold): {gold}

System's answer: {predicted}

Grade the system's answer:
- CORRECT: it conveys the same key facts as the gold answer (wording may differ).
- PARTIAL: some key facts right, but incomplete or with a minor error.
- WRONG: incorrect, fabricated, or misses the point.

Note: if the gold answer says "I don't know based on the provided documents",
then the system is CORRECT only if it ALSO declines / says it doesn't know.

Reply with ONLY one word: CORRECT, PARTIAL, or WRONG."""


def judge_one(item: dict) -> str:
    """Ask the judge to grade a single predicted answer."""
    prompt = _JUDGE_PROMPT.format(
        question=item["question"],
        gold=item["gold_answer"],
        predicted=item["predicted_answer"],
    )
    resp = _client.chat.completions.create(
        model=JUDGE_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0,     # deterministic grading
    )
    verdict = resp.choices[0].message.content.strip().upper()

    # Parse forgivingly, same defensive habit as your grade/ground nodes.
    if "CORRECT" in verdict:
        return "CORRECT"
    if "PARTIAL" in verdict:
        return "PARTIAL"
    return "WRONG"


def judge_config(config_name: str) -> dict:
    """Judge every answer in one config's results file; return a summary."""
    path = RESULTS_DIR / f"{config_name}.json"
    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    print(f"\nJudging {config_name} ({len(items)} answers)...")
    for item in items:
        verdict = judge_one(item)
        item["verdict"] = verdict
        print(f"  Q{item['id']:>2} [{item['category']:12}] -> {verdict}")

    # Save the graded results back (now with verdicts attached).
    graded_path = RESULTS_DIR / f"{config_name}_graded.json"
    with open(graded_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    # Build a quick summary: counts overall and per category.
    summary = {"config": config_name, "total": len(items),
               "CORRECT": 0, "PARTIAL": 0, "WRONG": 0, "by_category": {}}
    for item in items:
        v = item["verdict"]
        summary[v] += 1
        cat = item["category"]
        summary["by_category"].setdefault(cat, {"CORRECT": 0, "PARTIAL": 0, "WRONG": 0})
        summary["by_category"][cat][v] += 1
    return summary


if __name__ == "__main__":
    import sys
    # Which comparison to judge: pass "selfcorrect" or "vlm" as an argument.
    comparison = sys.argv[1] if len(sys.argv) > 1 else "selfcorrect"

    if comparison == "vlm":
        configs = ["vlm_off_text_only", "vlm_on_full"]
        out_name = "summary_vlm.json"
    else:
        configs = ["baseline_no_selfcorrect", "full_selfcorrect"]
        out_name = "summary_selfcorrect.json"

    summaries = {}
    for name in configs:
        summaries[name] = judge_config(name)

    with open(RESULTS_DIR / out_name, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {RESULTS_DIR / out_name}")



