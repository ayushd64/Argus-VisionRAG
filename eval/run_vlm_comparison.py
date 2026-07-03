"""
run_vlm_comparison.py
---------------------
Runs the gold set against BOTH indexes (full vs text_only) and saves
each config's answers — the VLM on/off before/after.

We keep self-correction OFF here so we isolate the VLM's effect: any
difference in scores is attributable to the visual layer alone, not the
loop. (Clean experiment design — change one variable at a time.)
"""

import sys
import json
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import config
from gold_questions import GOLD_QUESTIONS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_against_index(config_name: str, variant: str) -> None:
    """Point at one index variant, run the gold set, save answers."""
    print(f"\n=== {config_name}: querying '{variant}' index ===")

    # Isolate the VLM variable: self-correction OFF, pick the index.
    config.ENABLE_SELF_CORRECTION = False
    config.INDEX_VARIANT = variant
    config.INDEX_DIR = config.DATA_DIR / f"processed_{variant}"

    # Reload the modules that read INDEX_DIR / the flag, so they follow.
    import vector_store; importlib.reload(vector_store)
    import rag_pipeline; importlib.reload(rag_pipeline)

    results = []
    for q in GOLD_QUESTIONS:
        print(f"  Q{q['id']}: {q['question'][:45]}...")
        out = rag_pipeline.answer_question(q["question"])
        results.append({
            "id": q["id"], "category": q["category"],
            "question": q["question"], "gold_answer": q["answer"],
            "predicted_answer": out["answer"],
            "contexts": [c["text"] for c in out["sources"]],
        })

    out_path = RESULTS_DIR / f"{config_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    run_against_index("vlm_off_text_only", variant="text_only")
    run_against_index("vlm_on_full",       variant="full")
    print("\nDone. Now run the judge on these two configs.")

