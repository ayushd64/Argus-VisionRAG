"""
run_configs.py
--------------
The TOGGLE HARNESS. Runs the gold set through Argus in the four
configurations and saves each config's answers to disk for scoring.

The trick: the VLM and self-correction switches live in config.py, but
config is imported once at startup, so we can't just reassign the
variable and expect downstream modules to notice. Instead, each config
requires the INDEX to match (VLM on/off changes what's indexed) — so we
handle the two switches differently:

  - SELF-CORRECTION is a runtime flag: we flip config.ENABLE_SELF_CORRECTION
    and re-import the pipeline, which reads it fresh.
  - VLM affects the INDEX (built during ingestion), so "VLM off" means
    querying an index built WITHOUT visual chunks. We handle that by
    building two separate indexes once, and pointing at the right one.

For a first pass we keep it simple: we evaluate the TWO configs that
share the current (VLM-on) index — self-correction on vs off — which
already gives you one clean before/after axis. We'll add the VLM on/off
axis (which needs a second index) as a follow-up.
"""

import sys
from pathlib import Path

# eval/ lives outside src/, so add src/ to Python's import path — this is 
# # what lets us `import config`, `import rag_pipeline`, etc. from here. 
sys.path.insert(0, str(Path(__file__).parent.parent / "src")) 
sys.path.insert(0, str(Path(__file__).parent))      # also allow `import gold_questions` 


import json
import importlib
import config
from gold_questions import GOLD_QUESTIONS

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def run_gold_set(config_name: str) -> list[dict]:
    """Run every gold question through the CURRENT pipeline config."""
    # Re-import the pipeline so it picks up the current config flags.
    import rag_pipeline
    importlib.reload(rag_pipeline)

    results = []
    for q in GOLD_QUESTIONS:
        print(f"  [{config_name}] Q{q['id']}: {q['question'][:50]}...")
        out = rag_pipeline.answer_question(q["question"])
        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "gold_answer": q["answer"],
            "predicted_answer": out["answer"],
            # keep the retrieved source texts for RAGAS context metrics later
            "contexts": [c["text"] for c in out["sources"]],
        })
    return results


def run_config(config_name: str, self_correction: bool) -> None:
    """Set the flags for one config, run the gold set, save the results."""
    print(f"\n=== Running config: {config_name} "
          f"(self_correction={self_correction}) ===")

    # Flip the runtime flag on the imported config module.
    config.ENABLE_SELF_CORRECTION = self_correction

    results = run_gold_set(config_name)

    out_path = RESULTS_DIR / f"{config_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    # Two configs on the current (VLM-on) index: the self-correction axis.
    run_config("baseline_no_selfcorrect", self_correction=False)
    run_config("full_selfcorrect",        self_correction=True)

    print("\nDone. Both configs' answers saved to eval/results/.")

