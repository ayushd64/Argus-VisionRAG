"""
hallucination_score.py
----------------------
Computes an explicit HALLUCINATION RATE from graded results.

Definition: among the 'unanswerable' gold questions (whose correct
behavior is to REFUSE / say "I don't know"), a hallucination is when the
system instead produced a confident answer — i.e. the judge marked it
WRONG (it fabricated) rather than CORRECT (it refused).

    hallucination_rate = (unanswerable questions marked WRONG)
                         / (total unanswerable questions)

Lower is better. This turns the 'does it make things up?' question into
a single defensible number, from data the judge already produced.
"""

import sys
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def hallucination_rate(graded_file: str) -> dict:
    """Compute the hallucination rate from one config's graded results."""
    path = RESULTS_DIR / graded_file
    with open(path, encoding="utf-8") as f:
        items = json.load(f)

    # Only the 'unanswerable' questions test hallucination.
    unanswerable = [i for i in items if i["category"] == "unanswerable"]
    total = len(unanswerable)

    # A hallucination = an unanswerable question the system got WRONG
    # (it answered instead of refusing). CORRECT here means it refused.
    hallucinated = [i for i in unanswerable if i["verdict"] == "WRONG"]
    n_hall = len(hallucinated)

    rate = (n_hall / total * 100) if total else 0.0

    return {
        "config": graded_file.replace("_graded.json", ""),
        "unanswerable_total": total,
        "hallucinated": n_hall,
        "correctly_refused": total - n_hall,
        "hallucination_rate_pct": round(rate, 1),
        "hallucinated_questions": [i["question"] for i in hallucinated],
    }


if __name__ == "__main__":
    # Compare hallucination rate across the configs we graded.
    configs = [
        "baseline_no_selfcorrect_graded.json",
        "full_selfcorrect_graded.json",
    ]

    summary = {}
    for cfg in configs:
        path = RESULTS_DIR / cfg
        if not path.exists():
            print(f"Skipping {cfg} (not found — run run_judge.py first)")
            continue
        result = hallucination_rate(cfg)
        summary[result["config"]] = result
        print(f"\n=== {result['config']} ===")
        print(f"  Unanswerable questions:  {result['unanswerable_total']}")
        print(f"  Correctly refused:       {result['correctly_refused']}")
        print(f"  Hallucinated (answered): {result['hallucinated']}")
        print(f"  HALLUCINATION RATE:      {result['hallucination_rate_pct']}%")
        if result["hallucinated_questions"]:
            print("  Fabricated on:")
            for q in result["hallucinated_questions"]:
                print(f"    - {q}")

    # Save for the dashboard.
    with open(RESULTS_DIR / "hallucination_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {RESULTS_DIR / 'hallucination_summary.json'}")
