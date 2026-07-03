"""
retrieval_metrics.py
--------------------
Computes retrieval-quality metrics WITHOUT RAGAS (which has fragile
langchain imports). Uses our own NVIDIA judge to score, per question:

  context_precision : of the retrieved chunks, what fraction were
                      actually relevant to answering the question?
                      (Are we retrieving junk?)

  context_recall    : did the retrieved chunks contain the information
                      needed to produce the gold answer?
                      (Are we missing needed info?)

These are the two metrics RAGAS was going to add. We compute them by
asking the judge focused yes/no questions about each chunk — the same
LLM-as-judge pattern used elsewhere. Reads the saved contexts from
eval/results/<config>.json.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from openai import OpenAI

from config import JUDGE_MODEL_NAME, NVIDIA_BASE_URL

load_dotenv()
RESULTS_DIR = Path(__file__).parent / "results"

_client = OpenAI(
    base_url=NVIDIA_BASE_URL,
    api_key=os.environ["NVIDIA_API_KEY"],
    timeout=90,
    max_retries=2,
)


def _judge_yes_no(prompt: str) -> bool:
    """Ask the judge a yes/no question; return True for YES."""
    resp = _client.chat.completions.create(
        model=JUDGE_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=5,
        temperature=0,
    )
    return "YES" in resp.choices[0].message.content.strip().upper()


def context_precision(question: str, contexts: list[str]) -> float:
    """Fraction of retrieved chunks that are relevant to the question."""
    if not contexts:
        return 0.0
    relevant = 0
    for chunk in contexts:
        prompt = (
            f"Question: {question}\n\n"
            f"Passage: {chunk}\n\n"
            f"Is this passage relevant/useful for answering the question? "
            f"Reply YES or NO."
        )
        if _judge_yes_no(prompt):
            relevant += 1
    return relevant / len(contexts)


def context_recall(gold_answer: str, contexts: list[str]) -> float:
    """
    Does the retrieved context contain the info needed for the gold answer?

    We ask the judge to focus on whether the SPECIFIC facts/values in the
    gold answer appear in the context — even if surrounded by other data
    (e.g. a target number inside a large table). The previous version was
    too blunt and scored 'NO' when a value was present but buried among
    many similar numbers.
    """
    if not contexts:
        return 0.0
    combined = "\n\n".join(contexts)
    prompt = (
        f"Retrieved context:\n{combined}\n\n"
        f"Reference answer: {gold_answer}\n\n"
        f"Does the retrieved context contain the specific fact(s) or "
        f"value(s) stated in the reference answer? The information counts "
        f"as present even if it appears inside a table or alongside other "
        f"data. Focus only on whether the key fact/number is findable in "
        f"the context.\n\n"
        f"Reply YES or NO."
    )
    return 1.0 if _judge_yes_no(prompt) else 0.0



def run(config_name: str) -> dict:
    with open(RESULTS_DIR / f"{config_name}.json", encoding="utf-8") as f:
        items = json.load(f)

    print(f"\nComputing retrieval metrics for {config_name} "
          f"({len(items)} questions)...")

    precisions, recalls = [], []
    answerable_p, answerable_r = [], []   # excludes 'unanswerable'

    for i in items:
        p = context_precision(i["question"], i["contexts"])
        r = context_recall(i["gold_answer"], i["contexts"])
        precisions.append(p)
        recalls.append(r)
        # Retrieval metrics only make sense where there IS something to
        # retrieve — exclude unanswerable questions from the "real" score.
        if i["category"] != "unanswerable":
            answerable_p.append(p)
            answerable_r.append(r)
        print(f"  Q{i['id']:>2} [{i['category']:12}]: precision={p:.2f}  recall={r:.2f}")

    result = {
        "config": config_name,
        # Headline: metrics on answerable questions (the meaningful ones).
        "context_precision": round(sum(answerable_p) / len(answerable_p), 3),
        "context_recall": round(sum(answerable_r) / len(answerable_r), 3),
        # Also keep the all-questions version for transparency.
        "context_precision_all": round(sum(precisions) / len(precisions), 3),
        "context_recall_all": round(sum(recalls) / len(recalls), 3),
        "num_answerable": len(answerable_p),
        "num_total": len(items),
    }
    print(f"\n--- Retrieval metrics: {config_name} ---")
    print(f"  Context precision (answerable): {result['context_precision']}")
    print(f"  Context recall (answerable):    {result['context_recall']}")
    print(f"  (all-questions: precision {result['context_precision_all']}, "
          f"recall {result['context_recall_all']})")

    with open(RESULTS_DIR / f"retrieval_{config_name}.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> retrieval_{config_name}.json")
    return result



if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "full_selfcorrect"
    run(config_name)
