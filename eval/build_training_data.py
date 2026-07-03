"""
build_training_data.py  (runs in Windows, in the Argus project)
----------------------------------------------------------------
Generates DRAFT training examples for the LoRA fine-tune by pulling:
  - questions + gold answers from the gold set
  - retrieved context from the actual FAISS index

Each example = {question, context, answer}. The gold answer is used as
the target 'ideal answer' (it's already concise/correct). The context is
what retrieval actually returns, so the model learns on realistic inputs.

Output: eval/results/training_examples.json  (copy this into WSL)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from vector_store import search
from config import TOP_K
from gold_questions import GOLD_QUESTIONS

RESULTS_DIR = Path(__file__).parent / "results"


def build():
    examples = []
    for q in GOLD_QUESTIONS:
        # Get the real retrieved context for this question.
        chunks = search(q["question"], top_k=TOP_K)
        context = "\n\n".join(
            f"[{c['source']} p.{c['page']}] {c['text']}" for c in chunks
        )

        # Build the IDEAL answer: gold answer + a citation (or refusal).
        if q["category"] == "unanswerable":
            answer = "I don't know based on the provided documents."
        else:
            # Attach a source citation from the top retrieved chunk.
            top = chunks[0] if chunks else None
            cite = f" (source, p.{top['page']})" if top else ""
            answer = q["answer"].rstrip(".") + "." + cite

        examples.append({
            "question": q["question"],
            "context": context,
            "answer": answer,
        })
        print(f"  Q{q['id']}: built example")

    out_path = RESULTS_DIR / "training_examples.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(examples)} examples -> {out_path}")


if __name__ == "__main__":
    build()

