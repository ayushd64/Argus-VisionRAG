"""
run_ragas.py — robust version with retries and per-metric coverage.
Adds: retry wrapper on the LLM, a RunConfig with higher timeouts/retries,
and running metrics one at a time so one flaky metric can't blank others.
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
from datasets import Dataset

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from config import JUDGE_MODEL_NAME, NVIDIA_BASE_URL, EMBEDDING_MODEL_NAME

load_dotenv()
RESULTS_DIR = Path(__file__).parent / "results"


def build_llm_and_embeddings():
    judge_llm = ChatOpenAI(
        model=JUDGE_MODEL_NAME,
        base_url=NVIDIA_BASE_URL,
        api_key=os.environ["NVIDIA_API_KEY"],
        temperature=0,
        timeout=180,        # generous timeout — free tier can be slow
        max_retries=4,      # retry on transient failures at the client level
    )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return LangchainLLMWrapper(judge_llm), LangchainEmbeddingsWrapper(embeddings)


def score_one_metric(dataset, metric, llm, embeddings, run_config):
    """
    Run ONE metric at a time. If a whole metric errors out, we catch it and
    return an empty result rather than crashing the entire evaluation.
    Running metrics separately means a flaky metric can't blank the others.
    """
    name = metric.name
    print(f"\n  scoring '{name}'...")
    try:
        result = evaluate(
            dataset,
            metrics=[metric],
            llm=llm,
            embeddings=embeddings,
            run_config=run_config,
            raise_exceptions=False,   # don't crash on a single bad row
        )
        df = result.to_pandas()
        col = [c for c in df.columns if c == name or name in c]
        if not col:
            return name, None, 0, len(df)
        series = df[col[0]].dropna()
        n = len(series)
        total = len(df)
        avg = round(float(series.mean()), 3) if n else None
        print(f"    {name}: {avg}  (scored {n}/{total})")
        return name, avg, n, total
    except Exception as e:
        print(f"    {name} FAILED entirely: {e}")
        return name, None, 0, len(dataset)


def run_ragas(config_name: str) -> dict:
    with open(RESULTS_DIR / f"{config_name}.json", encoding="utf-8") as f:
        items = json.load(f)

    data = {
        "question": [i["question"] for i in items],
        "answer": [i["predicted_answer"] for i in items],
        "contexts": [i["contexts"] for i in items],
        "ground_truth": [i["gold_answer"] for i in items],
    }
    dataset = Dataset.from_dict(data)

    llm, embeddings = build_llm_and_embeddings()

    # RunConfig controls RAGAS's own retry/concurrency behavior.
    # Lower max_workers = fewer concurrent calls = fewer free-tier rate-limit
    # failures. More retries + longer timeout = fewer NaNs.
    run_config = RunConfig(
        timeout=180,
        max_retries=5,
        max_wait=60,
        max_workers=2,      # gentle on the NVIDIA free-tier rate limit
    )

    print(f"\nRunning RAGAS on {config_name} ({len(items)} questions)...")
    print("Running each metric separately for robustness.\n")

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    scores, coverage = {}, {}
    for m in metrics:
        name, avg, n, total = score_one_metric(dataset, m, llm, embeddings, run_config)
        scores[name] = avg
        coverage[name] = f"{n}/{total}"

    print(f"\n--- RAGAS scores for {config_name} ---")
    for name in scores:
        print(f"  {name:20} {scores[name]}   (scored on {coverage[name]})")

    out = {"config": config_name, "scores": scores, "coverage": coverage}
    with open(RESULTS_DIR / f"ragas_{config_name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> ragas_{config_name}.json")
    return out


if __name__ == "__main__":
    config_name = sys.argv[1] if len(sys.argv) > 1 else "full_selfcorrect"
    run_ragas(config_name)

