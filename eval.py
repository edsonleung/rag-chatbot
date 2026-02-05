"""
RAG evaluation using RAGAS v0.4.

Evaluates the RAG pipeline on 4 dimensions:

1. Faithfulness    — Is the answer grounded in retrieved context? (target > 0.8)
2. Answer Relevancy — Does the answer address the question?       (target > 0.8)
3. Context Precision — Are retrieved chunks relevant?              (target > 0.7)
4. Context Recall   — Did we retrieve all needed info?             (target > 0.7)

Flow:
    eval_dataset.json → run each Q through RAG chain → collect
    (question, answer, contexts, ground_truth) → RAGAS evaluate →
    eval_results.json

Usage:
    python eval.py                          # run with default eval_dataset.json
    python eval.py my_custom_dataset.json   # run with custom dataset
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from chain import build_rag_chain_with_context

load_dotenv()

# Score targets — used by the dashboard to color-code results
TARGETS = {
    "faithfulness": 0.8,
    "answer_relevancy": 0.8,
    "context_precision_with_reference": 0.7,
    "context_recall": 0.7,
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model used for RAGAS evaluation (LLM-as-judge).
# Using a different free model from the RAG chain spreads the rate limit
# across two model endpoints and avoids the 50 req/day free-tier cap
# hitting a single model. Google Gemma 3 27B is strong at evaluation tasks.
EVAL_MODEL = "google/gemma-3-27b-it:free"


def _retry_with_backoff(fn, max_retries: int = 8, base_delay: float = 15.0):
    """
    Retry a function with exponential backoff for rate limit errors.

    Free-tier OpenRouter limits: ~20 requests/min, 50 requests/day
    (1000/day if you've bought ≥$10 in credits). This retries with
    increasing delays so the eval can complete without manual intervention.
    """
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            error_str = str(e).lower()
            if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                delay = base_delay * (2 ** attempt)
                print(f"    ⏳ Rate limited. Waiting {delay:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting.")


def load_eval_dataset(path: str = "eval_dataset.json") -> list[dict]:
    """
    Load evaluation questions from a JSON file.

    Expected format:
    [
        {"question": "...", "ground_truth": "..."},
        ...
    ]
    """
    with open(path) as f:
        return json.load(f)


def collect_rag_responses(test_questions: list[dict]) -> list[dict]:
    """
    Run each question through the RAG pipeline and collect results.

    For each question, we capture:
    - user_input: the original question
    - response: the LLM's generated answer
    - retrieved_contexts: the exact chunks used to generate the answer
    - reference: the ground truth answer (for context recall)

    Uses build_rag_chain_with_context() so retrieval + generation
    happen in a single pass (same contexts that produced the answer).
    """
    chain_fn = build_rag_chain_with_context()
    results = []

    for i, item in enumerate(test_questions):
        question = item["question"]
        print(f"  [{i + 1}/{len(test_questions)}] {question}")

        # Retry with backoff for free-tier rate limits
        response = _retry_with_backoff(lambda q=question: chain_fn(q))

        results.append({
            "user_input": question,
            "response": response["answer"],
            "retrieved_contexts": response["contexts"],
            "reference": item["ground_truth"],
        })

        # Throttle between questions to stay under 20 req/min free-tier limit
        if i < len(test_questions) - 1:
            print("    💤 Waiting 5s between questions (free-tier throttle)...")
            time.sleep(5)

    return results


def build_ragas_llm():
    """
    Create an LLM wrapper for RAGAS evaluation.

    RAGAS uses an LLM internally to judge faithfulness, relevancy, etc.
    We use the same OpenRouter free model as our RAG chain.
    """
    llm = ChatOpenAI(
        model=EVAL_MODEL,
        temperature=0.0,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=OPENROUTER_BASE_URL,
    )
    return LangchainLLMWrapper(llm)


def build_ragas_embeddings():
    """
    Create an embeddings wrapper for RAGAS evaluation.

    Some metrics (like answer relevancy) need an embedding model
    to compute semantic similarity. We reuse the same local model.
    """
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(embeddings)


def run_evaluation(
    dataset_path: str = "eval_dataset.json",
    output_path: str = "eval_results.json",
) -> dict:
    """
    Full evaluation pipeline: load → query → evaluate → save.

    Steps:
    1. Load test questions from JSON
    2. Run each through the RAG chain (collect answer + contexts)
    3. Build RAGAS EvaluationDataset
    4. Run RAGAS evaluate() with 4 metrics
    5. Save results to JSON (both aggregate and per-question)

    Returns the results dict for programmatic use.
    """
    # 1. Load test questions
    print(f"Loading eval dataset from {dataset_path}...")
    test_questions = load_eval_dataset(dataset_path)
    print(f"  Found {len(test_questions)} questions\n")

    # 2. Run RAG chain on each question
    print("Running RAG pipeline on eval questions...")
    rag_results = collect_rag_responses(test_questions)
    print()

    # 3. Build RAGAS dataset
    print("Building RAGAS evaluation dataset...")
    samples = [
        SingleTurnSample(
            user_input=r["user_input"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in rag_results
    ]
    eval_dataset = EvaluationDataset(samples=samples)

    # 4. Configure evaluator LLM and metrics
    evaluator_llm = build_ragas_llm()
    evaluator_embeddings = build_ragas_embeddings()

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecisionWithReference(),
        ContextRecall(),
    ]

    # 5. Run RAGAS evaluation
    print("Running RAGAS evaluation (this calls the LLM for each metric)...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        show_progress=True,
    )

    # 6. Extract and format results
    aggregate_scores = {k: float(v) for k, v in result.scores.items()}

    # Per-question results from the result DataFrame
    per_question = []
    df = result.to_pandas()
    for _, row in df.iterrows():
        per_question.append({
            "question": row["user_input"],
            "answer": row["response"],
            "contexts": row["retrieved_contexts"],
            "reference": row["reference"],
            "scores": {
                col: float(row[col]) if not _is_nan(row[col]) else None
                for col in aggregate_scores.keys()
                if col in df.columns
            },
        })

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_path,
            "num_questions": len(test_questions),
            "model_rag": "google/gemma-3-27b-it:free",
            "model_eval": EVAL_MODEL,
        },
        "targets": TARGETS,
        "aggregate_scores": aggregate_scores,
        "per_question": per_question,
    }

    # 7. Save to JSON
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # 8. Print summary
    _print_summary(aggregate_scores)

    return output


def _is_nan(value) -> bool:
    """Check if a value is NaN (handles both float and numpy NaN)."""
    try:
        return value != value  # NaN != NaN is True
    except (TypeError, ValueError):
        return False


def _print_summary(scores: dict) -> None:
    """Print a formatted summary table to console."""
    print("\n" + "=" * 55)
    print("  RAGAS EVALUATION RESULTS")
    print("=" * 55)
    print(f"  {'Metric':<40} {'Score':>6}  {'Target':>6}")
    print("-" * 55)
    for metric, score in scores.items():
        target = TARGETS.get(metric, "N/A")
        status = "PASS" if score >= target else "FAIL"
        target_str = f"{target:.1f}" if isinstance(target, float) else target
        score_str = f"{score:.4f}" if score is not None else "N/A"
        print(f"  {metric:<40} {score_str:>6}  {target_str:>6}  {status}")
    print("=" * 55)


if __name__ == "__main__":
    dataset_file = sys.argv[1] if len(sys.argv) > 1 else "eval_dataset.json"

    if not Path(dataset_file).exists():
        print(f"Error: {dataset_file} not found.")
        print("Create an eval dataset first, or run with: python eval.py <path>")
        sys.exit(1)

    run_evaluation(dataset_path=dataset_file)
