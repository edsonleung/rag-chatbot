"""
RAG evaluation using RAGAS.

RAGAS (Retrieval Augmented Generation Assessment) scores your RAG pipeline
on multiple dimensions. This helps you tune chunking, retrieval k, prompts, etc.

Metrics:
- Faithfulness: Is the answer grounded in the retrieved context?
- Answer Relevancy: Does the answer address the question asked?
- Context Precision: Are retrieved chunks relevant? (no noise)
- Context Recall: Did we retrieve everything needed? (requires ground truth)

Usage:
    python eval.py
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from chain import build_rag_chain
from retriever import get_retriever


def run_evaluation(test_questions: list[dict]) -> dict:
    """
    Evaluate the RAG pipeline on a set of test questions.

    Each item in test_questions should have:
        - "question": the user query
        - "ground_truth": the expected correct answer (for context_recall)

    Returns a dict of metric scores.
    """
    chain = build_rag_chain()
    retriever = get_retriever()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for item in test_questions:
        question = item["question"]

        # Get retrieved docs separately for evaluation
        docs = retriever.invoke(question)
        answer = chain.invoke(question)

        questions.append(question)
        answers.append(answer)
        contexts.append([doc.page_content for doc in docs])
        ground_truths.append(item["ground_truth"])

    # RAGAS expects a HuggingFace Dataset
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    results = evaluate(
        dataset=eval_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    return results


if __name__ == "__main__":
    # Example test set — replace with questions relevant to your PDF
    test_set = [
        {
            "question": "What is the main topic of this document?",
            "ground_truth": "Replace with the actual expected answer.",
        },
        {
            "question": "What are the key findings?",
            "ground_truth": "Replace with the actual expected answer.",
        },
    ]

    print("Running RAGAS evaluation...")
    results = run_evaluation(test_set)
    print("\n--- RAGAS Scores ---")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}")
