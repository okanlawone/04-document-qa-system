"""Compare two small retrieval methods against hand-labeled questions."""

import json
from pathlib import Path

from main import DATA_PATH, load_documents, search_documents, search_documents_tfidf


QUESTIONS_PATH = Path(__file__).parent / "data" / "evaluation_questions.json"
TOP_K = 3


def load_questions(path):
    """Read questions and their expected evidence labels from JSON."""
    with path.open(encoding="utf-8") as file:
        questions = json.load(file)

    required_fields = {"id", "question", "expected_source_ids", "answerable"}
    for question in questions:
        missing_fields = required_fields - question.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Question {question!r} is missing: {missing}")
        if question["answerable"] != bool(question["expected_source_ids"]):
            raise ValueError(
                f"Question {question['id']} must have expected source IDs exactly "
                "when answerable is true."
            )
    return questions


def reciprocal_rank(ranked_ids, expected_ids):
    """Return 1/rank for the first expected source, or 0 when it is absent."""
    for rank, document_id in enumerate(ranked_ids, start=1):
        if document_id in expected_ids:
            return 1 / rank
    return 0.0


def evaluate_method(name, search, questions, documents):
    """Print question-level results and simple retrieval measures."""
    rows = []
    answerable_rows = []
    unanswerable_rows = []

    for item in questions:
        results = search(item["question"], documents, limit=TOP_K)
        ranked_ids = [document["id"] for _, document in results]
        expected_ids = item["expected_source_ids"]
        hit = any(document_id in expected_ids for document_id in ranked_ids)
        top_one_hit = bool(ranked_ids and ranked_ids[0] in expected_ids)
        row = {
            "item": item,
            "ranked_ids": ranked_ids,
            "top_one_hit": top_one_hit,
            "hit_at_k": hit,
        }
        rows.append(row)
        if item["answerable"]:
            answerable_rows.append(row)
        else:
            unanswerable_rows.append(row)

    print(f"\n{name}")
    print("ID   Expected   Ranked in top 3   Top-1   Hit@3")
    for row in rows:
        item = row["item"]
        expected = ",".join(item["expected_source_ids"]) or "none"
        ranked = ",".join(row["ranked_ids"]) or "no result"
        print(
            f"{item['id']:<4} {expected:<10} {ranked:<17} "
            f"{str(row['top_one_hit']):<6} {row['hit_at_k']}"
        )

    answerable_count = len(answerable_rows)
    top_one = sum(row["top_one_hit"] for row in answerable_rows) / answerable_count
    hit_at_k = sum(row["hit_at_k"] for row in answerable_rows) / answerable_count
    mean_reciprocal_rank = sum(
        reciprocal_rank(row["ranked_ids"], row["item"]["expected_source_ids"])
        for row in answerable_rows
    ) / answerable_count
    unanswerable_match_rate = sum(bool(row["ranked_ids"]) for row in unanswerable_rows) / len(
        unanswerable_rows
    )

    print(f"Answerable questions: {answerable_count}")
    print(f"Top-1 accuracy: {top_one:.1%}")
    print(f"Expected-source recall@{TOP_K}: {hit_at_k:.1%}")
    print(f"Mean reciprocal rank: {mean_reciprocal_rank:.3f}")
    print(f"Unanswerable questions receiving any result: {unanswerable_match_rate:.1%}")


def main():
    documents = load_documents(DATA_PATH)
    questions = load_questions(QUESTIONS_PATH)
    evaluate_method("Raw word-count cosine", search_documents, questions, documents)
    evaluate_method("TF-IDF cosine", search_documents_tfidf, questions, documents)
    print(
        "\nThese numbers describe only this tiny, hand-written practice set. "
        "A returned result is not proof that a question is answerable."
    )


if __name__ == "__main__":
    main()
