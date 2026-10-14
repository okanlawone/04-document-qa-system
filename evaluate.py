"""Measure retrieval and cautious extractive answers on invented questions."""

import json
from pathlib import Path

from main import (
    DATA_PATH,
    answer_question,
    load_documents,
    search_documents,
    search_documents_tfidf,
)


QUESTIONS_PATH = Path(__file__).parent / "data" / "evaluation_questions.json"
TOP_K = 3


def load_questions(path):
    """Read questions and their hand-written expected evidence labels from JSON."""
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
        if question["answerable"] and not question.get("expected_evidence"):
            raise ValueError(f"Answerable question {question['id']} needs expected_evidence.")
    return questions


def reciprocal_rank(ranked_ids, expected_ids):
    """Return 1/rank for the first expected source, or 0 when it is absent."""
    for rank, document_id in enumerate(ranked_ids, start=1):
        if document_id in expected_ids:
            return 1 / rank
    return 0.0


def evaluate_retrieval(name, search, questions, documents):
    """Retain the earlier ranking measures so the new layer has context."""
    answerable_rows = []
    unanswerable_rows = []
    for item in questions:
        results = search(item["question"], documents, limit=TOP_K)
        ranked_ids = [document["id"] for _, document in results]
        row = {
            "item": item,
            "ranked_ids": ranked_ids,
            "top_one_hit": bool(ranked_ids and ranked_ids[0] in item["expected_source_ids"]),
            "hit_at_k": any(document_id in item["expected_source_ids"] for document_id in ranked_ids),
        }
        (answerable_rows if item["answerable"] else unanswerable_rows).append(row)

    print(f"\n{name} retrieval, top {TOP_K}")
    print("ID   Expected   Ranked sources")
    for row in answerable_rows + unanswerable_rows:
        item = row["item"]
        expected = ",".join(item["expected_source_ids"]) or "none"
        ranked = ",".join(row["ranked_ids"]) or "no result"
        print(f"{item['id']:<4} {expected:<10} {ranked}")

    answerable_count = len(answerable_rows)
    top_one = sum(row["top_one_hit"] for row in answerable_rows) / answerable_count
    hit_at_k = sum(row["hit_at_k"] for row in answerable_rows) / answerable_count
    mrr = sum(
        reciprocal_rank(row["ranked_ids"], row["item"]["expected_source_ids"])
        for row in answerable_rows
    ) / answerable_count
    false_match_rate = sum(bool(row["ranked_ids"]) for row in unanswerable_rows) / len(
        unanswerable_rows
    )
    print(f"Expected source at rank 1: {top_one:.1%}")
    print(f"Expected source somewhere in top {TOP_K}: {hit_at_k:.1%}")
    print(f"Mean reciprocal rank: {mrr:.3f}")
    print(f"Unanswerable questions with any retrieved result: {false_match_rate:.1%}")


def evaluate_answers(questions, documents):
    """Measure answer coverage, exact labeled evidence, and abstention behavior."""
    answerable_rows = []
    unanswerable_rows = []

    print("\nExtractive answer and evidence-sufficiency check")
    print("ID   Expected   System decision   Evidence label")
    for item in questions:
        result = answer_question(item["question"], documents)
        answered = result["answerable"]
        expected_ids = item["expected_source_ids"]
        if answered:
            document = result["document"]
            correct = (
                document["id"] in expected_ids
                and result["sentence"] == item.get("expected_evidence")
            )
        else:
            correct = False

        row = {"item": item, "result": result, "correct": correct}
        (answerable_rows if item["answerable"] else unanswerable_rows).append(row)
        decision = "QUOTE" if answered else "ABSTAIN"
        label = "exact expected quote" if correct else ("no quote" if not answered else "different quote/source")
        expected = ",".join(expected_ids) or "none"
        print(f"{item['id']:<4} {expected:<10} {decision:<16} {label}")

    answerable_count = len(answerable_rows)
    unanswerable_count = len(unanswerable_rows)
    answered_answerables = [row for row in answerable_rows if row["result"]["answerable"]]
    correct_answerables = sum(row["correct"] for row in answerable_rows)
    correct_answered = sum(row["correct"] for row in answered_answerables)
    false_answers = sum(row["result"]["answerable"] for row in unanswerable_rows)
    correct_abstentions = unanswerable_count - false_answers
    decision_correct = len(answerable_rows) - (answerable_count - len(answered_answerables)) + correct_abstentions

    coverage = len(answered_answerables) / answerable_count
    exact_rate = correct_answerables / answerable_count
    selective_accuracy = correct_answered / len(answered_answerables) if answered_answerables else 0.0
    abstention_rate = correct_abstentions / unanswerable_count
    unsupported_answer_rate = false_answers / unanswerable_count
    decision_accuracy = decision_correct / len(questions)

    print(f"Answerable-question coverage: {coverage:.1%} ({len(answered_answerables)}/{answerable_count})")
    print(f"Exact expected source-and-sentence rate: {exact_rate:.1%} ({correct_answerables}/{answerable_count})")
    print(f"Exact evidence among emitted answers: {selective_accuracy:.1%}")
    print(f"Correct abstention on unanswerable questions: {abstention_rate:.1%} ({correct_abstentions}/{unanswerable_count})")
    print(f"Unsupported answers on unanswerable questions: {unsupported_answer_rate:.1%} ({false_answers}/{unanswerable_count})")
    print(f"Answer/abstain decision accuracy on this set: {decision_accuracy:.1%}")


def main():
    documents = load_documents(DATA_PATH)
    questions = load_questions(QUESTIONS_PATH)
    print(
        f"Invented evaluation set: {len(questions)} questions "
        f"({sum(item['answerable'] for item in questions)} answerable, "
        f"{sum(not item['answerable'] for item in questions)} unanswerable)."
    )
    evaluate_retrieval("Raw word-count cosine", search_documents, questions, documents)
    evaluate_retrieval("TF-IDF cosine", search_documents_tfidf, questions, documents)
    evaluate_answers(questions, documents)
    print(
        "\nAll handbook text and evaluation labels are invented practice/reference data. "
        "No model is trained. These small hand-written results do not establish "
        "general reliability; the concept groups are a transparent, manually chosen heuristic."
    )


if __name__ == "__main__":
    main()
