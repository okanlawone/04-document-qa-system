"""A tiny, retrieval-first document Q&A project using only Python's standard library."""

import json
import math
import re
from collections import Counter
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "handbook.json"


def load_documents(path):
    """Read the invented handbook records from a JSON file."""
    with path.open(encoding="utf-8") as file:
        documents = json.load(file)

    if not documents:
        raise ValueError("The handbook has no documents to search.")

    required_fields = {"id", "title", "source", "text"}
    for document in documents:
        missing_fields = required_fields - document.keys()
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Document {document!r} is missing: {missing}")

    return documents


def tokenize(text):
    """Lowercase text and keep groups of letters and digits as tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


def count_words(text):
    """Represent a piece of text as a word -> count dictionary."""
    return Counter(tokenize(text))


def cosine_similarity(left_counts, right_counts):
    """Compare two word-count vectors, returning a score from 0 to 1."""
    shared_words = left_counts.keys() & right_counts.keys()
    dot_product = sum(left_counts[word] * right_counts[word] for word in shared_words)

    left_length = math.sqrt(sum(count**2 for count in left_counts.values()))
    right_length = math.sqrt(sum(count**2 for count in right_counts.values()))

    if left_length == 0 or right_length == 0:
        return 0.0

    return dot_product / (left_length * right_length)


def search_documents(question, documents, limit=3):
    """Rank documents by how much their word-count patterns match a question."""
    question_counts = count_words(question)
    if not question_counts:
        return []

    ranked_documents = []
    for document in documents:
        document_counts = count_words(document["text"])
        score = cosine_similarity(question_counts, document_counts)
        if score > 0:
            ranked_documents.append((score, document))

    ranked_documents.sort(key=lambda result: (-result[0], result[1]["id"]))
    return ranked_documents[:limit]


def search_documents_tfidf(question, documents, limit=3):
    """Rank documents with word counts that give rarer words more weight."""
    question_counts = count_words(question)
    if not question_counts:
        return []

    # Document frequency counts how many documents contain each word.
    # A word is counted once per document, even if it repeats in that document.
    document_frequency = Counter()
    for document in documents:
        document_frequency.update(set(tokenize(document["text"])))

    document_count = len(documents)

    def weighted_counts(word_counts):
        weighted = {}
        for word, count in word_counts.items():
            # Smoothing keeps the weight defined for a query word unseen in the handbook.
            inverse_document_frequency = math.log(
                (document_count + 1) / (document_frequency[word] + 1)
            ) + 1
            weighted[word] = count * inverse_document_frequency
        return weighted

    question_vector = weighted_counts(question_counts)
    ranked_documents = []

    for document in documents:
        document_counts = count_words(document["text"])
        document_vector = weighted_counts(document_counts)
        score = cosine_similarity(question_vector, document_vector)
        if score > 0:
            ranked_documents.append((score, document))

    ranked_documents.sort(key=lambda result: (-result[0], result[1]["id"]))
    return ranked_documents[:limit]


def main():
    documents = load_documents(DATA_PATH)
    question = input("Ask a question about the Northstar Workshop: ").strip()

    if not question:
        print("Please enter a question so the program has words to search for.")
        return

    results = search_documents(question, documents)
    if not results:
        print("I could not find a matching passage. Try different words from the handbook.")
        return

    print("\nClosest handbook passages (this baseline does not generate new wording):")
    for rank, (score, document) in enumerate(results, start=1):
        print(f"\n{rank}. {document['title']} [{document['id']}]")
        print(f"   Match score: {score:.3f}")
        print(f"   Source: {document['source']}")
        print(f"   Evidence: {document['text']}")


if __name__ == "__main__":
    main()
