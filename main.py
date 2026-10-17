"""A tiny, retrieval-first document Q&A project using only Python's standard library."""

import json
import math
import re
from collections import Counter
from pathlib import Path


DATA_PATH = Path(__file__).parent / "data" / "handbook.json"

# These small, hand-written groups let the first answer layer recognize a few
# visible paraphrases. They are a teaching device, not a general language model.
CONCEPT_GROUPS = {
    "booking or session": (
        "booking", "book", "reserve", "reserved", "reservation", "reservations",
        "session",
    ),
    "schedule reference": ("calendar", "slot"),
    "woodworking": ("woodshop", "woodworking", "powered tool"),
    "duration": ("long", "length", "duration", "last", "lasts", "time", "minute", "minutes", "hour", "hours"),
    "advance notice": ("early", "advance", "notice", "before", "ahead"),
    "cancellation": ("cancel", "cancellation", "release"),
    "safety": ("safe", "safety", "protection", "protective", "eye protection"),
    "clothing": ("clothes", "clothing", "shoe", "shoes", "footwear"),
    "equipment": ("equipment", "gear", "protective", "eye protection"),
    "ceramics room": ("ceramic", "ceramics", "clay", "glaze"),
    "opening time": ("open", "night", "evening", "weekday", "weekdays", "hours"),
    "tool": ("tool", "tools"),
    "person who can help": ("who", "teach", "teacher", "help", "steward"),
    "setup instruction": ("setup", "set up", "explain", "show", "guide"),
    "children": ("child", "children", "kid", "kids"),
    "weekend": ("weekend", "weekends"),
    "parking": ("parking",),
    "inclusion or fee": ("include", "included", "pass", "fee", "cost"),
    "membership": ("member", "membership", "member's"),
    "price": ("price", "cost", "fee", "charge"),
    "wireless network": ("wifi", "wi fi", "wireless"),
    "password": ("password",),
}


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


def contains_phrase(text, phrase):
    """Return whether the phrase's token sequence occurs inside text's tokens."""
    text_tokens = tokenize(text)
    phrase_tokens = tokenize(phrase)
    width = len(phrase_tokens)
    return bool(width) and any(
        text_tokens[start : start + width] == phrase_tokens
        for start in range(len(text_tokens) - width + 1)
    )


def concepts_in(text):
    """Find the hand-written concept groups that have a visible clue in text."""
    return {
        concept
        for concept, aliases in CONCEPT_GROUPS.items()
        if any(contains_phrase(text, alias) for alias in aliases)
    }


def sentences_in(text):
    """Split this small English handbook into sentences at . ! or ? marks."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def answer_question(question, documents, max_window_sentences=1):
    """Quote evidence only when it covers every detected clue in the allowed window."""
    if max_window_sentences not in (1, 2):
        raise ValueError("max_window_sentences must be 1 or 2 for this comparison.")

    question_concepts = concepts_in(question)
    if not question_concepts:
        return {"answerable": False, "reason": "I could not identify a handbook topic in the question."}

    # Retrieval proposes candidate records. The separate clue check decides
    # whether the source window may be quoted; ranking alone never answers.
    ranked_documents = search_documents(question, documents, limit=len(documents))
    supported = []
    partial = []

    for retrieval_score, document in ranked_documents:
        source_sentences = sentences_in(document["text"])
        evidence_windows = []
        for width in range(1, min(max_window_sentences, len(source_sentences)) + 1):
            evidence_windows.extend(
                " ".join(source_sentences[start : start + width])
                for start in range(len(source_sentences) - width + 1)
            )

        for evidence in evidence_windows:
            # The title supplies a small amount of source context for topic
            # matching. Evidence itself remains an exact quote from the record.
            source_context = f"{document['title']} {evidence}"
            covered = concepts_in(source_context)
            missing = question_concepts - covered
            candidate = {
                "document": document,
                "sentence": evidence,
                "retrieval_score": retrieval_score,
                "covered_concepts": sorted(question_concepts & covered),
                "missing_concepts": sorted(missing),
            }
            partial.append(candidate)
            if not missing:
                supported.append(candidate)

    if supported:
        # Prefer the best-ranked document; within ties, prefer the shorter quote.
        supported.sort(key=lambda item: (-item["retrieval_score"], len(item["sentence"])))
        result = supported[0]
        return {"answerable": True, **result}

    if partial:
        partial.sort(
            key=lambda item: (
                -len(item["covered_concepts"]),
                -item["retrieval_score"],
                len(item["sentence"]),
            )
        )
        best_partial = partial[0]
        return {
            "answerable": False,
            "reason": (
                f"No quoted window of at most {max_window_sentences} sentence(s) "
                "and its title covered every detected topic clue."
            ),
            **best_partial,
        }

    return {"answerable": False, "reason": "No handbook passage shared searchable words with the question."}



FACTS_PATH = Path(__file__).parent / "data" / "handbook_facts.json"


def load_handbook_facts(path, documents):
    """Load hand-written relationship annotations and verify every quote."""
    with path.open(encoding="utf-8") as file:
        facts = json.load(file)

    documents_by_id = {document["id"]: document for document in documents}
    required_fields = {
        "id", "subject", "action", "object", "time_or_condition", "days",
        "location", "source_id", "evidence",
    }
    for fact in facts:
        missing_fields = required_fields - fact.keys()
        if missing_fields:
            raise ValueError(
                f"Fact {fact!r} is missing: {', '.join(sorted(missing_fields))}"
            )
        document = documents_by_id.get(fact["source_id"])
        if document is None or fact["evidence"] not in document["text"]:
            raise ValueError(
                f"Fact {fact['id']} must quote text from its labeled source document."
            )
    return facts


def explicit_subject_before(question, action_words):
    """Recognize a few named actors that appear before the requested action."""
    words = tokenize(question)
    action_positions = [index for index, word in enumerate(words) if word in action_words]
    if not action_positions:
        return None

    before_action = words[: min(action_positions)]
    if "steward" in before_action or "stewards" in before_action:
        return "shop stewards"
    if "member" in before_action or "members" in before_action:
        return "every member"
    if "front" in before_action and "desk" in before_action:
        return "front desk"
    return None


def relationship_frame(question):
    """Map a few visible question patterns to explicit relation constraints."""
    words = set(tokenize(question))

    # Check special requested details before broader relation types so that
    # matching topic words cannot hide an unsupported relation.
    if "parking" in words:
        return {"action": "parking_included", "requested": "parking inclusion"}

    if words & {"store", "stores", "stored"}:
        return {
            "action": "steward_stores_equipment",
            "requested": "shop steward stores protective equipment",
            "subject_required": "shop steward",
            "object_required": "protective equipment",
        }

    if "ceramic" in words or "ceramics" in words:
        if words & {"open", "opening", "late", "weekday", "weekdays", "saturday", "schedule", "hours", "visit"}:
            day = "Saturday" if "saturday" in words else ("weekdays" if words & {"weekday", "weekdays"} else None)
            return {
                "action": "room_open_schedule",
                "requested": "ceramics room opening schedule",
                "subject_required": "ceramics room",
                "day_required": day,
            }

    if words & {"sell", "sells", "sold", "selling"}:
        if "clay" in words or "glaze" in words:
            frame = {
                "action": "clay_sale",
                "requested": "who sells clay or glaze",
                "ask_for_subject": bool(words & {"who", "seller", "person", "staff"}),
                "location_required": "front desk" if "desk" in words else None,
            }
            subject = explicit_subject_before(question, {"sell", "sells", "sold", "selling"})
            if subject:
                frame["subject_required"] = subject
            return frame

    cancellation_words = {
        "cancel", "canceled", "cancelled", "cancellation", "release", "released",
        "releasing", "give", "giving", "drop", "dropped",
    }
    question_words = tokenize(question)
    calls_off = any(
        question_words[index : index + 2] == ["call", "off"]
        for index in range(len(question_words) - 1)
    )
    if words & cancellation_words or calls_off:
        frame = {
            "action": "cancel_booking_before_start",
            "requested": "who cancels a booking and when",
            "object_required": "booking",
            "ask_for_subject": bool(words & {"who", "which", "person", "responsible", "staff"}),
        }
        subject = explicit_subject_before(
            question,
            {"cancel", "cancels", "canceled", "cancelled", "release", "released", "releasing", "give", "giving", "drop", "dropped", "call"},
        )
        if subject:
            frame["subject_required"] = subject
        return frame

    if "orientation" in words and ("tool" in words or "powered" in words):
        return {
            "action": "orientation_before_powered_tool",
            "requested": "orientation timing relative to powered-tool use",
        }

    if "orientation" in words and words & {"when", "day", "evening", "evenings", "offered", "available"}:
        return {
            "action": "orientation_schedule",
            "requested": "when the orientation is offered",
        }

    if words & {"minutes", "minute", "duration", "length", "long"}:
        return {
            "action": "session_duration",
            "requested": "woodshop session duration",
            "object_required": "90 minutes",
        }

    if words & {"ask", "speak", "consult", "check"} and words & {"tool", "unfamiliar", "before", "new", "trying"}:
        frame = {
            "action": "ask_steward_before_new_tool",
            "requested": "who to ask before using an unfamiliar tool",
            "object_required": "shop steward",
        }
        subject = explicit_subject_before(question, {"ask", "asks", "speak", "speaks", "consult", "consults", "check", "checks"})
        if subject:
            frame["subject_required"] = subject
        return frame

    if words & {"explain", "guide", "prepare", "setup", "show", "demonstrate", "configure", "walk"} and words & {"who", "which", "person", "steward", "safely", "safe", "member", "members"}:
        frame = {
            "action": "explain_safe_setup",
            "requested": "who explains safe setup",
            "object_required": "safe setup",
            "ask_for_subject": bool(words & {"who", "which", "person"}),
        }
        subject = explicit_subject_before(
            question,
            {"explain", "explains", "explaining", "guide", "guides", "prepare", "prepares", "show", "shows", "demonstrate", "demonstrates", "configure", "configures", "walk", "walks"},
        )
        if subject:
            frame["subject_required"] = subject
        return frame

    return None


def fact_matches_frame(fact, frame):
    """Require the requested relationship slots to agree with the fact record."""
    if fact["action"] != frame["action"]:
        return False

    object_required = frame.get("object_required")
    if object_required and object_required not in (fact["object"] or ""):
        return False

    subject_required = frame.get("subject_required")
    if subject_required and subject_required != (fact["subject"] or "").lower():
        return False

    location_required = frame.get("location_required")
    if location_required and location_required != (fact["location"] or "").lower():
        return False

    day_required = frame.get("day_required")
    if day_required and day_required not in fact["days"]:
        return False

    if frame.get("ask_for_subject") and not fact["subject"]:
        return False

    return True


def answer_question_with_relationships(question, documents, facts=None):
    """Quote a cited fact only when its labeled relationship matches the question."""
    if facts is None:
        facts = load_handbook_facts(FACTS_PATH, documents)

    frame = relationship_frame(question)
    if frame is None:
        return {
            "answerable": False,
            "reason": "This teaching prototype has no question-to-relationship rule for that wording.",
        }

    matches = [fact for fact in facts if fact_matches_frame(fact, frame)]
    if not matches:
        return {
            "answerable": False,
            "reason": "No source fact matched all requested relationship details.",
            "frame": frame,
        }

    fact = matches[0]
    document = next(doc for doc in documents if doc["id"] == fact["source_id"])
    return {
        "answerable": True,
        "frame": frame,
        "fact": fact,
        "document": document,
        "sentence": fact["evidence"],
    }

def main():
    documents = load_documents(DATA_PATH)
    question = input("Ask a question about the Northstar Workshop: ").strip()

    if not question:
        print("Please enter a question so the program has words to search for.")
        return

    result = answer_question_with_relationships(question, documents)
    if result["answerable"]:
        document = result["document"]
        fact = result["fact"]
        print("\nRelevant source evidence (quoted, not generated):")
        print(f'“{result["sentence"]}”')
        print(f"Source: {document['title']} [{document['id']}] — {document['source']}")
        print(
            "Matched relationship fields: "
            f"subject={fact['subject'] or 'not stated'}; "
            f"action={fact['action']}; object={fact['object']}; "
            f"time/condition={fact['time_or_condition'] or 'not stated'}"
        )
        print("This is source evidence, not a generated yes/no response.")
        return

    print("\nI cannot answer from the handbook: " + result["reason"])


if __name__ == "__main__":
    main()
