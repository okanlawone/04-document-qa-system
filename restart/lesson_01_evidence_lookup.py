"""Lesson 1: a tiny evidence lookup using invented practice facts.

This is a transparent teaching prototype. It does not understand natural-language
questions; the action and method are supplied as structured values below.
"""

# Invented practice source text. This is not real policy.
DOCUMENT = "You can renew online. Call support with billing questions."

# Each record connects one action and method to the exact sentence that supports it.
# These records were written by a person from DOCUMENT; the program does not
# automatically extract them from the paragraph.
FACTS = [
    {
        "action": "renew",
        "method": "website",
        "evidence": "You can renew online.",
    },
    {
        "action": "billing_help",
        "method": "phone",
        "evidence": "Call support with billing questions.",
    },
]


def find_evidence(requested_action, requested_method):
    """Return a matching source fact, or None when this table has no match."""
    for fact in FACTS:
        same_action = fact["action"] == requested_action
        same_method = fact["method"] == requested_method

        if same_action and same_method:
            return fact

    return None


def answer_from_facts(question, requested_action, requested_method):
    """Show a source quote only when both requested fields match one fact."""
    fact = find_evidence(requested_action, requested_method)

    if fact is None:
        return f"Question: {question}\nResult: The document does not state this."

    return f"Question: {question}\nEvidence: {fact['evidence']}"


if __name__ == "__main__":
    examples = [
        ("Can I renew on the website?", "renew", "website"),
        ("Can I renew by phone?", "renew", "phone"),
        ("Can I get billing help by phone?", "billing_help", "phone"),
    ]

    print(f"Invented source: {DOCUMENT}\n")
    for question, action, method in examples:
        print(answer_from_facts(question, action, method))
        print()
