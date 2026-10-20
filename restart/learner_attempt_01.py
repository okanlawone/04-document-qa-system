"""Learner-authored practice attempt: exact action-and-method lookup.

All facts below come from an invented practice source. This script classifies
manually supplied action/method pairs; it does not understand the English
question or quote the original source sentence.
"""

facts = [
    {"action": "request a refund", "method": "app"},
    {"action": "cancel a subscription", "method": "phone"},
    {"action": "send a refund question", "method": "email"},
]


def find_evidence(requested_action, requested_method):
    for fact in facts:
        action_presence = fact["action"] == requested_action
        method_presence = fact["method"] == requested_method

        if action_presence and method_presence:
            return fact
    return None


def analyze_query(requested_action, requested_method):
    fact = find_evidence(requested_action, requested_method)

    if fact:
        return "Stated"

    return "Not Stated"


if __name__ == "__main__":
    questions = [
        {"action": "request a refund", "method": "app"},
        {"action": "request a refund", "method": "phone"},
        {"action": "cancel a subscription", "method": "phone"},
        {"action": "cancel a subscription", "method": "email"},
    ]

    for question in questions:
        print(analyze_query(question["action"], question["method"]))