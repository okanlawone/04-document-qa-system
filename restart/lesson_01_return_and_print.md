# Daily Build — Your First Lookup, Return, and Print

**Project:** Document Q&A System, fresh-start track  
**Date:** October 20, 2026  
**One idea:** a function can return a result, while the calling code decides when to display it.  
**Practice data:** the refund and subscription statements below are invented; they are not real policy and are not model-training data.

## The project work you wrote

Your learner-authored code is saved in [`learner_attempt_01.py`](learner_attempt_01.py). It represents the challenge source as a Python list of dictionaries:

```python
facts = [
    {"action": "request a refund", "method": "app"},
    {"action": "cancel a subscription", "method": "phone"},
    {"action": "send a refund question", "method": "email"},
]
```

The imagined source is:

> “Customers can request a refund through the app. To cancel a subscription, contact support by phone. Refund questions can be sent by email.”

Each dictionary keeps one action with the method the source associates with it. The source says that refund requests use the app, subscription cancellations use the phone, and refund questions use email. It does not say that refund requests use the phone or that subscription cancellations use email.

## How your functions work

`find_evidence` visits each dictionary in `facts`. It compares the requested action and method with the two values in the *same* dictionary. When both comparisons match, `return fact` sends that dictionary back. If the loop finds no match, `return None` sends back Python’s special “no value” result.

`analyze_query` receives that result and returns a label:

- A matching dictionary is treated as **Stated**.
- `None` is treated as **Not Stated**.

This simple version only checks the labels you supplied. It does not read the English question, discover facts by itself, or return the source sentence as a quotation.

## `return` and `print` do different jobs

In the final loop, this line is the important boundary:

```python
print(analyze_query(question["action"], question["method"]))
```

Python first calls `analyze_query`. That function returns the text `"Stated"` or `"Not Stated"`. Then `print` displays the returned text on screen. So your revision removed printing from inside the function, but it still uses `return` to pass the result back.

A tiny example shows the same flow:

```python
def get_status():
    return "Stated"

status = get_status()  # status now stores the returned text
print(status)          # print displays it
```

`return` does not show text by itself. `print` does not send a useful result back to the caller; it displays text and itself returns `None`.

## Trace the four questions

Read each dictionary and look for an exact action-and-method pair. By tracing the code, the labels should be:

1. refund request + app → **Stated**
2. refund request + phone → **Not Stated**
3. subscription cancellation + phone → **Stated**
4. subscription cancellation + email → **Not Stated**

The easy trap is to borrow “phone” from the cancellation fact or “email” from the refund-question fact and attach it to a different action. Your same-dictionary check prevents that mix-up.

## Limitation and next layer

This learner version returns only a label. It cannot quote the original source because its dictionaries do not contain an `evidence` field. The project’s starter program already shows one way to keep a source quote attached to a matching fact. A later small build can add that field to this version so it returns a citation as well as a label.

This submission is a useful project checkpoint, not proof that every concept is mastered. Continue from the learner check below before moving to the next concept.

## Your learner check

If you replace `return "Not Stated"` inside `analyze_query` with `print("Not Stated")`, while leaving the outer `print(analyze_query(...))` unchanged, what appears for an unsupported question? Explain why there may be an extra `None`.