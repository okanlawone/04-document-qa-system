# Daily Build — Trace What the Caller Receives

**Project:** Document Q&A System, fresh-start track  
**Date:** October 21, 2026  
**One idea:** printing text inside a function does not return that text to the code that called the function.  
**Practice data:** the labels below are invented examples, not real policy and not model-training data.

## Runnable example

Run this file with `python3 restart/return_print_trace.py`:

```python
def return_status():
    return "Stated"


def print_status():
    print("Stated")


print("Calling the function that returns a value:")
print(return_status())

print("Calling the function that only prints:")
print(print_status())
```

## Read it line by line

- `def return_status():` creates a function named `return_status`.
- `return "Stated"` sends the text `Stated` back to the caller. It does not display anything by itself.
- `def print_status():` creates a second function.
- `print("Stated")` displays the word `Stated` while that function runs. There is no `return` statement in this function.
- The first heading is displayed by the outer `print`.
- In `print(return_status())`, Python calls `return_status()` first. It gets the returned text, then the outer `print` displays it.
- The second heading is then displayed.
- In `print(print_status())`, Python calls `print_status()` first. Its inner `print` displays `Stated`. The function reaches its end without returning a value, so Python gives the caller `None`. The outer `print` then displays `None`.

The complete output is:

```text
Calling the function that returns a value:
Stated
Calling the function that only prints:
Stated
None
```

The two `Stated` lines look the same on screen, but they came from different places. In the first case, the function returned text and the outer `print` displayed it. In the second, the function displayed text itself, then the outer `print` displayed the function's `None` result.

## Connection to your lookup

Your `analyze_query` function currently uses `return "Not Stated"` for an unsupported action/method pair. The loop calls it inside `print(...)`, so the function returns the label and the loop displays it. If the function printed the label but did not return it, the loop's `print(...)` would receive `None`.

This is still the same return-versus-print idea from the previous walkthrough. Your earlier learner check remains open; this extra trace gives another concrete way to reason through it. Repository publication records project work, not proof of mastery.

## Your learner check

In `print(print_status())`, which call displays the first `Stated` line, and what value does the outer `print` display afterward? Answer in your own words before we move to another concept.