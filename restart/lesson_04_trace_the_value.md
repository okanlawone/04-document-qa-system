# Restart Lesson 4: Trace the Returned Value

This page is a **slow-motion replay** of the same idea from Lesson 3: a function returns a value, the calling code saves it, and later `print` lines display it. It is a companion to Lesson 3, not a new topic or a passed checkpoint.

## The example

```python
def return_status():
    return "Stated"

status = return_status()

print("First display:")
print(status)

print("Second display:")
print(status)
```

`status` is a variable name. Here it refers to the text `"Stated"` after the assignment finishes.

## Replay the code in order

| Step | Line or action | What happens |
| --- | --- | --- |
| 1 | `def return_status():` | Python creates a function named `return_status`. The indented body is ready, but it does not run yet. |
| 2 | `status = return_status()` | Python reaches the function call on the right side of `=` and runs the function once. |
| 3 | `return "Stated"` | The function sends the text `"Stated"` back to the call site, then finishes. |
| 4 | Finish `status = ...` | The returned text is saved under the name `status`. |
| 5 | `print("First display:")` | The program displays a label. This does not call the function. |
| 6 | `print(status)` | Python looks up the saved text and displays `Stated`. This does not call the function. |
| 7 | `print("Second display:")` | The program displays another label. This does not call the function. |
| 8 | `print(status)` | Python looks up the same saved text and displays `Stated` again. This does not call the function. |

The output is:

```text
First display:
Stated
Second display:
Stated
```

## Keep the jobs separate

- **Call:** `return_status()` asks the function to run.
- **Return:** `return "Stated"` sends a value back to the caller.
- **Store:** `status = ...` gives the returned value a name so the program can reuse it.
- **Display:** `print(status)` shows the saved value. It does not run `return_status()` again.

So this example calls the function **once** and displays the returned word **twice**. The two labels are displayed too.

## Your open checkpoint

This is the same learner check as Lesson 3, shown again with the execution order spelled out. In the code above, identify the line that calls the function and stores its result, the line that returns the text, and the two lines that display `status`. If you added one more `print(status)` at the end, how many times would the function run, and how many times would `Stated` appear? Explain the order in your own words. I’ll wait for your answer before moving to another concept.

This example uses a fixed invented status string to make Python’s flow visible. It is a programming example, not a trained model or a document-Q&A evaluation.
