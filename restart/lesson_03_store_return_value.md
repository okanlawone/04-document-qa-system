# Restart Lesson 3: Store and Reuse a Returned Value

## One idea

A function can **return** a value to the code that called it. If you save that value in a **variable**, you can use it again without calling the function again.

A **variable** is a name that refers to a value. In this example, `status` refers to the text `"Stated"` after the function call finishes.

## Run the example

From the project root, run:

```bash
python3 restart/store_return_value.py
```

The program displays:

```text
First display:
Stated
Second display:
Stated
```

## Read the code line by line

```python
def return_status():
    return "Stated"

status = return_status()

print("First display:")
print(status)

print("Second display:")
print(status)
```

1. `def return_status():` defines a function named `return_status`. Defining it prepares the function; it does not run the function yet.
2. `return "Stated"` sends the text `"Stated"` back to whatever part of the program calls the function.
3. `status = return_status()` calls the function once. When the function returns `"Stated"`, the assignment stores that text under the name `status`.
4. `print("First display:")` displays a label. It does not call `return_status`.
5. `print(status)` looks up the saved text and displays it. It does not call `return_status`.
6. `print("Second display:")` displays another label.
7. The final `print(status)` looks up and displays the same saved text a second time. The function still has only been called once.

The key sequence is **call → return → store → display → display**. `return` gives a value back to the caller; `print` shows a value on the screen. The two jobs are different.

## Connection to the project

A question-analysis function can return a status such as “evidence found” or “not supported.” The calling code can save that returned status and then decide when to display it or use it in another step. This tiny example uses a fixed invented status so the flow is easy to see; it is not a trained model and does not evaluate a real document.

## Your check

In this example, which line calls the function and stores its returned text, which line sends a value back, and which lines display it? If `print(status)` appeared three times, how many times would the function run and how many times would the text be displayed? Explain the order in one or two sentences. I’ll wait for your answer before teaching another concept.
