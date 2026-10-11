# Project 4 — Build a Document Q&A System

## Session 1: How can a program find evidence in documents?

**Status:** first milestone of a multi-session project  
**Article idea:** “Building a Document Q&A System” in the retrieval and knowledge systems category  
**Today’s deliverable:** a small Python program that ranks handbook passages and shows their sources  
**Not built yet:** an LLM, generated answers, embeddings, a trained model, or a quality benchmark

The goal of this lesson is not to memorize Python spellings. It is to understand what problem each part solves, what information flows through the program, and why this first version has limits. You should be able to draw the pipeline and explain every box before we add more machinery.

## 1. The problem we are solving

Imagine a person asks, “How early should I reserve a woodshop session?” The answer is somewhere inside a handful of handbook passages. A person can scan them. A computer needs a method for deciding which passage is most relevant.

Our first system does this:

1. Read a small collection of documents.
2. Read a question.
3. Break both the question and each document into words.
4. Count those words.
5. Compare the question’s counts with each document’s counts.
6. Show the best matching passage and where it came from.

This is a **retrieval** system: it finds existing information. A complete RAG system also uses retrieved information to help a **generator** compose a response. We are deliberately making retrieval visible first, because if the wrong evidence is retrieved, a fluent generator can still produce a wrong answer.

### A useful mental picture

```text
Question ──> represent its words ──┐
                                    ├──> compare ──> ranked evidence + source
Documents ─> represent their words ┘
```

The program has no understanding in the human sense. It applies a defined comparison rule to text and returns the result of that rule.

## 2. What is in the data?

Open `data/handbook.json`. It contains five records. Every record has:

- `id`: a short identifier such as `WS-02` so we can refer to a passage consistently.
- `title`: a human-readable label.
- `source`: the fictional handbook from which this passage came.
- `text`: the passage the search program will compare with the question.

**All Northstar Workshop information is invented practice data.** It is not real-world guidance.

These text passages are the system’s **reference collection** or **knowledge base**. They are not examples used to fit a model. This program does not fit or train a model at all. It performs a calculation each time the program runs. In a later system, retrieved passages might be sent to a language model, but providing a document as context still does not mean that document was used to train that model.

This distinction matters:

- A **training example** is used to adjust a model’s learned parameters during training.
- A **reference document** is information a system searches or supplies as context at use time.
- Our invented handbook is only a reference document.

## 3. Vocabulary, without assuming prior knowledge

### Character and word

A character is one symbol such as `a`, `7`, or `?`. A word is a sequence of characters that we treat as one unit. Natural language has complications—contractions, hyphens, accents, and word forms—but our first tokenizer uses a deliberately simple rule.

### Token

A **token** is a unit of text chosen by a processing rule. Here, each run of English letters or digits becomes one token. For example:

```text
"Book the woodshop, 24 hours early!"
```

becomes approximately:

```text
["book", "the", "woodshop", "24", "hours", "early"]
```

The comma and exclamation point are not kept. All letters become lowercase, so `Woodshop` and `woodshop` match.

### Word-count vector

A vector is an ordered set of numbers. If our vocabulary is `book`, `hours`, and `woodshop`, the text “book the woodshop” can be represented as:

```text
book: 1, hours: 0, woodshop: 1  ->  [1, 0, 1]
```

The name “vector” may sound advanced, but at this point it is simply a list of numbers with a consistent meaning for each position. `Counter` stores the same idea as a word-to-count mapping.

### Similarity score

A similarity score is a rule-based estimate of how alike two representations are. Our cosine similarity looks at the angle between two count vectors. With nonnegative word counts, the score is between 0 and 1 here: `0` means no words overlap; a larger score means their word-count patterns point in a more similar direction.

The score is **not** a probability, confidence guarantee, truth score, or model accuracy. A score of `0.8` does not mean “80% sure this answer is correct.” It only describes the result of this particular comparison formula.

## 4. Why cosine similarity?

Suppose a short question uses the words `reserve` and `session`. One long document might contain those words plus many unrelated words. A raw overlap count tends to favor longer text because long text has more chances to share words.

Cosine similarity divides the shared-word dot product by the lengths of both word-count vectors:

```text
similarity = dot product / (question length × document length)
```

This normalizes for the overall number and repetition of words. It is useful for a first experiment, but it still only notices shared tokens. It does not know that “book” can mean “reserve,” or that “session” and “appointment” can be related.

### One small worked example

If the question vector is `[1, 1, 0]` and a passage vector is `[2, 0, 1]`, then:

- Shared-position products add to `1×2 + 1×0 + 0×1 = 2`.
- The question length is `sqrt(1² + 1² + 0²) = sqrt(2)`.
- The passage length is `sqrt(2² + 0² + 1²) = sqrt(5)`.
- The similarity is `2 / (sqrt(2) × sqrt(5))`, approximately `0.632`.

You do not need to memorize the formula. Understand its parts: multiply values at corresponding word positions, add those products, then scale by both vector lengths.

## 5. Read the code as a sequence of responsibilities

Open `main.py` while reading this section. The functions are separated so each one has a clear job.

### Imports and path

```python
import json
import math
import re
from collections import Counter
from pathlib import Path
```

- `json` reads the structured records in the data file.
- `math` supplies square roots for vector lengths.
- `re` finds letter-and-digit groups for tokenization.
- `Counter` counts how often each token occurs.
- `Path` builds a file location in a way that works across operating systems.

`DATA_PATH = Path(__file__).parent / "data" / "handbook.json"` means: start from the folder containing `main.py`, then enter `data`, then select `handbook.json`. This makes the program find its practice data even if you run it from a different current folder.

### Loading and checking records

`load_documents(path)` opens the JSON file using UTF-8 text encoding and turns its contents into ordinary Python values: a list of dictionaries. It checks that the list is not empty and that every dictionary contains the four fields the rest of the program needs.

Why check? If a required field is absent, the search code would fail later with a confusing error. A clear early check gives a more useful explanation close to the source of the bad data.

### Turning text into tokens

`tokenize(text)` lowercases the text and applies this pattern:

```python
r"[a-z0-9]+"
```

The square brackets describe allowed characters, and `+` means “one or more in a row.” The result is a list of words and number groups. This rule is intentionally simple, not a universal solution for every language.

### Counting tokens

`count_words(text)` runs the tokenizer and passes its list to `Counter`. If a token appears more than once, its count is larger. For example, “woodshop woodshop safety” becomes counts like `woodshop: 2` and `safety: 1`.

### Comparing two pieces of text

`cosine_similarity(left_counts, right_counts)` receives two word-count mappings.

1. `left_counts.keys() & right_counts.keys()` finds words present in both mappings.
2. The `dot_product` multiplies the two counts for each shared word and adds the products.
3. Each `length` is the square root of the sum of squared word counts.
4. If either length is zero, there is no usable vector, so the score is zero.
5. Otherwise, the function divides the dot product by both lengths.

The word `left` or `right` is just a local name; it does not mean one text is more important. In this call, the left mapping is the question and the right mapping is one document.

### Ranking documents

`search_documents(question, documents, limit=3)` counts the question’s words, then repeats these steps for every document:

- Count words in that passage.
- Calculate the similarity score.
- Keep the passage if its score is greater than zero.

The list is sorted with the highest score first. The document ID breaks ties consistently, so equal scores do not jump around unpredictably. Finally, `[:limit]` keeps at most three results.

A useful question to ask is: why discard a zero-score passage? Because zero means this word-based method found no shared token. Showing it as a match would imply evidence the method did not find. It does not prove the passage is semantically irrelevant; it only proves this baseline found no shared words.

### Running the interaction

`main()` loads the documents, asks you to type a question, checks for an empty input, and handles the no-match case. For each result, it prints the title, ID, score, source, and full evidence passage.

The final guard:

```python
if __name__ == "__main__":
    main()
```

means “run the interactive program when this file is launched directly.” It also lets a later Python file import these functions without automatically prompting for input.

## 6. Run it and investigate, don’t just copy it

From this project folder, run:

```bash
python3 main.py
```

Try each question separately and write down what you observe:

1. `How early should I reserve a woodshop session?`
2. `When do I need to book the workshop?`
3. `What do I need before using a powered tool?`
4. `Can I use the saw tomorrow?`
5. `Tell me about moon gardening.`

For each question, inspect the source text. Do not judge a result only by whether the score is numerically high. Ask:

- Did the expected source appear near the top?
- Which words overlapped?
- Did the program miss a paraphrase because its vocabulary was different?
- Did it return anything for a question the handbook cannot answer?
- Does showing the exact passage help you detect a bad match?

Record these observations in `progress.md`. This is how you move from running code to forming an engineering explanation.

## 7. What this baseline can and cannot do

### It can

- Read several small text records.
- Rank records that share words with a question.
- Show source labels and the exact text that created the match.
- Explain its calculation in ordinary Python.

### It cannot

- Understand synonyms or meaning in context.
- Handle long documents intelligently; the current unit is one whole record.
- Guarantee that a top-ranked result answers the question.
- Compose a new answer in natural language.
- Prove quality from a few examples.
- Learn from data or improve itself through training.

A program can produce a number with several decimal places while still being a weak system. Precision of formatting is not evidence of real-world accuracy.

## 8. How we will grow the project

The next milestone should be driven by what you observed. We can create a tiny evaluation set: several questions, the expected supporting document IDs, and some questions that no document can answer. Then we can calculate simple retrieval measures and compare a better word-weighting method with this baseline.

Only after we understand retrieval and have examples to check it against should we add answer generation. A generated answer must be grounded in retrieved passages, include traceable citations, and admit when evidence is missing. We will choose an approach that we can run and explain; a framework will not substitute for understanding the pipeline.

Possible later steps include splitting long documents into chunks, retaining page/source metadata, comparing lexical retrieval with semantic embeddings, and testing a language model’s answers. Each new step should solve a limitation we have actually observed.

## 9. Check your understanding

Try explaining these aloud without reading the code:

1. What goes into the reference collection, and what does not happen to it?
2. Why do we lowercase and tokenize text?
3. What does one position in a word-count vector represent?
4. Why does the similarity formula divide by vector lengths?
5. What does a zero score tell us—and what does it not tell us?
6. Why print the exact passage and source instead of only the score?
7. What can go wrong if we add a language model before checking retrieval quality?

If an answer is fuzzy, that is a signal to inspect one function or one example again—not a reason to memorize the function name.

## 10. Open-ended challenge

Improve this system for a user who has a growing handbook. Choose one limitation, explain why it matters, change one part, and compare the result with the original behavior. Keep the evidence visible. Do not change several things at once; if the result improves or gets worse, we want to know which change caused it.

A strong project explanation will eventually include:

- The user problem and the system boundary.
- How source documents become searchable representations.
- How retrieval quality is checked.
- How answer claims stay connected to sources.
- What kinds of questions the system should decline.
- What you measured, what you observed, and what you still do not know.
