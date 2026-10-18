# Project 4 — Build a Document Q&A System

> **Fresh start:** begin with [`restart/lesson_01.md`](restart/lesson_01.md). The earlier sessions remain below as project history; start here if you want the foundations retaught from zero.

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

A vector is an ordered set of numbers. If our vocabulary is `book`, `hours`, and `woodshop`, the text “book woodshop” can be represented as:

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

# Session 2: How can we tell whether retrieval is working?

**Status:** second milestone of the multi-session project

**Today’s deliverable:** a small labeled question set, an evaluator, a comparison of two word-based ranking methods, and a reasoned decision about how to add answers safely
**Still not built:** a language model, semantic embeddings, or a trained model

In Session 1, the program returned passages that looked similar to a question. But “looks similar” is not the same as “contains the right answer.” This session gives us a repeatable way to check that difference.

## 1. The new data: questions with expected sources

Open `data/evaluation_questions.json`. It contains ten questions. For an answerable question, `expected_source_ids` lists the handbook record or records that contain evidence needed for the answer. For an unanswerable question, that list is empty.

For example, Q4 asks when a reservation can be canceled. We label `WS-03` as its expected source because that invented passage states the cancellation rule. Q9 asks whether parking is included. None of the five handbook passages says that, so Q9 has no expected source.

The expected IDs are labels written by a person after reading the handbook. In machine-learning language, people sometimes call labels like these **ground truth**. Here that phrase means “the answer key we intend to use for this small exercise”; it does not mean the labels are perfect or that the data is real.

There are two separate kinds of data in this project:

- `data/handbook.json` is the invented **reference collection** the search program looks through.
- `data/evaluation_questions.json` is invented **evaluation data** that lets us compare returned sources with expected sources.

Neither file trains a model. There is no training step in this project so far. The evaluation questions are also not proof that the system will work on a large collection or on questions written by other people.

## 2. What does the evaluator do?

Run:

```bash
python3 evaluate.py
```

The evaluator follows the same steps for each question:

1. Give the question and all five handbook passages to one search method.
2. Keep the three highest-ranked results.
3. Record the returned source IDs in order.
4. Compare those IDs with the expected IDs.
5. Summarize the rankings with a few measurements.

The code does not alter the handbook or the answer key. It simply repeats a clearly defined comparison for each question so we do not have to judge every output by memory.

## 3. Three ranking measurements, in plain language

### Top-1 accuracy

For answerable questions, this asks: “How often was the expected passage ranked first?” If the correct passage is second, that question does not count as a top-1 success.

Raw word counts got the expected source first for 3 of 7 answerable questions:

```text
3 ÷ 7 = 0.429..., or about 42.9%
```

This is a measurement on our seven answerable examples only.

### Expected-source recall@3

“Recall at 3” asks: “Did at least one expected source appear anywhere in the first three results?” It gives credit even if another passage was ranked first.

Both methods got 7 out of 7 answerable expected sources into their first three results. This means the small collection is often present in the shortlist. It does **not** mean the first result is correct, and it does not mean the result contains enough evidence to answer every detail.

### Mean reciprocal rank

For each answerable question, find the rank of the first expected source. Turn that rank into a score by taking one divided by the rank:

- expected passage at rank 1 → `1 ÷ 1 = 1`
- expected passage at rank 2 → `1 ÷ 2 = 0.5`
- expected passage at rank 3 → `1 ÷ 3 ≈ 0.333`
- expected passage missing → `0`

Then average those scores across the answerable questions. A higher mean reciprocal rank means expected passages tend to appear closer to the top.

## 4. The comparison we observed

| Method | Expected source first | Expected source in top 3 | Mean reciprocal rank | Unanswerable questions with any result |
|---|---:|---:|---:|---:|
| Raw word-count cosine | 3/7 (42.9%) | 7/7 (100%) | 0.667 | 3/3 (100%) |
| TF-IDF cosine | 1/7 (14.3%) | 7/7 (100%) | 0.548 | 3/3 (100%) |

**On this particular question set, TF-IDF ranked the expected source first less often than raw word counts.** That is a useful result: a method that sounds more sophisticated is not automatically better. We should keep the simpler method as our current baseline and retain the comparison so future changes have something to beat.

These measurements are descriptive, not a promise about future questions. Ten examples are far too few to establish broad accuracy. The questions were written by a person who already knows the five documents, so a larger project should add new questions and paraphrases, ideally written without looking at the search results.

## 5. A worked ranking example

For Q1, “How long is one woodshop booking?”, the raw-count method ranked the records:

```text
WS-01, WS-02, WS-03
```

The answer key expects `WS-02`, so the expected source is second. Its reciprocal-rank score for this question is `1 ÷ 2 = 0.5`. The method has not failed to include the useful passage, but it has put a different passage above it. That distinction is why looking only at “a result appeared” would hide an important problem.

The question includes “woodshop,” which also appears in another handbook passage. The program counts word matches; it does not know that the phrase “how long” asks specifically for duration. Its score is a ranking signal, not a probability that the passage is correct.

## 6. What changed in the second method?

The original method counts every word in a question and passage. TF-IDF keeps those counts but changes their weights.

- **Term frequency (TF)** is how often a word occurs in one passage.
- **Document frequency (DF)** is how many different passages contain that word.
- **Inverse document frequency (IDF)** gives relatively more weight to words found in fewer passages.

In our code, the smoothed IDF weight is:

```text
IDF(word) = log((number of passages + 1) / (passages containing the word + 1)) + 1
```

The `+1` values are called smoothing: they prevent a division by zero and keep the formula defined for a query word that appears in none of the handbook passages. The extra `+1` at the end keeps weights positive.

There are five handbook passages. If a word appears in every passage, its IDF is `log(6/6) + 1 = 1`. If it appears in just one passage, its IDF is `log(6/2) + 1`, which is about `2.10`. The rarer word receives more weight because it can distinguish one passage from the others.

The program then compares these weighted word dictionaries with the same cosine-similarity calculation from Session 1. In this tiny example, TF-IDF changed the ordering, but did not improve our measured ranking. It still only recognizes the tokens produced by our simple tokenizer. It does not know that “ceramic” and “ceramics,” “safe” and “safety,” or “booking” and “reservation” can be related.

This is still not semantic search. TF-IDF is a way of weighting words; it does not learn meanings from examples and does not use an embedding model.

## 7. The largest issue: a result is not proof of an answer

The evaluator includes three questions that the handbook cannot answer. Both search methods returned at least one passage for all three.

Q9 asks whether parking is included with a 90-minute session. The handbook does mention a 90-minute session, so word overlap can retrieve that passage. But the passage says nothing about parking. If we treated every returned passage as proof, the program could turn a true detail about session length into a false answer about parking.

The current similarity scores are not calibrated confidence values. A score such as `0.42` does not mean “42% chance this passage answers the question.” We have not found a reliable cutoff that separates supported from unsupported questions, and three unanswerable examples are not enough to select one.

## 8. Decision: use evidence-first, extractive answers before free-form generation

The next answer layer should first be **extractive**: it should choose a short sentence or exact phrase already present in a retrieved passage, then show the passage ID and source. It should not invent new facts while we are still learning how to retrieve and verify evidence.

If no retrieved sentence supports the question, the response should say that the handbook does not provide enough information. A retrieval result alone must not disable that refusal behavior. Before we implement the response layer, we need to make the evidence check explicit and add evaluation examples that distinguish “related passage” from “passage that actually answers the question.”

We are not adding free-form LLM generation now. The first-ranked source is correct on only 3/7 answerable questions with raw counts, and both methods return something for every unanswerable example. A language model could make irrelevant passages sound like confident answers. We should improve and inspect the evidence path before asking a generator to write from it.

This is a project decision, not a claim that extractive answers solve every problem. Exact quotations can still be incomplete, ambiguous, or misleading if the wrong source is selected. Each response will need its source displayed so a person can inspect it.

## 9. Experiments to try

Change one thing at a time and rerun `python3 evaluate.py`:

1. Rewrite Q3 using wording that appears in the booking guide. Does its expected source move upward? What does that tell you about vocabulary overlap?
2. Add a paraphrase that uses “reserve” instead of “book.” Does TF-IDF help when the important word itself is missing from the passage?
3. Add another unanswerable question that shares one ordinary word with a passage. Does either method know the passage is insufficient?
4. Change `TOP_K` in `evaluate.py` from 3 to 1. Which questions lose their expected source from the shortlist?
5. Add an additional handbook passage that repeats common words. Does TF-IDF change rankings because of document frequency?

When you edit the evaluation questions, keep the expected source IDs honest: read the handbook first, and only label a source if its text really supports the answer. Keep a copy of the original results in mind when interpreting your changed set; after tuning on examples, those examples are no longer an independent final check.

## 10. Check your understanding

Try answering in your own words, without copying the code:

1. What is the difference between the handbook reference collection and the evaluation questions?
2. Why does Q9 count as an unanswerable question even though the handbook mentions a 90-minute session?
3. What is the difference between top-1 accuracy and recall@3?
4. If an expected source ranks third, what reciprocal-rank score does that question receive?
5. What does IDF try to emphasize, and what kind of meaning does it still fail to understand?
6. Why did we keep raw counts even though TF-IDF has a more advanced name?
7. Why is a similarity score not the same thing as confidence?
8. What exact source evidence should an extractive response show?
9. Why are these ten questions neither training data nor proof of general performance?

# Session 3: When is a passage enough to answer?

**Today’s goal:** keep the search results visible, then add a cautious layer that either quotes source text with a citation or abstains. We will inspect why a close match can still fail to answer a question.

**New files and changes:** `main.py` now checks question-topic clues against a source title and one or two neighboring sentences. `data/evaluation_questions.json` has 13 hand-written examples. `evaluate.py` reports whether the system quoted the expected evidence or correctly abstained.

**Still not built:** this is not a generative AI system. It has no LLM, embeddings, training step, or probability-calibrated confidence score. Every Northstar Workshop record remains invented practice content.

## 1. The difference between finding and answering

Imagine searching a paper handbook for “Is parking included with the 90-minute session?” A word-matching search can find the booking passage because it contains “session” and “90 minutes.” That only establishes that the passage is related to part of the question. It does not establish that the passage says anything about parking.

Keep these two decisions separate:

1. **Retrieval:** Which existing passages share words with the question?
2. **Evidence sufficiency:** Does a short piece of one passage cover the topics the question asks about?

The first decision makes a candidate list. It is not permission to answer. The second decision in this project is a small, visible rule system. It can still be wrong, so we test both answerable and unanswerable questions.

## 2. A new plain-language idea: topic clues

A topic clue is a word or short phrase that gives a clue about what a question is asking. For example, “parking,” “included,” and “session” are clues in the parking question. The program groups a few related spellings or phrases in `CONCEPT_GROUPS` near the top of `main.py`:

```python
"parking": ("parking",),
"inclusion or fee": ("include", "included", "pass", "fee", "cost"),
```

The braces `{...}` around `CONCEPT_GROUPS` make a dictionary. A dictionary stores pairs: a label such as `parking`, and a tuple of text clues that can activate that label. A tuple is a short, ordered collection of values. These groups are handwritten by us; they were not learned from examples.

A group is meant to cover a few phrases we decide are related for this toy handbook. It does **not** prove those phrases always mean the same thing. For example, treating “night” and “evening” as related may be useful here, but the code does not understand opening hours as a person does.

## 3. What happens to one question?

Let’s trace “Is parking included with the 90-minute session?” through the program.

### Step A: Find its topic clues

`concepts_in(question)` checks the question against every hand-written group. The tokenizer lowercases the text and splits it into letter and number runs. Punctuation is removed. Depending on the word form, this question activates groups such as `parking`, `inclusion or fee`, and `booking or session`.

`contains_phrase(text, phrase)` asks a narrow question: do these phrase tokens appear next to each other in the text tokens? It does not look up a dictionary, search the internet, or infer the meaning of a sentence.

### Step B: Retrieve candidate handbook records

`answer_question` calls the existing raw word-count `search_documents` function. This gives the same kind of ranking we studied earlier. In this example, the booking passage is a strong candidate because “session” overlaps. The ranking score is still a word-overlap measurement, not a measure of truth or answer confidence.

### Step C: Examine short evidence windows

`sentences_in` splits each short handbook text at periods, question marks, and exclamation marks. The answer layer examines each sentence on its own and also each pair of neighboring sentences. A two-sentence window can preserve context when one sentence names the subject and the next gives the detail.

For each possible window, the program combines the document title and the original sentence or sentence pair. The title supplies context, such as “Booking a woodshop session.” It never rewrites the quotation. The candidate passes this simple gate only when every topic group detected in the question has at least one clue in that title-plus-window text.

The parking passage has clues about booking and duration, but it has no clue in the `parking` or `inclusion or fee` groups. It therefore cannot pass the gate. The program abstains and may show the closest passage under a label that says it is **not an answer**.

### Step D: Quote or abstain

If a window passes the gate, the program copies that exact original text and prints its record ID, title, and source. “Extractive” means the answer is extracted from existing text. This code does not write a new sentence that could add unsupported facts.

If no window passes, the program says the handbook does not give enough evidence. An **abstention** is a deliberate decision not to answer when the available evidence is insufficient. It is not a crash or an empty search; it is an answer behavior we chose.

## 4. A useful example and a useful limit

Try the answerable question:

```text
How much time should I set aside after getting a slot on the calendar?
```

The question uses a different phrasing from the handbook. The program detects a duration clue and a calendar/slot clue. The first booking sentence contains the calendar context; the next sentence gives the session length. A two-sentence window can connect them, so the returned quote includes both original sentences and cites `WS-02`.

Now try:

```text
Does the 24-hour booking rule include a parking pass?
```

The same booking record can rank highly because of “24-hour” and “booking.” But its source never mentions a parking pass. The evidence gate should abstain. This is the key lesson: a true sentence about one part of a question is not necessarily evidence for the whole question.

The rule is deliberately conservative and simplistic. It requires all detected topic groups to appear in one short window, but word presence does not equal logical support. A sentence can contain every required word and still fail to answer the question. The handwritten synonym groups can also miss real paraphrases and can accidentally connect words whose meanings differ in context. The exact quote and citation help a person inspect the result; they do not guarantee correctness.

## 5. Read the answer code without memorizing it

Here is a map of the new functions:

- `contains_phrase(text, phrase)` tokenizes both pieces of text and checks whether the phrase’s token sequence occurs in order.
- `concepts_in(text)` returns the labels whose clue phrases occur in the text. A `set` is used because a label should only appear once, even if several of its clues match.
- `sentences_in(text)` returns a list of sentence strings. A list preserves order, which lets us create adjacent sentence pairs.
- `answer_question(question, documents)` coordinates the work: detect clues, retrieve candidates, build windows up to the selected one- or two-sentence limit, check which clues each window covers, then quote a passing window or abstain. Its interactive default is one sentence.
- `main()` reads one question from the keyboard, calls `answer_question`, and prints either a citation or an abstention.

In the evaluator, `expected_evidence` is a human-written label. It lets `evaluate.py` compare the returned quote to the source passage we expected. `main.py` never loads that label. The evaluator’s labels are not fed into the answer routine, and no model parameters are adjusted. So the question set is evaluation data, not model-training data.

## 6. What the expanded evaluation tells us

The hand-written set now has 13 questions: seven answerable and six unanswerable. Three new unanswerable questions deliberately share words with real handbook topics, including woodshop membership price, a parking pass with the 24-hour booking rule, and parking at night in the ceramics room.

In this run of the evaluator:

- The raw word-count baseline put the expected source first for 3/7 answerable questions and somewhere in its top three for 7/7.
- TF-IDF put the expected source first for 1/7 and somewhere in its top three for 7/7.
- Both retrieval methods returned a candidate for all 6/6 unanswerable questions. This confirms that retrieval alone still does not know when to refuse.
- The extractive rule emitted the hand-labeled source quote for 7/7 answerable questions and abstained on 6/6 unanswerable questions in this small set.

Those last results are **not** evidence that the program will answer every real question correctly. We wrote both the questions and the concept groups by hand, and the extra negative examples were designed to probe known overlap failures. The evaluation is useful for checking the code’s behavior on these cases, but it is tiny and not an independent real-world test. A new paraphrase may be missed; a cleverly worded unsupported question may still pass.

The output distinguishes several ideas:

- **Coverage** asks what share of answerable questions received a quote.
- **Exact expected evidence rate** asks what share of answerable questions received the labeled source and exact expected sentence or sentence pair.
- **Correct abstention rate** asks what share of unanswerable questions were refused.
- **Unsupported answer rate** asks what share of unanswerable questions received a quote anyway.

These measurements describe this fixed, invented set only. Adding or changing questions changes the denominator and can change every percentage. The question labels and expected quotations are not training examples.

## 7. Run it and inspect the decision

From the project folder, use Python 3:

```bash
python3 main.py
```

Try an answerable question:

```text
How long is one woodshop booking?
```

Then try the overlapping but unsupported question:

```text
Is parking included with the 90-minute session?
```

For a repeatable report over every labeled question, run:

```bash
python3 evaluate.py
```

When you read a result, inspect three things separately: which record retrieval ranked, which original sentence was quoted, and why the system did or did not abstain. Do not read a similarity score as a probability.

## 8. Experiments and open-ended challenge

Change one idea at a time, then rerun the evaluator and inspect question-level rows:

1. Write a fresh paraphrase without reusing the handbook’s words. Which clue group fails to activate?
2. Add a question that contains all the right topic words but asks for a relationship the passage never states. Does the gate still answer? This probes the difference between word coverage and actual support.
3. Temporarily remove `parking` from its concept group. What happens to the parking questions? Explain why the result changes.
4. Make the system inspect one sentence only, then compare its results with one- or two-sentence windows. Which answers lose necessary context?
5. Add another invented source about a different topic. Does retrieval find it, and does the citation remain identifiable?
6. Create several new questions before changing the concept groups. Keep those questions aside, make one code change, then use them as a fresh check. This reduces the temptation to tune only to examples already seen.

For each experiment, write down what you predicted before running the code, what happened, and what you learned. A surprising result is useful evidence about the rule you implemented.

## 9. Check your understanding

Explain these ideas in your own words, as if teaching another beginner:

1. Why can a highly ranked passage still be irrelevant to part of a question?
2. What is a topic clue in this program, and who decided which words belong together?
3. Why does the parking question need both booking-related and parking-related evidence?
4. What does the system print when it cannot find a window with all detected clues?
5. Why can a one- or two-sentence quotation be easier to verify than a newly generated paragraph?
6. Why does matching every topic clue still fail to prove that the sentence logically supports the answer?
7. What is the difference between the invented handbook, the expected-evidence labels, and model-training data?
8. Why are 7/7 and 6/6 results not a guarantee about new questions?
9. What additional evidence would you want before trusting this approach on a real handbook?

## Session 3 completion checks

- Run both `python3 main.py` examples and explain the quote/citation and the abstention.
- Run `python3 evaluate.py` and describe what each reported rate uses as its denominator.
- Point to the handwritten concept groups and explain one way they could create a false answer or miss a correct one.
- Explain why the handbook is reference content, why expected evidence is evaluation labeling, and why neither is training data.
- Propose an unseen question that could challenge the current evidence gate and predict what it will do before running it.

A passed code check is not the same as personally understanding the system. Use the questions above to find gaps in your explanation and investigate them before moving into a more advanced design.


## Session 4: What changes when evidence spans two sentences?

**Status:** Session 4 baseline evaluation complete; Project 4 remains active.
**Question:** Does looking at two nearby sentences help without making unsupported connections easier?
**Today’s change:** add a window-size setting and five fresh, invented challenge questions. The hand-written concept groups from Session 3 were kept exactly as they were.

### 1. The key idea: evidence has both words and relationships

A **sentence window** is the piece of source text the program checks at one time. A one-sentence window checks one sentence. An up-to-two-sentence window checks each sentence and each pair of neighboring sentences. The second setting allows more context to appear together.

More context can help. A question may combine details that a document places in nearby sentences. But more text can also make unrelated details appear together. Seeing the words “steward,” “protective equipment,” “before,” and “tool” in a two-sentence quote does not prove that the steward stores the equipment before tool use.

This is the distinction we are studying:

- **Lexical coverage** asks, “Did our chosen words or clue groups appear in this text?”
- **Relationship support** asks, “Does the text actually say that the requested thing happened to, caused, included, or preceded the other thing?”

The current rule checks the first question. It does not reliably check the second. After observing the S4-Q3 false positive, the interactive program defaults to one sentence; this cautious choice avoids that tested trap, but it creates the S4-Q1 abstention. The evaluator retains both window settings so this trade-off stays visible.

### 2. Keep the data roles separate

`data/handbook.json` is an invented reference collection. It is the fictional handbook the program searches. `data/evaluation_questions.json` records the earlier 13-question evaluation, and `data/session4_stress_questions.json` adds five fresh challenge questions written before changing the clue rules.

The questions, expected answers, and handbook are **practice/reference/evaluation data**. They are not model-training data. There is no learned model in this program and no training step. We use the labels only to compare what this small program did with what the question author expected.

Because these questions are now visible and have been used to inspect the baseline, treat them as a fixed regression set. If you change the rule to fit them, write a different new challenge set before calling the changed results an independent check.

### 3. Follow one question through the program

Try S4-Q1:

> After I reserve a place on the workshop calendar, how long should I set aside for the woodshop?

The intended source is `WS-02`, whose first two sentences say that members reserve a session through the booking calendar and that a session lasts 90 minutes.

1. The program tokenizes the question into lowercase word tokens.
2. The existing clue rules notice clues for booking/session, calendar, woodshop, and duration.
3. Raw word-count retrieval proposes handbook records. This is only a search ranking; it does not prove that a record answers the question.
4. In one-sentence mode, no single sentence has both the calendar clue and the session-duration clue. The program abstains. Since the handbook does contain the answer, this is an **answerable miss** (a false negative in this evaluation).
5. In up-to-two-sentence mode, the first two adjacent `WS-02` sentences together cover all four clue groups. The program returns those exact sentences and the source ID. They state the booking-calendar detail and the 90-minute duration.

The two-sentence result fixes this particular miss, but that does not establish that two sentences are always safer.

### 4. Follow the trap through the program

S4-Q3 asks:

> Does the shop steward store protective equipment before I use an unfamiliar tool?

The invented source says:

- “Ask the shop steward before using a tool you have not used before.”
- “Shop stewards can explain safe setup and help identify the correct protective equipment.”

The two-sentence window contains the clues for a steward, a tool, “before,” and protective equipment. The rule therefore quotes both sentences. Yet the handbook says nothing about the steward storing equipment. The first sentence says when to ask the steward; the next says that stewards can identify protective equipment. Topic clues co-occur, but the requested action (“store”) is absent.

That emitted quote is **not fabricated**: it is copied from the source. The unsupported step is treating those statements as evidence that the steward stores the equipment. This is a false positive for evidence sufficiency. It shows why “the quote is exact” and “the quote supports the requested answer” are different checks.

S4-Q4 is a useful comparison. It asks whether a two-hour cancellation period includes free parking. The cancellation passage shares the cancellation and time clues, but it contains nothing about a parking pass or a fee. Both window settings abstain, a correct abstention on this invented example.

### 5. The measured result

Run the full evaluation from this repository:

```bash
python3 evaluate.py
```

The first part repeats the original Session 2 retrieval and Session 3 answer checks. The final part compares one-sentence windows with windows containing up to two adjacent sentences on the five Session 4 questions. The interactive program now defaults to one sentence because the wider window produced a false positive on the fresh challenge set; the evaluator still measures both choices.

| Measure on the five Session 4 questions | One sentence | Up to two sentences |
| --- | ---: | ---: |
| Exact expected quote for answerable questions | 2/3 (66.7%) | 3/3 (100%) |
| Answerable questions missed by abstaining | 1/3 (33.3%) | 0/3 (0%) |
| Unsupported questions that got a quote | 0/2 (0%) | 1/2 (50%) |
| Unsupported questions correctly refused | 2/2 (100%) | 1/2 (50%) |

An **answerable miss** here means the system abstained even though the label says the handbook can answer. A **false positive** means it emitted an answer for a question labeled unsupported. If it emits an answer to an answerable question but cites the wrong source or sentence, the evaluator reports that separately as a source/quote mismatch.

These fractions are counts from five hand-written invented questions. One question changes a rate by 20 percentage points in the unsupported group. The results diagnose these examples; they do not estimate how a real document-Q&A system would perform in general. Do not read any similarity score as confidence.

### 6. Why a larger window is not automatically better

A larger window can reduce missed answers because related information is not always placed in one sentence. It can increase false positives because a broad window may contain all of the right topic words even when the requested connection is missing. The trade-off depends on the documents, the question set, and what the rule considers evidence.

The simple clue matcher also has other limits:

- It sees a hand-written list of aliases, not the full meaning of a question.
- It can miss a new paraphrase if the words are absent from the clue groups.
- A clue can appear in a sentence with a different subject or role than the question asks about.
- Adjacent sentences are not guaranteed to describe one event or one relationship.
- Exact quote checking catches a mismatch with a label, but it cannot prove the label or rule is valid in the real world.

No language model was added. This remains a retrieval-first, extractive project. It does not yet have a free-form answer generator and should not be described as a complete RAG system.

### 7. Experiments and reflection

Keep the Session 3 clue groups fixed while reasoning about these results. Then answer in your own words:

1. Why did S4-Q1 need two sentences even though each individual sentence was useful?
2. Which two different claims are stated in the S4-Q3 quote? Which relationship did the question ask for that the quote did not state?
3. If a system must choose between abstaining often and occasionally giving an unsupported answer, what information would you need before choosing a policy?
4. Write a new question whose important relationship is missing from the handbook even though its nouns appear in one or two sentences. Do not add it to the current five-question results after seeing the output; preserve it as a future challenge.
5. Write a paraphrase that uses none of the existing clue aliases. Does the system abstain because no answer exists, or because the wording is unfamiliar? How could you tell?
6. Propose one way to represent who did what, to whom, and when. Explain how it might help and what language cases it could still misunderstand.

### 8. Completion checks for this session

- [x] Add a `max_window_sentences` setting for one sentence versus up to two adjacent sentences.
- [x] Write five fresh invented questions before changing any clue groups.
- [x] Include answerable paraphrases, an unsupported relationship trap, and unsupported details beside a matching fact.
- [x] Compare exact evidence, answerable abstentions, false positives, and correct refusals.
- [x] Keep reference and evaluation data distinct from model-training data; no model is trained.
- [x] Explain why exact quotation and logical support are different.

Project 4 remains open. The next checkpoint should investigate relationship-aware evidence checks with a new question set written before any rule changes. Keep the Session 4 challenge questions fixed as a regression comparison, and continue to avoid free-form generation until evidence behavior is evaluated more broadly.


# Session 5: Can we check who did what and when?

**Status:** Session 5 implementation and evaluation checkpoint complete; Project 4 remains active.

**Today’s question:** If two sentences contain the right nouns, how can the program check whether they describe the relationship the question asks for?

**What changed:** We wrote a new relationship-focused evaluation set and measured the old system before changing its rules. Then we added a small table of explicit relationships and a simple question-pattern matcher. The interactive program now uses that relationship path to quote one source sentence or abstain. The old clue and window approach remains available inside the evaluator for comparison.

No generative model was added. No model was trained.

## 1. From topic words to relationships

Suppose a question asks, “Does the shop steward store protective equipment before tool use?” The words *shop steward*, *protective equipment*, and *tool* all appear in the handbook. But the handbook says two different things:

1. Ask the shop steward before using a tool you have not used before.
2. Shop stewards can explain safe setup and help identify protective equipment.

It never says that a steward stores the equipment. A word-overlap rule can mistake nearby topics for evidence of an action.

A relationship record makes more of the sentence structure visible. A common beginner-friendly shape is:

```text
subject — action — object — time/condition — location
```

For example:

```text
shop stewards — explain — safe setup — (not stated) — (not stated)
```

Another source sentence can be represented as:

```text
orientation — offered — event — Wednesday evenings — (not stated)
```

The words **subject**, **action**, and **object** answer three different questions:

- **Subject:** who or what is doing something?
- **Action** (also called a predicate or relation): what is happening?
- **Object:** what receives the action, or what the action describes?

The extra fields retain time, condition, day, or location when the source states them. A missing field stays missing. In particular, a location is not automatically the actor. “Clay and glaze are sold at the front desk” states a location, but does not name the seller. Our table records that missing seller as `null`.

## 2. Where the relationships live

Open `data/handbook_facts.json`. Each row contains:

- `subject`, `action`, and `object`: the core relationship;
- `time_or_condition`, `days`, and `location`: optional details;
- `source_id`: the handbook record where the fact came from;
- `evidence`: an exact source sentence that can be quoted.

`load_handbook_facts` checks that each quote really occurs in the cited handbook text. This is a basic consistency check: it can catch a typo or a citation pointing to the wrong record. It cannot prove the annotation is a faithful interpretation. A person still decides which subject, action, and object the sentence expresses.

These fact rows are **reference annotations**. They are not used to adjust learned model weights. In fact, no model exists in this project yet. The handbook is invented reference text; the fact table is a hand-written structured view of that text; the three question files are invented evaluation data. The expected answer labels stay in the evaluator and are not sent into the answer function.

## 3. How a question reaches a quote

The new path has four visible steps:

```text
question text
    ↓ a few hand-written wording rules
requested relationship frame
    ↓ exact field checks
matching fact row
    ↓ verify source and quote
quoted sentence + citation, or abstention
```

A **frame** is the small set of relationship fields that the question appears to request. For example, the code maps “Before I use an unfamiliar tool, who should I consult?” to the action `ask_steward_before_new_tool`. That action maps to a source row whose object is `shop steward` and whose condition is before use of a tool that is new to the visitor.

For “Does the front desk cancel the booking two hours before it starts?”, the frame requests `cancel_booking_before_start` and requires the subject `front desk`. The cancellation fact has a null subject because the handbook does not specify who performs the cancellation. It also has no relationship tying the front desk to the cancellation act. So the action name alone is not enough: the requested subject constraint fails, and the system abstains.

The question parser is ordinary, visible Python rules in `relationship_frame`. It checks words and a handful of patterns. It is not a general language parser. The exact action names make the intended logic easy to inspect, but they also make the system narrow: a new way to ask the same question may not match any rule.

## 4. Write questions first, then measure the old behavior

Before changing any rules, we created `data/session5_relationship_questions.json`. It contains ten invented questions:

- six answerable questions about session duration, steward guidance, asking before tool use, orientation scheduling, cancellation timing, and orientation order;
- four unsupported questions about storing equipment, a front-desk cancellation role, Saturday hours, and an unstated seller.

The Session 4 questions remain unchanged in `data/session4_stress_questions.json`. That lets us compare the new experiment with the earlier evidence-window behavior.

First we ran the old clue matcher. On the new ten-question set:

| Existing method | Exact evidence on 6 answerable questions | False negatives (abstentions) | False positives on 4 unsupported questions | Correct abstentions |
| --- | ---: | ---: | ---: | ---: |
| Up to one sentence | 6/6 | 0/6 | 1/4 | 3/4 |
| Up to two sentences | 6/6 | 0/6 | 2/4 | 2/4 |

The one-sentence false positive was the front-desk question. The clue rule found cancellation timing in one sentence and accepted it even though that sentence does not assign the cancellation to the front desk. With two-sentence windows, the unsupported storage question also passed because the adjacent sentences together contained the steward, tool, setup, and protective-equipment clues. Those words still did not prove a storage action.

This is the key reasoning step: **retrieving the right source is not the same as confirming the requested relationship.** A quote can come from the expected document and still fail to support the claim.

## 5. What the relationship experiment returned

The explicit fact checker produced these counts on the same ten questions:

| Measure | Relationship-fact prototype |
| --- | ---: |
| Exact labeled source quote for answerable questions | 6/6 |
| False negatives (abstentions on answerable questions) | 0/6 |
| Wrong source or evidence on answerable questions | 0 |
| False positives on unsupported questions | 0/4 |
| Correct abstentions on unsupported questions | 4/4 |

The Session 4 regression comparison returned source-correct evidence spans for 3/3 answerable questions, abstained on 2/2 unsupported questions, and produced no wrong evidence. Two Session 4 quotes exactly matched their labels; for one question the fact checker returned a single-sentence quote contained inside the two-sentence expected label.

These numbers are **not** a reliability estimate. Although we wrote the new questions before editing the rules and recorded the old baseline first, we then used those question patterns to guide the relationship parser. The Session 5 set is therefore development material for the new system, not an independent blind test. Session 4 is a useful regression check, but its earlier questions and failure are already known. The next session must write a fresh challenge before making further changes and keep it unchanged while evaluating.

## 6. Important limits of an explicit fact table

A relationship table can make certain mismatches easier to see, but this version still has limits:

1. **Manual annotation can be wrong.** Someone must decide how each sentence maps to fields. The quote validator checks that words are present, not that the interpretation is correct.
2. **The parser has narrow vocabulary.** `relationship_frame` knows only a few visible patterns. “Who should I check with?” might mean the same thing as “who should I consult?” and still fail.
3. **The fact table is incomplete.** A missing row makes a supported question look unsupported. Abstention can be caused by missing knowledge as well as by truly absent evidence.
4. **Negation and question polarity need care.** For “Does orientation happen after tool use?”, the evidence says orientation must be completed before tool use. The prototype shows the relevant quote; it does not compose the word “No” or claim to generate a yes/no answer.
5. **One source sentence can contain more than one clause.** The 90-minute duration quote also mentions reservation notice. A more advanced system could split facts and preserve a smaller exact quote for each one.
6. **The sample is tiny and fictional.** The rates change sharply when one example changes and say nothing about real handbooks or users.

## 7. Experiments to try

Make one change at a time and predict the result before running the evaluator:

- Change “shop steward” to a new paraphrase such as “workshop guide.” Does the frame still match? Is an abstention a missing fact or unfamiliar wording?
- Change the required actor from `front desk` to a participant. What must the handbook actually say before that actor can be placed in the record?
- Change `days` from `weekdays` to a specific day. What evidence is needed before the system should claim Saturday hours?
- Temporarily remove the orientation fact. Does the answer path abstain even though the handbook sentence still exists? Why?
- Split the multi-clause session-duration sentence into smaller facts while keeping the exact source quote. Which evidence span should be returned for a question asking only how long a session lasts?
- Invent a new question whose subject, action, or timing is absent. Write it down before changing `relationship_frame` or `handbook_facts.json`.

## 8. Reflection questions

1. Why does the presence of “front desk,” “cancel,” and “two hours” in nearby text fail to prove that the front desk cancels the booking?
2. In the clay-sale record, why is `location="front desk"` different from `subject="front desk"`?
3. If the fact table has no seller, does that prove nobody sells the clay? What is the difference between “the source does not say” and “the event does not happen”?
4. Which parts of the relationship are extracted from text, and which were supplied by the person who wrote the fact record?
5. The new rule returned 6/6 exact quotes and abstained on 4/4 unsupported cases here. What new examples would make you less confident in that result?
6. Would you rather see a system abstain on unfamiliar wording or risk matching an incorrect relationship? What additional evaluation evidence would help you choose?

## 9. Session 5 completion checks

- [x] Write the fresh relationship question file before editing the relationship rules.
- [x] Record one-sentence and two-sentence baseline results first.
- [x] Represent subjects, actions, objects, times/conditions, days, locations, and source quotes explicitly.
- [x] Check that each structured evidence quote exists in its cited source.
- [x] Compare exact evidence, false positives, false negatives, wrong evidence, and correct abstentions.
- [x] Run the unchanged Session 4 questions as a regression comparison.
- [x] Keep the interactive path quote-based and avoid free-form generation.
- [x] State clearly that Session 5 was used to develop the new rules and is not a blind evaluation.

Project 4 remains active. For Session 7, write a genuinely fresh challenge set before changing the parser or fact records. Keep Sessions 4–6 unchanged as regression checks. Continue to avoid free-form model generation until the evidence checks have broader, less self-referential evaluation.


# Session 6: Who did what, and when?

## 1. The new challenge

A question can mention the right people and topic while asking for a relationship that the source never states. This session checks three slots together: who did something, what action happened, and when or under what condition it happened.

The ten questions in `data/session6_relationship_questions.json` were written before changing the parser. Four questions can be adjudicated by a source quote; six ask for unsupported exact relationships. They include unfamiliar paraphrases, actor swaps, a reversed time proposal, and missing agents. In this evaluation file, `answerable: true` means the handbook contains evidence that can resolve the question. That evidence can contradict a proposal. For example, a sentence saying “before” is useful counterevidence to a question that proposes “after.” The program only returns a source quote; it does not produce a yes/no answer.

All handbook text is invented reference material. The fact table is hand-written reference metadata tied to exact quotations. The questions and expected labels are evaluation data. None of these files trains a model. There is no learned model in this project.

## 2. Measure the old behavior before editing

The relationship matcher was run unchanged against the fresh questions. It returned exact expected evidence for 1/4 answerable questions and abstained on 3/4. Those three abstentions were false negatives: the expected quote existed, but the wording had no matching frame. On the six unsupported questions, the old matcher returned a quote for 4/6 (false positives) and correctly abstained on 2/6. It produced no wrong evidence on the answerable items because the failures there were abstentions.

The errors reveal two different problems:

- New wording such as “giving up a reserved time,” “show a first-time user,” and “check with” did not match the small set of known paraphrases.
- Actor-swapped questions could match the same topic and action words while naming the wrong doer. The sale sentence named a place but not the person selling the clay; treating that location as an agent created another false positive.

A metric is a count defined for a particular question set. Here, **false negative** means abstaining on an answerable item; **false positive** means quoting on an unsupported item; **wrong evidence** means quoting the wrong source or quote for an answerable item; and **correct abstention** means returning no quote for an unsupported item. Exact evidence and source-correct subspans are reported separately.

## 3. Follow one fresh paraphrase through the code

Question: “Who should a beginner check with before trying a tool they have never used?”

1. `tokenize` lowercases the question and splits it into simple words.
2. `relationship_frame` sees a small cue combination: `check`, a tool, and a before/first-use idea. It produces a frame asking for the `ask_steward_before_new_tool` action and the `shop steward` object.
3. `fact_matches_frame` compares the requested action and object against each reference fact. The fact table contains a matching row whose evidence is the exact sentence “Ask the shop steward before using a tool you have not used before.”
4. The answer function looks up that fact's labeled source document and returns its quotation. If no fact matches, it returns a short abstention reason.
5. `evaluate.py` compares the output to the expected source and exact quote. It uses the expected labels for measurement; `main.py` does not read them to answer the question.

The parser did not learn the phrase “check with” from examples. A person added a visible cue rule. This keeps the decision easy to inspect, but every supported paraphrase still depends on hand-written coverage.

## 4. Why actor direction matters

Compare these statements:

- “Ask the shop steward before using a tool.” The implied visitor asks; the steward is the person asked.
- “The shop steward asks the visitor before using a tool.” The participants have swapped roles.

The old matcher could return the first sentence for both because it noticed “steward,” “ask,” and “tool.” Session 6 adds a small check for selected named actors that appear before the action. If the question names a shop steward as the doer, but the fact's subject is an implied visitor, the fact no longer matches.

The same check catches a question that says every member explains safe setup: the source assigns that action to shop stewards. It also prevents a question from turning “sold at the front desk” into “the front desk sells.” In that fact, `location` is “front desk” and `subject` is `null`. A location tells where; a subject tells who or what performs the action.

This is a narrow position-based rule for a few roles, not a general grammar parser. Different sentence order, pronouns, passive voice, or a new participant may confuse it. The table itself can also omit a true relationship or contain a mistaken annotation.

## 5. Read the time reversal carefully

One fresh question asks whether members can use a powered tool first and complete the orientation afterward. The source quote says every member must complete orientation **before** using a powered tool. That is an exact and relevant quote, but it contradicts the proposed order.

The system does not generate “No.” It prints the quote and relationship fields. A person must compare the source's “before” with the question's “after.” The output should not be mistaken for a generated yes/no response or for proof that the matcher understood the question's polarity.

## 6. Results after the targeted rule changes

The hand-written fact table was left unchanged. The parser gained a few cues for giving up/calling off a booking, showing safe setup, and checking with someone before trying a tool. It also checks selected named actors before actions and recognizes questions that ask who performed an action.

| Set | Exact expected evidence | Source-correct spans | Wrong evidence | False negatives | False positives | Correct abstentions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Session 6 development questions | 4/4 answerable | 4/4 | 0 | 0/4 | 0/6 unsupported | 6/6 |
| Session 4 regression questions | 2/3 answerable | 3/3 | 0 | 0/3 | 0/2 unsupported | 2/2 |
| Session 5 regression/development questions | 6/6 answerable | 6/6 | 0 | 0/6 | 0/4 unsupported | 4/4 |

One Session 4 result is a source-correct sentence contained in the longer expected evidence window, so it is counted as a correct span but not an exact full quote. Session 5 wording informed earlier rule development. Session 6 wording guided the current changes. Therefore those results are development observations, not blind evaluation or broad reliability evidence. The examples are tiny and invented; one changed question could alter the counts sharply.

Reproduce all three sets and the earlier retrieval comparisons with:

```bash
python3 evaluate.py
```

## 7. Experiments and questions

Make predictions before changing code, and change one rule at a time:

- Replace “check with” with a new phrase. Does the answer abstain because the wording is unknown or because no matching fact exists?
- Swap the subject and object in a supported relation. Which field should reject the match?
- Remove an agent from a source sentence. Should the system guess who acted, or abstain?
- Reverse “before” and “after.” Is the returned quote support, contradiction, or irrelevant evidence?
- Change “weekdays” to “Saturday.” Which fact field carries the day information?
- Write an entirely new question set before changing another rule. Retain Sessions 4–6 unchanged as regression checks.

Reflection:

1. Why is a source-correct quotation not always an exact match to the requested relationship?
2. Why does “at the front desk” fail to identify who sold the clay?
3. Can an abstention mean “the words were unfamiliar,” “the fact table is incomplete,” or “the relation is absent”? How could you tell which happened?
4. What actor wording can the before-the-verb rule miss?
5. Why are expected quotes evaluation references and not training examples?
6. What evidence would you want before adding a model that writes free-form answers?

## 8. Session 6 completion checks

- [x] Write a fresh relationship-focused set before editing the parser.
- [x] Include paraphrases, actor swaps, time-order reversal, and missing-agent cases.
- [x] Record unchanged-baseline errors before making targeted changes.
- [x] Report exact evidence, source-correct spans, wrong evidence, false positives, false negatives, and correct abstentions separately.
- [x] Keep Sessions 4 and 5 unchanged and run both after edits.
- [x] Explain that the new set informed rule development and is not a blind test.
- [x] Keep answers extractive; no model training or free-form generation was added.

Session 6 is complete. Project 4 remains active. Session 7 should begin with a fresh evaluation set before any new parser or fact-table change, while preserving Sessions 4–6 as regression checks.
