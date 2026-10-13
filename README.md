# Document Q&A System — Retrieval-First Baseline

This project is inspired by **“Building a Document Q&A System,”** one of the retrieval and knowledge projects in [40 AI Projects That Show Real AI Engineering Skills](https://amanxai.com/2026/10/09/40-ai-projects-that-show-real-ai-engineering-skills/).

## The project challenge

Build a small system that can answer questions from a collection of documents and show the evidence it used. We will grow it in milestones so you understand each part instead of copying a large framework.

**Current milestone:** compare two small retrieval methods against a hand-labeled question set. The program shows passages and sources instead of inventing answers. It does not call an LLM or train a machine-learning model.

## Practice data

`data/handbook.json` contains an **invented Northstar Community Workshop handbook** made for this project. It is fictional practice content, not real policy and not personal or customer data. These passages are the system's lookup knowledge; they are not a dataset used to train a model. `data/evaluation_questions.json` is a separate, hand-written practice set with expected evidence IDs and questions the handbook cannot answer. Neither file trains a model, and ten questions cannot prove broad performance.

## Run it

You need Python 3. The first milestone uses only Python's standard library, so no package installation is required.

```bash
python3 main.py
```

When prompted, try a question such as:

```text
How early should I reserve a woodshop session?
```

Then try a question using different wording:

```text
When do I need to book the workshop?
```

Compare the returned passage, source, and score. The simple baseline recognizes shared words; it does not understand that “book” and “reserve” are related meanings.

To reproduce the Session 2 comparison, run:

```bash
python3 evaluate.py
```

The evaluator compares raw word-count cosine similarity with TF-IDF cosine similarity. TF-IDF gives relatively more weight to words that occur in fewer handbook passages. Both methods still depend on word overlap.

## Learning path

- **Session 1 — Find evidence:** inspect the documents, turn words into counts, compare a question with each document using cosine similarity, and study where the ranking fails.
- **Session 2 — Measure retrieval:** compare raw word counts with TF-IDF using seven answerable and three unanswerable questions. On this small set, raw counts ranked the expected source first for 3/7 answerable questions; TF-IDF did so for 1/7. Both placed the expected source in the top three for 7/7 answerable questions, and both returned a passage for all 3/3 unanswerable questions. These are diagnostic results for this invented set, not general performance claims.
- **Next response milestone — Quote evidence carefully:** first build a short extractive answer from an original supporting sentence and show its source. If the system cannot find support, it should say so. We are not adding free-form LLM generation yet because current rankings and unanswerable behavior need more work; similarity scores are not calibrated probabilities.
- **Possible later extensions:** split longer documents into chunks, add semantic embeddings, generate answers from retrieved evidence, show more precise citations, and test unsupported questions.

A session is a learning milestone, not a promise that every extension is complete. We will only move on when the core ideas make sense.

## Completion goal

The project is complete when it can retrieve useful evidence from more than one document, show where the evidence came from, handle a question with no useful match honestly, and explain its known limitations using a small evaluation set.
