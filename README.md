# Document Q&A System — Retrieval-First Baseline

This project is inspired by **“Building a Document Q&A System,”** one of the retrieval and knowledge projects in [40 AI Projects That Show Real AI Engineering Skills](https://amanxai.com/2026/10/09/40-ai-projects-that-show-real-ai-engineering-skills/).

## The project challenge

Build a small system that can answer questions from a collection of documents and show the evidence it used. We will grow it in milestones so you understand each part instead of copying a large framework.

**Current milestone:** search a tiny collection of documents and return the closest passages with source labels. The program uses word counts and cosine similarity. It does not call an LLM, train a machine-learning model, or invent a paraphrased answer yet. Returning evidence first gives us something visible to inspect before we add more complexity.

## Practice data

`data/handbook.json` contains an **invented Northstar Community Workshop handbook** made for this project. It is fictional practice content, not real policy and not personal or customer data. These passages are the system's lookup knowledge; they are not a dataset used to train a model. No model is trained in this milestone.

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

Compare the returned passage, source, and score. The second question may score lower because this simple baseline recognizes shared words; it does not understand that “book” and “reserve” are related meanings.

## Learning path

- **Session 1 — Find evidence:** inspect the documents, turn words into counts, compare a question with each document using cosine similarity, and study where the ranking fails.
- **Session 2 — Improve and inspect answers:** add a small, hand-written question/evidence evaluation set; compare a word-count baseline with a better retrieval representation; define what the system should say when evidence is weak. We will choose a generation approach after seeing those limits.
- **Possible later extensions:** split longer documents into chunks, add semantic embeddings, generate answers from retrieved evidence, show more precise citations, and test unsupported questions.

A session is a learning milestone, not a promise that every extension is complete. We will only move on when the core ideas make sense.

## Completion goal

The project is complete when it can retrieve useful evidence from more than one document, show where the evidence came from, handle a question with no useful match honestly, and explain its known limitations using a small evaluation set.