# Document Q&A System — Retrieval-First Baseline

This project is inspired by **“Building a Document Q&A System,”** one of the retrieval and knowledge projects in [40 AI Projects That Show Real AI Engineering Skills](https://amanxai.com/2026/10/09/40-ai-projects-that-show-real-ai-engineering-skills/).

## The project challenge

Build a small system that can answer questions from a collection of documents and show the evidence it used. We will grow it in milestones so you understand each part instead of copying a large framework.

**Current milestone:** stress-test a cautious extractive evidence rule using one-sentence and up-to-two-sentence windows. The interactive program defaults to one-sentence evidence and quotes its source or abstains; the evaluator also compares a two-sentence option. A fresh challenge set exposes both missed answers and unsupported relationships. It does not call an LLM or train a machine-learning model.

## Practice data

`data/handbook.json` contains an **invented Northstar Community Workshop handbook** made for this project. It is fictional practice content, not real policy and not personal or customer data. These passages are the system's lookup knowledge; they are not a dataset used to train a model. `data/evaluation_questions.json` holds the original 13 hand-written evaluation questions; `data/session4_stress_questions.json` holds five fresh Session 4 challenge questions. Expected evidence labels are evaluation references, not training targets. No file trains a model, and these small sets cannot prove broad performance.

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

To reproduce the retrieval, answer, and evidence-window evaluations, run:

```bash
python3 evaluate.py
```

The evaluator compares raw word-count cosine similarity with TF-IDF, repeats the Session 3 answer checks, and compares one-sentence with up-to-two-sentence evidence windows on the Session 4 challenge set. TF-IDF gives relatively more weight to words that occur in fewer handbook passages. Both retrieval methods still depend on word overlap.

## Learning path

- **Session 1 — Find evidence:** inspect the documents, turn words into counts, compare a question with each document using cosine similarity, and study where the ranking fails.
- **Session 2 — Measure retrieval:** compare raw word counts with TF-IDF. On the original seven answerable questions, raw counts ranked the expected source first on 3/7 and TF-IDF on 1/7; both placed it in the top three on 7/7. Both retrieval methods returned a passage for all three original unanswerable questions.
- **Session 3 — Check evidence before answering:** quote one or two original adjacent sentences with a record citation only when every manually detected question-topic group appears in that evidence window and title; otherwise abstain. Across the 13-question invented evaluation set, the code returned the labeled source quote for 7/7 answerable questions and abstained for 6/6 unanswerable questions. Retrieval alone still returned candidates for all 6/6 unanswerable questions. These are results on a tiny, hand-written set, not evidence of general reliability.
- **Session 4 — Stress-test evidence windows:** the five-question invented challenge set was evaluated with the Session 3 clue groups unchanged. One-sentence windows produced exact expected quotes for 2/3 answerable questions, abstained on 1/3 answerable questions, and correctly abstained on both unsupported questions. Up-to-two-sentence windows produced exact expected quotes for 3/3 answerable questions, but also quoted evidence for 1/2 unsupported questions by combining nearby clues without supporting the requested storage action. These tiny hand-written results are diagnostic examples, not general performance estimates. The interactive program defaults to one sentence after the two-sentence configuration produced an unsupported quote on this set; one-sentence mode still misses one answerable paraphrase.
- **Next learning checkpoint — Check relationships:** preserve the Session 4 set as a regression check, write another fresh set before changing rules, and investigate how a system can distinguish who did what and when rather than only checking whether topic clues co-occur.
- **Later experiments:** split longer documents into chunks and compare keyword retrieval with semantic embeddings. Consider free-form generation only after the evidence checks are better understood; verify every generated claim against a source.

The clue groups are manually written, not learned language understanding. They can miss paraphrases or approve words that do not logically support an answer. This project is still a retrieval-first extractive system, not a complete RAG system: it does not include a generator. No model has been trained.

## Completion goal

The project is complete when it can retrieve useful evidence from more than one document, show where the evidence came from, handle a question with no useful match honestly, and explain its known limitations using a small evaluation set.
