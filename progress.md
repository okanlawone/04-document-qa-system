# Project Progress

## Overall goal

Build and understand a document question-answering system that retrieves relevant source passages, produces evidence-grounded responses, and exposes the cases where it is unsure or wrong.

## Milestone plan

### Session 1 — Retrieval foundation

- [x] Create a small, clearly labeled invented document collection.
- [x] Load and validate the document records.
- [x] Normalize text into simple word tokens.
- [x] Count words and rank passages with cosine similarity.
- [x] Display the passage and its source so the match can be inspected.
- [ ] Run several questions and record which ranking surprised you.

### Session 2 — Retrieval quality and answer behavior

- [ ] Create a small set of questions with expected source IDs.
- [ ] Measure whether the right source appears near the top.
- [ ] Compare the first word-count approach with an improved representation.
- [ ] Decide how to respond when no retrieved evidence supports an answer.
- [ ] Add a response layer while preserving visible source evidence.

### Later extensions (only after the foundations are understood)

- [ ] Split long documents into smaller passages and retain document/page metadata.
- [ ] Compare keyword retrieval with semantic embeddings.
- [ ] Add an optional language model and check every generated claim against retrieved evidence.
- [ ] Expand the evaluation set with paraphrases and unanswerable questions.

## Session 1 checkpoint

The current baseline is runnable and returns ranked evidence passages. It is not yet a complete RAG pipeline: there is no generative model, no embedding model, no training step, and no evaluation set. The handbook is invented lookup content, not model-training data.

## Reflection log

Add your own observations here after trying the program:

- A question that retrieved the expected passage:
- A question that retrieved an unexpected passage:
- Words that appeared in both the question and the passage:
- One change you want to investigate and why:

## Completion criteria

- The search logic can be explained in your own words, including what a token, word-count vector, dot product, vector length, and cosine score mean.
- The program surfaces evidence and source identifiers.
- A small evaluation set shows retrieval behavior on both answerable and unanswerable questions.
- The system communicates its limits instead of presenting unsupported guesses as facts.
