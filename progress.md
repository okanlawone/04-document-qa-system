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
- [x] Run several questions using a labeled evaluation set and record the observed ranking behavior.

### Session 2 — Retrieval quality and answer behavior

- [x] Create a small set of questions with expected source IDs.
- [x] Measure whether the right source appears near the top.
- [x] Compare the first word-count approach with TF-IDF weighting.
- [x] Decide how to respond when no retrieved evidence supports an answer.
- [ ] Add an extractive response layer with visible source evidence and an abstention path.

### Later extensions (only after the foundations are understood)

- [ ] Split long documents into smaller passages and retain document/page metadata.
- [ ] Compare keyword retrieval with semantic embeddings.
- [ ] Add an optional language model and check every generated claim against retrieved evidence.
- [ ] Expand the evaluation set with paraphrases and unanswerable questions.

## Session 1 checkpoint

The current baseline is runnable and returns ranked evidence passages. Session 2 adds ten hand-written evaluation questions and a retrieval comparison. On this small set, raw word counts put the expected source first on 3/7 answerable questions; TF-IDF did so on 1/7. Both methods found the expected source within the top three on 7/7 answerable questions, and both returned a passage for all 3/3 unanswerable questions. These are diagnostic results for this invented set, not general performance claims.

The project is not yet a complete RAG pipeline: there is no generative model, embedding model, training step, or calibrated abstention rule. The handbook is invented lookup content; the question set is invented evaluation data. Neither is model-training data.

## Session 2 checkpoint

Keep raw word counts as the current baseline because they ranked the expected source first more often than TF-IDF on this question set. The next answer layer will be extractive: quote a supporting source sentence and cite its record, or say the handbook does not provide enough support. Do not add free-form generation until evidence sufficiency and abstention are addressed.

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
