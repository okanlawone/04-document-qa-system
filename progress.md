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
- [x] Add an extractive response layer with visible source evidence and an abstention path.

### Session 3 — Evidence sufficiency and cautious answers

- [x] Add transparent, hand-written topic clue groups.
- [x] Quote one or two original adjacent source sentences and identify the record.
- [x] Abstain when no single title-plus-evidence window covers every detected question clue.
- [x] Expand the invented evaluation set from 10 to 13 questions with three overlap traps.
- [x] Measure exact expected evidence, answerable coverage, correct abstention, and unsupported answers.
- [x] Explain how this lexical coverage rule can still miss or falsely approve evidence.

### Next learning checkpoint — Challenge the heuristic

- [ ] Write unseen paraphrases before changing clue groups.
- [ ] Add an unsupported question that uses all the right topic words but asks an unstated relationship.
- [ ] Compare one-sentence and two-sentence evidence windows.
- [ ] Explain a false positive and a false negative without relying on a score as confidence.

### Later extensions (only after the foundations are understood)

- [ ] Split long documents into smaller passages and retain document/page metadata.
- [ ] Compare keyword retrieval with semantic embeddings.
- [ ] Consider optional language-model generation only after evidence checking is stronger; verify every generated claim against retrieved evidence.
- [ ] Keep new evaluation questions separate from examples used while adjusting the rules.

## Session 1 checkpoint

The baseline loads five invented handbook passages, tokenizes words, counts them, and ranks passages with cosine similarity. Session 2 added a ten-question hand-written evaluation set and compared raw word counts with TF-IDF. On the original seven answerable questions, raw counts ranked the expected source first on 3/7; TF-IDF did so on 1/7. Both placed the expected source within the top three on 7/7. Both returned a passage on all three original unanswerable questions. These are diagnostic results for this invented set, not general performance claims.

## Session 2 checkpoint

Keep raw word counts as the transparent baseline because they ranked the expected source first more often than TF-IDF on the original questions. Retrieval ranking is not answer support. Similarity scores are not calibrated confidence. The next layer should quote source text and cite its record or abstain.

## Session 3 checkpoint

`main.py` now checks manually defined clue groups against one sentence or two adjacent sentences plus the record title. If every detected group has a clue in that window, the program quotes the original text and cites its record. If not, it abstains and may display the closest passage as a non-answer.

The expanded evaluation contains seven answerable and six unanswerable questions. In the current run, the extractive layer returned the labeled source quote for 7/7 answerable questions and abstained on 6/6 unanswerable questions. Raw retrieval still returned some candidate for 6/6 unanswerable questions. The rule is tuned and checked on a tiny, invented, hand-written set; these results are not broad reliability evidence. No model was trained. The handbook is reference data; the questions and expected quotes are evaluation data.

## Reflection log

Add your own observations here after trying the program:

- A question that retrieved the expected passage:
- A question that retrieved an unexpected passage:
- Words or clue groups that appeared in both the question and evidence:
- One case where lexical clues did not actually prove the requested relationship:
- One change you want to investigate and why:

## Completion criteria

- The search logic can be explained in your own words, including what a token, word-count vector, dot product, vector length, and cosine score mean.
- The program surfaces quoted evidence and source identifiers or explains why it abstained.
- A small evaluation set shows retrieval and response behavior on answerable and unanswerable questions.
- The learner can explain the difference between lexical clue coverage and logical evidence sufficiency.
- The learner can describe the limits of this hand-written evaluation and propose a fresh challenge question.
- The system communicates its limits instead of presenting unsupported guesses as facts.
