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

### Session 4 — Evidence-window stress test

- [x] Write five fresh invented challenge questions before changing clue groups.
- [x] Include new answerable paraphrases and unsupported relationship/detail questions.
- [x] Compare one-sentence and up-to-two-adjacent-sentence windows with the same clue groups.
- [x] Measure exact expected evidence, answerable abstentions, unsupported answers, and correct abstentions.
- [x] Explain how a larger window recovers an answerable case and creates a false positive by joining nearby topic clues.
- [x] Default the interactive answer path to one sentence after observing the wider-window false positive; keep both settings in the evaluator.

### Next learning checkpoint — Test relationships

- [ ] Keep the five Session 4 questions as a fixed regression comparison.
- [ ] Write a second, fresh question set before changing rules.
- [ ] Investigate whether a relationship-aware representation can distinguish who did what and when.
- [ ] Report exact evidence, false positives, false negatives, and abstentions separately.
- [ ] Do not introduce free-form generation until evidence behavior is evaluated more broadly.

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


## Session 4 checkpoint

The Session 4 challenge set contains five fresh, hand-written invented evaluation questions: three are answerable from the handbook, and two ask for unsupported information or relationships. The Session 3 clue groups were not changed before or during this comparison. `evaluate.py` runs both a one-sentence configuration and an up-to-two-adjacent-sentence configuration.

- One-sentence windows returned the exact expected source quote for 2/3 answerable questions, abstained on the remaining answerable question (1/3 false negatives by the evaluation's abstention definition), and correctly abstained on 2/2 unsupported questions.
- Up-to-two-sentence windows returned exact expected source quotes for 3/3 answerable questions and abstained on 0/3 answerable questions. They also quoted a passage for 1/2 unsupported questions, while correctly abstaining on 1/2.
- The added false positive asks whether the shop steward stores protective equipment before tool use. The adjacent source sentences say to ask a steward before using a tool and that stewards can identify protective equipment. They do not say the steward stores equipment. Co-occurring clues are not proof of the requested action.

These are counts on five invented evaluation examples, not general performance estimates. The handbook remains invented reference data; both question files are invented evaluation data. No model is trained. Session 4 is complete, but Project 4 remains active because the rule still confuses clue coverage with relationship support.

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
- The learner can explain the measured one-sentence versus two-sentence trade-off, including the relationship false positive.
- The system communicates its limits instead of presenting unsupported guesses as facts.
