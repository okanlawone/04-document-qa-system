# Project Progress

## First-principles restart

- [x] Preserve the published Sessions 1–6 and their evaluation history.
- [x] Add a beginner restart lesson and a small runnable evidence-lookup prototype.
- [x] Discuss the Lesson 1 checkpoint with Tomi before advancing.
- [x] Publish the learner-authored action/method lookup attempt and a beginner walkthrough of `return` versus `print` (October 20, 2026).
- [x] Publish a runnable return-versus-print trace with a line-by-line walkthrough (October 21, 2026).
- [ ] Discuss the return/print learner check before advancing to another concept.
- [ ] Continue building and publishing one explained project layer at a time.

The prior sessions are completed repository milestones, not proof that Tomi learned their contents. On October 20, Tomi supplied a first Python lookup implementation; the published [learner attempt](restart/learner_attempt_01.py) preserves it, and a [return-versus-print guide](restart/lesson_01_return_and_print.md) explains its current flow. The return-versus-print learner check remains open. On October 21, a runnable trace and a second worked explanation were added within that same concept; do not advance based on publication. Session 7 is deferred until the foundations have been rebuilt.

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

### Session 5 — Represent relationships explicitly

- [x] Write ten fresh relationship-focused questions before changing the answer rules.
- [x] Record the original one-sentence and two-sentence clue baselines first.
- [x] Add hand-written subject, action, object, time/condition, day, and location facts tied to exact source quotes.
- [x] Require requested relationship fields to match before quoting evidence.
- [x] Keep the question labels out of the answer function and validate fact quotes against their cited source.
- [x] Report exact evidence, false positives, false negatives, wrong evidence, and correct abstentions separately.
- [x] Run the unchanged Session 4 set as a regression check.
- [x] Document that Session 5 was used to shape the new rules and is not a blind evaluation.
- [x] Keep answers extractive and add no free-form generation.

### Historical planned checkpoint — Stress-test relationships

- [ ] Keep the Session 4 and Session 5 question files unchanged as regression checks.
- [ ] Write a fresh Session 6 relationship set before changing the parser or fact annotations.
- [ ] Include unseen paraphrases, actor swaps, time-order reversals, and missing-agent questions.
- [ ] Measure exact evidence, wrong evidence, false positives, false negatives, and correct abstentions separately.
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


## Session 5 checkpoint — relationship-aware evidence

`data/session5_relationship_questions.json` contains six answerable and four unsupported invented questions. Before editing the rules, the existing topic-clue system returned exact expected evidence for 6/6 answerable items; its one-sentence configuration produced 1/4 false positives and 3/4 correct abstentions, while the two-sentence configuration produced 2/4 false positives and 2/4 correct abstentions. The false positives assigned cancellation to the front desk and storage to the steward without source support.

`data/handbook_facts.json` now holds manually authored subject/action/object records, optional time/day/location fields, source IDs, and exact quotes. `load_handbook_facts` checks that each quote appears in its labeled document. `relationship_frame` maps a few visible question patterns into requested relationship fields; `fact_matches_frame` requires the requested fields to match. The CLI quotes a matching source sentence and its fields, or abstains. It does not generate a yes/no answer.

On Session 5, this prototype returned exact labeled evidence for 6/6 answerable questions, abstained on 0/6 answerable questions, produced 0/4 false positives, and correctly abstained on 4/4 unsupported questions. On the unchanged Session 4 set, it returned source-correct evidence spans for 3/3 answerable cases (two exact full labels and one exact subspan), abstained on 2/2 unsupported cases, and produced no wrong evidence. These are results on tiny invented examples. Session 5's question wording influenced rule design, so these results are development observations rather than blind-test evidence. All handbook/fact-table content is reference data, question labels are evaluation data, and no model is trained.

Session 5 is complete. Project 4 remains active. Session 6 must write a fresh relationship challenge before changing rules, retain Sessions 4 and 5 as regression checks, and continue to exclude generative answers until evidence behavior has broader evaluation.


## Session 6 checkpoint — actor roles, paraphrases, and time order

`data/session6_relationship_questions.json` was written before changing the parser. It has four questions a handbook quote can adjudicate and six unsupported exact relationships. The old matcher returned exact evidence on 1/4 answerable items, abstained on 3/4 (false negatives), returned quotes for 4/6 unsupported items (false positives), correctly abstained on 2/6, and produced no wrong source/evidence on answerable items. The false positives came from actor swaps and treating a sales location as a named seller.

The hand-authored fact table stayed unchanged. The parser now recognizes a few new cancellation, safe-setup, and ask/check paraphrases; checks selected named subjects before actions; and detects questions explicitly asking who performed an action. On Session 6, the updated system returned exact evidence for 4/4 answerable questions, abstained on 6/6 unsupported questions, and had 0 wrong evidence, 0 false negatives, and 0 false positives. The time-reversal case received the exact quote stating orientation must come before powered-tool use; the quote is counterevidence to the question's proposed order, and the system still does not generate a yes/no answer.

Regression checks after the changes: Session 4 returned exact evidence on 2/3 answerable questions and source-correct evidence spans on 3/3 (one quote was a labeled subspan), with 0 wrong evidence, 0/3 false negatives, and 0/2 false positives; it correctly abstained on 2/2 unsupported items. Session 5 returned exact evidence on 6/6, with 0 wrong evidence, 0/6 false negatives, 0/4 false positives, and 4/4 correct abstentions.

The Session 6 questions informed the parser changes, so post-change results are development observations, not a blind evaluation. Session 5 also informed earlier rule development. All handbook and fact records remain invented reference data; all question sets are evaluation data, not training data. No model was trained, and no free-form answer generation was added.

Session 6 is complete. Project 4 remains active. For Session 7, write a new relationship-focused evaluation set before any parser or fact edits; retain Sessions 4–6 as regression checks and report exact evidence, wrong evidence, false positives, false negatives, and correct abstentions separately.