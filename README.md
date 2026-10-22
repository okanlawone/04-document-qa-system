# Document Q&A System — Retrieval and Evidence Workshop

This project is inspired by **“Building a Document Q&A System,”** one of the retrieval and knowledge projects in [40 AI Projects That Show Real AI Engineering Skills](https://amanxai.com/2026/10/09/40-ai-projects-that-show-real-ai-engineering-skills/).

## The project challenge

Build a small system that finds source passages for questions, checks whether the requested relationship is present, and either quotes the evidence or abstains. The work proceeds in sessions so each stage can be understood before more machinery is added.

**Current learning milestone:** Tomi is restarting Project 4 from first principles because the earlier lessons did not stick. Begin with [`restart/lesson_01.md`](restart/lesson_01.md) and its runnable [`restart/lesson_01_evidence_lookup.py`](restart/lesson_01_evidence_lookup.py). Sessions 1–6 remain preserved as project history; they are not evidence that the learner understands the material. The full system still quotes source evidence or abstains; it does not generate a yes/no answer, use an LLM, use embeddings, or train a machine-learning model.

## Practice data and labels

- `data/handbook.json` is an invented Northstar Workshop handbook used only as reference text. It is not real guidance.
- `data/handbook_facts.json` contains hand-written relationship annotations tied to exact source quotes. These annotations are reference metadata, not model-training examples.
- `data/evaluation_questions.json`, `data/session4_stress_questions.json`, `data/session5_relationship_questions.json`, and `data/session6_relationship_questions.json` contain hand-written evaluation questions and labels. Expected sources and quotes are evaluation references; the answer code does not read them.
- No model is trained. The tiny invented sets diagnose examples and do not establish general performance.

## Run it

You need Python 3. The project uses only Python's standard library.

```bash
python3 main.py
```

Try a supported question:

```text
Before I use an unfamiliar tool, who should I consult?
```

Then try a relationship trap:

```text
Does the front desk cancel the booking two hours before it starts?
```

For the first question, the program quotes the source sentence and shows the relationship fields. For the second, it abstains: the handbook mentions cancellation timing and the front desk, but it does not say the front desk performs the cancellation. The system returns evidence, not a generated yes/no answer.

Reproduce all retrieval, topic-clue, window-size, and relationship evaluations with:

```bash
python3 evaluate.py
```

## Fresh-start path

Start with the small evidence-lookup exercise before using the older session guides:

```bash
python3 restart/lesson_01_evidence_lookup.py
```

The lesson explains every new term and code section, works through the phone-renewal example, and ends with a learner checkpoint. Continue only after the checkpoint is discussed.

## Learning path

- **Session 1 — Find evidence:** tokenize text, count words, rank passages with cosine similarity, and inspect source citations.
- **Session 2 — Measure retrieval:** compare raw word counts with TF-IDF. On the original seven answerable questions, raw counts ranked the expected source first on 3/7 and TF-IDF on 1/7; both placed it in the top three on 7/7. Both returned a candidate for all 3/3 unanswerable questions.
- **Session 3 — Check evidence before answering:** add manually written topic clues and an extractive response with an abstention path. On that 13-question invented set, the expected quote was returned for 7/7 answerable cases and the system abstained for 6/6 unanswerable cases. These are diagnostic counts, not general performance estimates.
- **Session 4 — Stress-test evidence windows:** with the same clue groups, one-sentence windows quoted the expected evidence for 2/3 answerable questions, abstained on 1/3 answerable questions, and correctly abstained on 2/2 unsupported questions. Up-to-two-sentence windows quoted 3/3 answerable cases but produced a false positive on 1/2 unsupported cases by joining nearby clues without proving that the steward stored equipment.
- **Session 5 — Represent relationships:** the new ten-question set has six answerable and four unsupported questions. Before changing the code, the existing one-sentence clue method produced exact expected quotes for 6/6 answerable questions and one false positive among four unsupported questions. The two-sentence version produced exact expected quotes for 6/6 and two false positives among four unsupported questions. The relationship-fact prototype produced exact expected quotes for 6/6 and abstained on all 4/4 unsupported questions. On the preserved Session 4 set, it returned source-correct evidence for 3/3 answerable questions (one quote was a labeled subspan) and abstained on 2/2 unsupported questions.

Session 5 wording informed its rules; Session 6 questions were written before the latest parser changes and then guided those changes. Therefore the post-change Session 6 results are development observations, not a blind test. All results come from small invented examples. The detailed guide explains the baseline, the evaluation categories, and what the evidence can and cannot establish.

## What the relationship fields mean

A **relationship** records which participant did what, to what, and under which conditions. In `handbook_facts.json`, each record has a subject, action, object, optional time or condition, optional days or location, a source ID, and an exact evidence quote. A `null` subject means the source did not state who performed the action. The matcher checks these fields instead of treating all nearby topic words as proof.

This is still a narrow hand-built prototype. Its question parser recognizes a few visible wording patterns, and the fact records were annotated by a person. It can abstain on a supported question if the wording is new, and a mistaken annotation or overly broad rule can still produce an error. The project does not claim human-like understanding.

## Next learning checkpoint

Keep the Session 4, Session 5, and Session 6 question files unchanged as regression checks. Session 7 should start with a new relationship-focused evaluation set before any rule changes, probing roles and time direction in fresh wording. Measure exact evidence, wrong evidence, false negatives, false positives, and correct abstentions separately. The current Session 6 results informed rule development and are not blind-test evidence. Do not add free-form generation until evidence behavior has been evaluated more broadly.


## Latest daily build — October 20, 2026

The fresh-start track now includes the learner-authored [`restart/learner_attempt_01.py`](restart/learner_attempt_01.py) and a step-by-step [`return` versus `print` walkthrough](restart/lesson_01_return_and_print.md). The practice version is labeled and intentionally limited; publication records project work, not proof of mastery.

## Latest daily build — October 21, 2026

A runnable [return-versus-print trace](restart/return_print_trace.py) and its [line-by-line walkthrough](restart/lesson_02_return_print_trace.md) show what the calling code receives when a function prints instead of returning a value. The learner check remains open; do not advance to another concept based on publication.