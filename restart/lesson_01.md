# Fresh-start track — Lesson 1: What counts as evidence?

**Project:** Document Q&A System  
**Starting point:** assume no AI, Python, or retrieval knowledge  
**Today’s one idea:** a source supports an answer only when it connects the requested action to the requested method  
**Code:** [`lesson_01_evidence_lookup.py`](lesson_01_evidence_lookup.py)  
**Practice data:** invented example text; it is not real policy and is not model-training data.

> Start here if the earlier sessions felt too fast. The old sessions remain in the repository as project history. Their publication does not mean you learned them.

## 1. The problem in everyday language

Suppose a document says:

> “You can renew online. Call support with billing questions.”

Someone asks, “Can I renew by phone?”

The document contains both a renewal sentence and a phone-related sentence. But the phone sentence is about **billing help**. Nothing in it connects a phone call to the **renewal action**. So the careful result is: **the document does not state that phone renewal is available**. That wording is precise: it does not claim that phone renewal is impossible; it says this source has not established it.

This is the first foundation of document question answering: finding text that sounds related is not enough. Evidence must support the specific relationship the question asks about.

## 2. Before the code: a few plain-language terms

- **Program:** instructions a computer follows.
- **Data:** information the program works with. Here, the invented document and its fact records are data.
- **Source:** the original text we want to rely on.
- **Evidence:** a source passage that supports a particular claim.
- **Action:** what someone wants to do, such as renew.
- **Method:** how they want to do it, such as online or by phone.
- **Match:** the action and method in one source fact are the same as the action and method asked about.
- **Abstain:** decline to claim an answer when the supplied evidence is not enough.

We will first make the relationship explicit in a small table-like structure. Later project lessons can work toward finding those facts from ordinary paragraphs.

## 3. Run the first practical prototype

From the project’s top folder, run:

```bash
python3 restart/lesson_01_evidence_lookup.py
```

The program prints three examples. The first finds evidence for website renewal. The second does not find evidence for phone renewal. The third finds a phone sentence, but for billing help. That third example matters: a phone mention alone does not support every action someone might do by phone.

## 4. Read the code slowly

Open `restart/lesson_01_evidence_lookup.py`. We will trace its data instead of memorizing the spelling.

### The source text

```python
DOCUMENT = "You can renew online. Call support with billing questions."
```

`DOCUMENT` is a name for a piece of text. The quotation marks tell Python that the value is text. Programmers call a piece of text a **string**. The name is written in capital letters by convention to signal that this sample value is treated as fixed while the script runs.

### The fact records

```python
FACTS = [
    {"action": "renew", "method": "website", "evidence": "You can renew online."},
    {"action": "billing_help", "method": "phone", "evidence": "Call support with billing questions."},
]
```

The square brackets `[...]` make a **list**: an ordered collection of items. Each item here is a **dictionary**, written with curly braces `{...}`. A dictionary stores values under names called **keys**. For example, the first dictionary says:

- key `action` has value `renew`;
- key `method` has value `website`;
- key `evidence` has the exact quote from the source.

The second record connects `phone` to `billing_help`. It does **not** connect `phone` to `renew`. Keeping those facts separate prevents the program from borrowing a nearby phone mention to answer the wrong question.

A person wrote these records from the source. This starter does not read the paragraph and discover facts by itself. That limitation is intentional: it lets us inspect the rule clearly before building retrieval or language understanding.

### The evidence-finding function

```python
def find_evidence(requested_action, requested_method):
    for fact in FACTS:
        same_action = fact["action"] == requested_action
        same_method = fact["method"] == requested_method

        if same_action and same_method:
            return fact

    return None
```

Line by line:

1. `def` starts a **function**, a named set of instructions we can run when needed. `find_evidence` is the function’s name.
2. `requested_action` and `requested_method` are inputs to the function. For the phone-renewal question, they will contain `renew` and `phone`.
3. `for fact in FACTS` visits the records one at a time. First `fact` means the website-renewal record; next it means the phone-billing record.
4. `fact["action"] == requested_action` checks whether the action values are equal. The double equals `==` asks a question; a single equals `=` stores a value.
5. `same_method` makes the corresponding comparison for the method.
6. `if same_action and same_method` means both comparisons must be true for this record. Matching just one field is not enough.
7. `return fact` sends the matching record back to the code that called the function.
8. If the loop checks every record and finds no match, `return None` reports that there was no matching fact. `None` is Python’s way to represent “no value found.” In this small example, we interpret that as “not stated in the supplied fact records.”

### Turning a found fact into a cautious result

```python
def answer_from_facts(question, requested_action, requested_method):
    fact = find_evidence(requested_action, requested_method)

    if fact is None:
        return f"Question: {question}\nResult: The document does not state this."

    return f"Question: {question}\nEvidence: {fact['evidence']}"
```

The function asks `find_evidence` for a matching fact. If no fact came back, it returns a cautious message. Otherwise, it prints the exact quote stored in the record. The `f` before a quoted string lets Python insert values such as `{question}` into the text. `\n` means “start a new line.”

This prototype returns evidence or abstains. It does not generate a yes/no answer. That keeps us focused on the source.

### The three examples

```python
examples = [
    ("Can I renew on the website?", "renew", "website"),
    ("Can I renew by phone?", "renew", "phone"),
    ("Can I get billing help by phone?", "billing_help", "phone"),
]
```

Each **tuple** in this list holds a question and the two fields a person assigned to that question. For the second question, those fields are `renew` and `phone`. The code does not infer them from the English sentence; we supply them so we can study the matching rule by itself.

The final `for` loop calls the answer function on each example and displays the result. That is why the second example abstains: no single fact record has both `action` equal to `renew` and `method` equal to `phone`.

## 5. Trace the key example by hand

Question: **Can I renew by phone?**

| Step | What the program checks | Result |
| --- | --- | --- |
| 1 | Requested action is `renew` | — |
| 2 | Requested method is `phone` | — |
| 3 | First fact: action `renew` matches; method `website` does not match `phone` | Not a full match |
| 4 | Second fact: action `billing_help` does not match `renew`; method `phone` matches | Not a full match |
| 5 | No record matches both fields | Return `None` |
| 6 | No fact was found | Say the document does not state this |

The phrase “Call support” appears in the source, but it belongs to the billing-help fact. The program does not treat that as proof about renewal.

## 6. What this prototype can and cannot do

It can:

- keep a source quote attached to the fact it supports;
- require both the requested action and method to match;
- show the quote or abstain when no fact matches.

It cannot:

- read a plain-English question and understand its meaning; a person supplies the action and method fields;
- create the fact records from the document; a person wrote them by hand;
- prove a real policy from this invented example;
- conclude that phone renewal is impossible just because this tiny table has no phone-renewal record;
- generalize to other wording, actors, times, or documents.

This is not a trained AI model. The fact table is reference data, and the three examples are hand-written demonstrations. Nothing here is training data or a performance benchmark.

## 7. Small experiments

Try these one at a time, then explain what changed:

1. In the second example, change the method from `phone` to `website`. Which quote appears?
2. Add a new practice fact with action `renew`, method `phone`, and a quote that explicitly says renewal can happen by phone. Does the second example now return that quote?
3. Change only the action in that new record to `billing_help`. Does it still answer the renewal question? Why not?

Use only invented sentences in these experiments. Adding a made-up fact demonstrates the code; it does not make it true in a real policy.

## 8. Your learning checkpoint

Before moving on, explain in your own words:

1. Why does the phone-billing sentence fail to prove phone renewal?
2. What must match for the program to return a quote?
3. Does “the document does not state this” mean “phone renewal is definitely impossible”?

You can answer casually. I’ll respond to your explanation before we add the next piece.

## Where the project goes next

After this idea is clear, we can build the next small layer: split a short document into sentences and keep each sentence attached to its source. Later, we can retrieve likely sentences, map natural-language questions to action/method fields, and evaluate mistakes. We will not skip the foundations or treat published code as proof that you understand it.
