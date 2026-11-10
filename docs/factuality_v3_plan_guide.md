# Writing source-only expected-fact plans (v3)

A plan lists the facts a competent clinician would put in **the requested section** of a note written from this dialogue. Plans are written from the dialogue alone, before you see any summary. Later, each summary is checked against its plan: a planned fact the summary leaves out is an **omission**. So the plan defines what "omitted" means. List what a good section *should* contain, not everything the dialogue says.

## The decision for each candidate fact

Ask three questions in order:

1. **Is it about this patient and actually established?** It must be stated by the patient or family, or stated or confirmed by the clinician. A clinician question with no answer is not a fact. A "no" answer to a relevant question *is* a fact (a pertinent negative, e.g. "denies chest pain").
2. **Does it belong in the requested section?** Use the section in the page header. A medication list belongs in *medications*, not *chief complaint*. When unsure, picture the section heading in a real note.
3. **Would leaving it out make the section worse?** Include it if a clinician reading the section would reasonably expect it. Leave out small talk, logistics and pleasantries.

Three yeses → add the fact as **pending** (expected). Otherwise, either:

- **Leave it out entirely.** This is the default for clearly irrelevant material; don't list it.
- **Add it as excluded, with a rationale.** Use this only for **borderline** facts a summary might plausibly include but that shouldn't count as an omission: out-of-section, hypothetical, ambiguous or conflicting. Excluded facts never count as omissions. They record that you considered the fact.

## How big is one fact?

**One fact = one thing a summary could get right or leave out on its own.** Use this split consistently:

| Source says | Facts |
|---|---|
| "I take metformin 500 mg twice a day" | (1) takes metformin — `medication`; (2) metformin 500 mg twice daily — `quantity` |
| "Sharp pain in my right knee for three weeks" | (1) right knee pain — `symptom`; (2) duration three weeks — `date` |
| "No fever, no chills" | (1) denies fever — `symptom`; (2) denies chills — `symptom` |
| "I'm 58" (HPI or intro section) | (1) age 58 — `age` |
| "My mother had breast cancer" (family history) | (1) mother: breast cancer — `diagnosis`, subject `relative` |

Write the fact text in plain words: who, what, and any qualifier (negation, side, time, dose).

## Demographics

- **Age:** expected in history of present illness (HPI) and similar narrative sections when the dialogue states it. Elsewhere, leave it out.
- **Sex:** only when the dialogue states it explicitly ("the patient is female", "I'm a woman"). Never infer it from "sir"/"ma'am", names or pronouns.
- **Race/ethnicity:** only when stated explicitly about the patient. Never infer it from names, language, nationality or location.
- **Pronouns:** don't add pronoun facts. They're checked on the summary side, not as omissions.

## Fields

- **fact_type:** the closest of age, sex, gender, pronoun, race_ethnicity, medication, diagnosis, symptom, quantity (doses, measurements), date (durations, timing), other.
- **subject:** `patient` unless the fact is about a relative (`relative`) or someone else (`other`).
- **Source span:** select the **shortest exact passage** in the dialogue that establishes the fact, then click *Use selection as source span*. For a question-and-answer ("Any allergies? / No."), select from the question through the answer.
- **Rationale:** one line on why it belongs in this section. For an excluded fact, say why it's excluded.

## Calibration

- **How many:** usually 2–10 facts per dialogue; short sections have fewer. If nothing relevant is established, add no facts and freeze the plan. An empty plan is a valid answer.
- **How long:** aim for 5–10 minutes per dialogue. If you agonise, pick one and note the reason in the rationale.
- **Consistency:** keep a short notes file of any rule you settle while working, e.g. "vitals go in exam, not HPI". If a rule changes after the first ~5 dialogues, revisit those dialogues before freezing. Record the change; it becomes the pilot note in the report.

## Blinding

Don't open any summary, and don't use what you remember of outputs from the audit. Tick *Plan complete and frozen* only for a dialogue you've finished. Then click **Export JSONL**, save it as `plans.jsonl` in `results/factuality/local/v3/plans/`, and run `python -m factuality.v3_cli freeze-plans`. Freezing checks every span and refuses incomplete plans.
