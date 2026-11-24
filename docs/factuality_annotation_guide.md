# Writing expected-fact plans

A plan lists the facts a competent clinician would put in **the requested section** of a note written from this dialogue. Plans are drafted from the dialogue alone, before any summary is shown, and reviewed by the human reviewer. Later, each summary is checked against its plan: a planned fact the summary leaves out is an **omission**. So the plan defines what "omitted" means. List what a good section *should* contain, not everything the dialogue says.

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
- **Sex:** expect it in a plan only when the dialogue states it explicitly ("the patient is female", "I'm a woman"). Address terms, names and pronouns don't make a summary *miss* a fact, because a note can say "the patient". (When you later label what a summary says, they are evidence: see the gender-reference rule below.)
- **Race/ethnicity:** only when stated explicitly about the patient. Never infer it from names, language, nationality or location.
- **Pronouns:** don't add pronoun facts. They're checked on the summary side, not as omissions.
- **Gender reference (when you label a summary's claims):** sex terms ("male", "female") and pronouns ("he", "she", "his", "her") are one dimension. The summarizer sees only the dialogue, so these are checked against everything the dialogue gives about the patient:
  - explicit statements ("I'm a woman", "the patient is male");
  - a third-person pronoun that refers to the patient ("give him Norflex"; check who it refers to: "he was only fifty two" about the patient's father says nothing about the patient);
  - gendered address directed at the patient ("sir", "ma'am", "miss", "mister", "Mr./Mrs./Ms.", "young man").

  A claim that matches the evidence is `supported`, one that conflicts with it is `contradicted`, and one that asserts a gender when the dialogue gives no evidence at all is `unsupported`. If the cues disagree with each other ("sir" and "she"), if the only cue is about someone else, or if the only basis is an inference such as a pregnancy test, label it `unclear`. Names and casual address ("bud") are not evidence. Label such claims `sex` or `pronoun`; they are scored together. Gender identity (`gender`) is separate and needs an explicit statement.

## Fields

- **fact_type:** the closest of age, sex, gender, pronoun, race_ethnicity, medication, diagnosis, symptom, quantity (doses, measurements), date (durations, timing), other.
- **subject:** `patient` unless the fact is about a relative (`relative`) or someone else (`other`).
- **Source span:** select the **shortest exact passage** in the dialogue that establishes the fact, then click *Use selection as source span*. For a question-and-answer ("Any allergies? / No."), select from the question through the answer.
- **Rationale:** one line on why it belongs in this section. For an excluded fact, say why it's excluded.

## Comparing a plan with a summary

Each planned fact gets one label against each summary: `present`, `omitted`, `unclear` or `excluded`. Use `unclear` when a careful reader could say either, for example the dialogue answers a list of questions and the summary covers the list only in general terms ("cold-like symptoms" for fever, cough and sore throat). An `unclear` fact is left out of both omission counts, like a claim the scorer cannot assess. Keep it consistent with the claim labels: if a claim about the same content is `unclear`, the matching fact is not `present`.

A source span must establish the fact on its own. "you need it." does not establish "substance abuse"; the clinician's "try to stop your substance abuse" does.

## Calibration

- **How many:** usually 2–10 facts per dialogue; short sections have fewer. If nothing relevant is established, add no facts and freeze the plan. An empty plan is a valid answer.
- **How long:** aim for 5–10 minutes per dialogue. If you agonise, pick one and note the reason in the rationale.
- **Consistency:** keep a short notes file of any rule you settle while working, e.g. "vitals go in exam, not HPI". If a rule changes after the first ~5 dialogues, revisit those dialogues before freezing. Record the change, so the report can describe how the rules settled.

## Blinding

Don't open any summary, and don't rely on outputs you may have seen earlier. Tick *Plan complete and frozen* only for a dialogue you've finished. Then click **Export JSONL**, save it as `plans.jsonl` in the plans folder that `python -m factuality.cli plans` printed, and run `python -m factuality.cli freeze-plans`. Freezing checks every span and refuses incomplete plans.
