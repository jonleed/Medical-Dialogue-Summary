import os

import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "GanjinZero/biobart-large")
ADAPTER_MODEL_ID = os.environ.get("ADAPTER_MODEL_ID", "./biobart-large-lora")
# Optional pins (commit hashes); unset loads the latest revision.
BASE_MODEL_REVISION = os.environ.get("BASE_MODEL_REVISION") or None
ADAPTER_REVISION = os.environ.get("ADAPTER_REVISION") or None
HF_TOKEN = os.environ.get("HF_TOKEN")

INPUT_MAX_LENGTH = 512
DATASET_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"
DEFAULT_PRIMARY_SECTION = "History of Present Illness"

SECTION_STANDARD_NAMES = {
    'CC': 'chief complaint',
    'GENHX': 'history of present illness',
    'PASTMEDICALHX': 'past medical history',
    'PASTSURGICAL': 'past surgeries',
    'MEDICATIONS': 'medications',
    'ALLERGY': 'allergies',
    'FAM/SOCHX': 'social history',
    'EDCOURSE': 'educational courses',
    'ROS': 'review of systems',
    'EXAM': 'physical exam',
    'ASSESSMENT': 'assessment',
    'PROCEDURES': 'procedures',
    'LABS': 'labs',
    'PLAN': 'plan',
    'DISPOSITION': 'disposition',
}

SECTION_LABELS = {
    'CC': 'Chief Complaint',
    'GENHX': 'History of Present Illness',
    'PASTMEDICALHX': 'Past Medical History',
    'PASTSURGICAL': 'Past Surgeries',
    'MEDICATIONS': 'Medications',
    'ALLERGY': 'Allergies',
    'FAM/SOCHX': 'Social History',
    'EDCOURSE': 'Educational Courses',
    'ROS': 'Review of Systems',
    'EXAM': 'Physical Exam',
    'ASSESSMENT': 'Assessment',
    'PROCEDURES': 'Procedures',
    'LABS': 'Labs',
    'PLAN': 'Plan',
    'DISPOSITION': 'Disposition',
}

app = Flask(__name__)
APP_NAME = "whole-transcript note generator (app.py)"

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if model is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN, revision=BASE_MODEL_REVISION)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN, revision=BASE_MODEL_REVISION)
    try:
        peft_model = PeftModel.from_pretrained(base, ADAPTER_MODEL_ID, token=HF_TOKEN, revision=ADAPTER_REVISION)
    except Exception:
        app.logger.exception(
            "Could not load adapter %s - refusing to fall back to a different model "
            "so served outputs keep matching the evaluation harness",
            ADAPTER_MODEL_ID,
        )
        raise
    model = peft_model.merge_and_unload()
    model.eval()


generation_config = GenerationConfig(
    do_sample=False,
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,
    num_beams=4,
    max_length=512,
)

BOILERPLATE_TEXTS = {
    "",
    "none",
    "noncontributory",
    "uncontributory",
    "negative",
    "unremarkable",
    "unremarkable except for etoh abuse",
    "not otherwise pertinent",
    "no known drug allergies",
    "no allergies",
    "reviewed and unchanged",
    "reviewed and remained unchanged",
}


def is_boilerplate(text):
    cleaned = text.strip().lower().rstrip(".").strip()
    return cleaned in BOILERPLATE_TEXTS


def build_samples():
    raw = pd.read_csv(DATASET_CSV)
    samples = []
    for dialogue_id, group in raw.groupby(raw["ID"].astype(str), sort=False):
        dialogue = " ".join(group["dialogue"].tolist())
        header_code = str(group["section_header"].iloc[0])
        sections = [
            {"label": SECTION_LABELS.get(str(h).upper(), str(h)), "text": str(t)}
            for h, t in zip(group["section_header"], group["section_text"])
        ]
        primary = SECTION_STANDARD_NAMES.get(header_code.upper(), DEFAULT_PRIMARY_SECTION)
        samples.append({
            "id": dialogue_id,
            "dialogue": dialogue,
            "primary": primary,
            "primary_label": SECTION_LABELS.get(header_code.upper(), header_code),
            "sections": sections,
        })
    return samples


SAMPLES = build_samples()


def generate_note(dialogue, primary):
    load_model()
    prompt = f"Summarize the following doctor-patient dialogue into a detailed {primary} clinical note: {dialogue}"
    n_tokens = len(tokenizer(prompt)["input_ids"])
    inputs = tokenizer(
        prompt,
        max_length=INPUT_MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
        )
    note = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return note, n_tokens


@app.route('/')
def index():
    return render_template('note.html', samples=SAMPLES)


@app.route('/generate_note', methods=['POST'])
def generate():
    data = request.get_json(silent=True) or {}
    dialogue = (data.get('dialogue') or '').strip()
    primary = (data.get('primary') or DEFAULT_PRIMARY_SECTION).strip()

    if not dialogue:
        return jsonify({'error': 'Missing dialogue'}), 400

    try:
        note, n_tokens = generate_note(dialogue, primary)
    except Exception as exc:
        app.logger.exception("Generation failed")
        return jsonify({'error': str(exc)}), 500

    body = note.strip()
    prefix = f"{primary.lower()}:"
    if body.lower().startswith(prefix):
        body = body[len(prefix):].strip()

    return jsonify({
        'note': note,
        'boilerplate': is_boilerplate(body),
        'tokens': n_tokens,
        'truncated': n_tokens > INPUT_MAX_LENGTH,
    })


if __name__ == '__main__':
    app.run(debug=False, port=int(os.environ.get("PORT", "5002")))
