import os

import pandas as pd
import torch
from datasets import Dataset
from flask import Flask, jsonify, render_template, request
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "facebook/bart-base")
ADAPTER_MODEL_ID = os.environ.get("ADAPTER_MODEL_ID", "jonleed/clinical-note-bart-lora")
ORIGINAL_ADAPTER_ID = "mdlam/clinical-note-model"
HF_TOKEN = os.environ.get("HF_TOKEN")

DATASET_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"

app = Flask(__name__)

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if model is not None:
        return
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)
    try:
        peft_model = PeftModel.from_pretrained(base, ADAPTER_MODEL_ID, token=HF_TOKEN)
    except Exception:
        app.logger.warning(
            "Could not load adapter %s, falling back to original %s",
            ADAPTER_MODEL_ID,
            ORIGINAL_ADAPTER_ID,
        )
        peft_model = PeftModel.from_pretrained(base, ORIGINAL_ADAPTER_ID, token=HF_TOKEN)
    model = peft_model.merge_and_unload()
    model.eval()


Test_data = pd.read_csv(DATASET_CSV)
test_dataset = Dataset.from_pandas(Test_data)

SECTION_NAME_MAP = {
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
    'DISPOSITION': 'Disposition'
}

generation_config = GenerationConfig(
    temperature=0.9,
    top_k=50,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=2.0,
    no_repeat_ngram_size=4,
    num_beams=4,
    max_length=256
)


def generate_note(dialogue, section_header):
    load_model()
    prompt = f"Summarize the following doctor-patient dialogue into a detailed {section_header} clinical note: {dialogue}"

    inputs = tokenizer(
        prompt,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            generation_config=generation_config
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


@app.route('/')
def index():
    return render_template('index.html', test_data=Test_data, section_map=SECTION_NAME_MAP)


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json(silent=True) or {}
    dialogue = data.get('dialogue')
    section_header = data.get('section_header')

    if not dialogue or not section_header:
        return jsonify({'error': 'Missing dialogue or section_header'}), 400

    try:
        section_full_name = SECTION_NAME_MAP.get(section_header.upper(), section_header)
        note = generate_note(dialogue, section_full_name)
    except Exception as exc:
        app.logger.exception("Generation failed")
        return jsonify({'error': str(exc)}), 500

    return jsonify({'note': note})


if __name__ == '__main__':
    app.run(debug=False)
