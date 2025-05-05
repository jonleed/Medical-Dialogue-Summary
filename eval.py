from flask import Flask, render_template, request, jsonify

from transformers import BartTokenizer, BartTokenizer, BartForConditionalGeneration, GenerationConfig
# import kagglehub
import pandas as pd
import numpy as np
import torch

app = Flask(__name__)  

# Load data
from datasets import Dataset
Test_data = pd.read_csv('./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv')
test_dataset = Dataset.from_pandas(Test_data)

from huggingface_hub import login
login()

PATH_TO_MODEL = "mdlam/clinical-note-model"

model = BartForConditionalGeneration.from_pretrained(PATH_TO_MODEL)
tokenizer = BartTokenizer.from_pretrained(PATH_TO_MODEL)

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

# Define your generation config once
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

# Function to generate notes from dialogue
def generate_note(dialogue, section_header):
    # Inject the target header as a prompt prefix
    prompt = f"Summarize the following doctor-patient dialogue into a detailed {section_header} clinical note: {dialogue}"

    inputs = tokenizer(
        prompt,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        generation_config=generation_config
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html', test_data=Test_data, section_map=SECTION_NAME_MAP)

# API endpoint to generate summaries
@app.route('/generate', methods=['POST'])
def generate():
    dialogue = request.json['dialogue']
    section_header = request.json['section_header']
    
    # Handle missing fields
    if not dialogue or not section_header:
        return jsonify({'error': 'Missing dialogue or section_header'}), 400
    
    section_full_name = SECTION_NAME_MAP.get(section_header.upper(), section_header)
    note = generate_note(dialogue, section_full_name)   
    return jsonify({'note': note})

if __name__ == '__main__':
    app.run(debug=True)