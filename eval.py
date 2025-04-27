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


# Define your generation config once
generation_config = GenerationConfig(
    temperature=0.7,
    top_k=60,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=2.4,
    no_repeat_ngram_size=2,
    num_beams=1,
    max_length=128  # You can adjust this
)

from transformers import GenerationConfig

# Define your generation config once
generation_config = GenerationConfig(
    temperature=0.7,
    top_k=60,
    top_p=0.95,
    do_sample=True,
    repetition_penalty=2.4,
    no_repeat_ngram_size=2,
    num_beams=1,
    max_length=128  
)

# Function to generate notes from dialogue
def generate_note(dialogue):
    inputs = tokenizer(
        dialogue,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        generation_config=generation_config  # ✅ This is where it goes
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html', test_data=Test_data)

# API endpoint to generate summaries
@app.route('/generate', methods=['POST'])
def generate():
    dialogue = request.json['dialogue']
    note = generate_note(dialogue)
    return jsonify({'note': note})

if __name__ == '__main__':
    app.run(debug=True)