import argparse

import evaluate
import nltk
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from compute_metrics import ADAPTER_MODEL_ID, BASE_MODEL_ID

SECTION_ORDER = [
    ('chief_complaint', 'cc'),
    ('history_of_present_illness', 'genhx'),
    ('past_medical_history', 'pastmedicalhx'),
    ('past_surgeries', 'pastsurgical'),
    ('medications', 'medications'),
    ('allergies', 'allergy'),
    ('social_history', 'fam/sochx'),
    ('educational_courses', 'edcourse'),
    ('review_of_systems', 'ros'),
    ('physical_exam', 'exam'),
    ('assessment', 'assessment'),
    ('exam', 'exam'),
    ('procedures', 'procedures'),
    ('labs', 'labs'),
    ('plan', 'plan'),
    ('disposition', 'disposition')
]

SPLIT_PATHS = {
    "val": "./MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv",
    "test": "./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
    "train": "./MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv",
}


def preprocess_data(df):
    df['dialogue_id'] = df['ID'].astype(str)
    grouped = df.groupby('dialogue_id')

    structured_data = []

    for dialogue_id, group in grouped:
        full_dialogue = ' '.join(group['dialogue'].tolist())

        section_header = group['section_header'].tolist()

        section_header_name = ''
        for standard_name, header_name in SECTION_ORDER:
            if section_header[0].lower() == header_name:
                section_header_name = standard_name

        combined_sections = []
        for text, header in zip(group['section_text'], group['section_header']):
            full_name = section_header_name.replace('_', ' ')
            combined_sections.append(f"{full_name.strip()}: {text.strip()}")

        full_note = "\n".join(combined_sections)

        structured_data.append({
            'input_text': f"Summarize the following doctor-patient dialogue into a detailed {full_name} clinical note: {full_dialogue}",
            'target_text': full_note,
            'section_header': section_header[0],
            'dialogue_id': dialogue_id
        })
    return pd.DataFrame(structured_data)


def main():
    parser = argparse.ArgumentParser(description="Evaluate under the original notebook protocol")
    parser.add_argument("--split", choices=list(SPLIT_PATHS.keys()), default="val")
    parser.add_argument("--n", type=int, default=None, help="limit number of dialogues")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--adapter_model",
        default=ADAPTER_MODEL_ID,
        help="LoRA adapter path or Hub ID; use 'none' for the base model only",
    )
    args = parser.parse_args()

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
    if args.adapter_model.lower() in {"none", "base"}:
        model = base
    else:
        model = PeftModel.from_pretrained(base, args.adapter_model).merge_and_unload()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    raw = pd.read_csv(SPLIT_PATHS[args.split])
    data = preprocess_data(raw)
    if args.n:
        data = data.sample(n=min(args.n, len(data)), random_state=args.seed)
    print(f"{args.split}: {len(data)} dialogues")

    rouge = evaluate.load("rouge")

    generation_config = GenerationConfig(
        temperature=0.9,
        top_k=50,
        top_p=0.95,
        do_sample=False,
        repetition_penalty=2.0,
        no_repeat_ngram_size=4,
        num_beams=4,
        max_length=128
    )

    preds, refs = [], []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Generating"):
        inputs = tokenizer(
            row["input_text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=generation_config,
            )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        target_ids = tokenizer(
            row["target_text"],
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        target_ids[target_ids == tokenizer.pad_token_id] = -100
        target_ids[target_ids == -100] = tokenizer.pad_token_id
        ref = tokenizer.decode(target_ids[0], skip_special_tokens=True)

        preds.append(pred)
        refs.append(ref)

    result = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    print(f"\nOriginal-notebook protocol | split={args.split} | dialogues={len(preds)}")
    print(f"  rouge1:    {result['rouge1']:.4f}")
    print(f"  rouge2:    {result['rouge2']:.4f}")
    print(f"  rougeL:    {result['rougeL']:.4f}")
    print(f"  rougeLsum: {result['rougeLsum']:.4f}")


if __name__ == "__main__":
    main()
