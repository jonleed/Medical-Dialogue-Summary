import argparse
import os

import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "facebook/bart-base")
ADAPTER_MODEL_ID = os.environ.get("ADAPTER_MODEL_ID", "jonleed/clinical-note-bart-lora")
DATASET_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"

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


def main():
    parser = argparse.ArgumentParser(description="Compute ROUGE for the fine-tuned model")
    parser.add_argument("--n", type=int, default=100, help="number of test examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="write predictions to this JSON file")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL_ID).merge_and_unload()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"Loaded {ADAPTER_MODEL_ID} onto {device}")

    df = pd.read_csv(DATASET_CSV).sample(n=args.n, random_state=args.seed)

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Summarizing"):
        section_full = SECTION_NAME_MAP.get(str(row["section_header"]).upper(), str(row["section_header"]))
        prompt = f"Summarize the following doctor-patient dialogue into a detailed {section_full} clinical note: {row['dialogue']}"
        inputs = tokenizer(prompt, max_length=INPUT_MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt").to(device)
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            generation_config=generation_config,
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference = str(row["section_text"])
        scores = scorer.score(reference, "\n".join(pred.split(". ")) if "\n" not in pred else pred)
        for k in totals:
            totals[k] += scores[k].fmeasure
        predictions.append({
            "section_header": row["section_header"],
            "reference": reference,
            "prediction": pred,
            "rougeL": scores["rougeL"].fmeasure,
        })

    print(f"\nResults over {len(df)} sampled test examples (seed={args.seed}):")
    for k, v in totals.items():
        print(f"  {k}: {v / len(df):.4f}")

    if args.out:
        import json

        with open(args.out, "w") as f:
            json.dump({"n": len(df), "seed": args.seed, "metrics": {k: v / len(df) for k, v in totals.items()},
                       "predictions": predictions}, f, indent=2)
        print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
