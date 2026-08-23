import argparse

import nltk
import pandas as pd
import torch
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
from peft import PeftModel

from compute_metrics import ADAPTER_MODEL_ID, BASE_MODEL_ID, DATASET_CSV, SECTION_NAME_MAP

DECODE_CONFIGS = {
    "original_sampling": GenerationConfig(
        temperature=0.9, top_k=50, top_p=0.95, do_sample=True,
        repetition_penalty=2.0, no_repeat_ngram_size=4, num_beams=4, max_length=256),
    "beam4_deterministic": GenerationConfig(
        do_sample=False, repetition_penalty=2.0,
        no_repeat_ngram_size=4, num_beams=4, max_length=256),
    "greedy": GenerationConfig(
        do_sample=False, repetition_penalty=2.0,
        no_repeat_ngram_size=4, num_beams=1, max_length=256),
}


def sent_split(text):
    return "\n".join(nltk.sent_tokenize(text))


def main():
    parser = argparse.ArgumentParser(description="Sweep decode configs x scoring methods")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
        help="which MTS-Dialog csv to evaluate",
    )
    args = parser.parse_args()

    split_paths = {
        "test": "./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv",
        "val": "./MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv",
        "train": "./MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv",
    }
    csv_path = split_paths[args.split]

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
    model = PeftModel.from_pretrained(base, ADAPTER_MODEL_ID).merge_and_unload()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"model on {device}, n={args.n}, seed={args.seed}")

    df = pd.read_csv(csv_path).sample(n=args.n, random_state=args.seed)

    plain = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    stem = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    lsum = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)

    for name, gc in DECODE_CONFIGS.items():
        torch.manual_seed(args.seed)
        totals = {"plain_rougeL": 0.0, "stem_rougeL": 0.0, "hf_rougeLsum": 0.0}
        for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
            section_full = SECTION_NAME_MAP.get(str(row["section_header"]).upper(), str(row["section_header"]))
            prompt = f"Summarize the following doctor-patient dialogue into a detailed {section_full} clinical note: {row['dialogue']}"
            inputs = tokenizer(prompt, max_length=128, padding="max_length", truncation=True, return_tensors="pt").to(device)
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                generation_config=gc,
            )
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ref = str(row["section_text"])
            totals["plain_rougeL"] += plain.score(ref, pred)["rougeL"].fmeasure
            totals["stem_rougeL"] += stem.score(ref, pred)["rougeL"].fmeasure
            totals["hf_rougeLsum"] += lsum.score(sent_split(ref), sent_split(pred))["rougeLsum"].fmeasure
        print(f"\n== {name} ==")
        for k, v in totals.items():
            print(f"  {k}: {v / len(df):.4f}")


if __name__ == "__main__":
    main()
