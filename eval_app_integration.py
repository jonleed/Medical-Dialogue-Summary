import argparse
import json

import pandas as pd
from tqdm import tqdm

VAL_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv"
TEST_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"

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

DEFAULT_PRIMARY_SECTION = "History of Present Illness"


def run_section_mode(args):
    import eval as app_module
    from rouge_score import rouge_scorer

    df = pd.read_csv(TEST_CSV).sample(n=args.n, random_state=args.seed)

    client = app_module.app.test_client()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    predictions = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Serving + scoring"):
        resp = client.post(
            "/generate",
            json={"dialogue": str(row["dialogue"]), "section_header": str(row["section_header"])},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"/generate failed ({resp.status_code}): {resp.get_json(silent=True)}")
        pred = resp.get_json()["note"]
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

    return app_module.APP_NAME, len(df), totals, predictions


def run_note_mode(args):
    import evaluate as hf_evaluate
    import app as note_app_module
    from eval_notebook_protocol import SPLIT_PATHS, preprocess_data
    from transformers import AutoTokenizer

    raw = pd.read_csv(SPLIT_PATHS[args.split])
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
    data = preprocess_data(raw)
    if args.n:
        data = data.sample(n=min(args.n, len(data)), random_state=args.seed)

    client = note_app_module.app.test_client()
    rouge = hf_evaluate.load("rouge")
    preds, refs, predictions = [], [], []

    for _, row in tqdm(data.iterrows(), total=len(data), desc="Serving + scoring"):
        dialogue = row["input_text"].split("clinical note: ", 1)[1]
        primary = SECTION_STANDARD_NAMES.get(
            str(row["section_header"]).upper(), DEFAULT_PRIMARY_SECTION
        )
        resp = client.post(
            "/generate_note",
            json={"dialogue": dialogue, "primary": primary},
        )
        if resp.status_code != 200:
            raise RuntimeError(f"/generate_note failed ({resp.status_code}): {resp.get_json(silent=True)}")
        pred = resp.get_json()["note"]
        target_ids = tokenizer(row["target_text"], max_length=256, truncation=True)["input_ids"]
        ref = tokenizer.decode(target_ids, skip_special_tokens=True)
        preds.append(pred)
        refs.append(ref)
        predictions.append({"reference": ref, "prediction": pred})

    metrics = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    totals = {k: float(v) for k, v in metrics.items()}
    return note_app_module.APP_NAME, len(preds), totals, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Score a Flask demo end-to-end through its test client"
    )
    parser.add_argument("--app", choices=["section", "note"], default="section",
                        help="section = eval.py per-section demo on test CSV; note = app.py whole-transcript demo under the original notebook protocol (validation split)")
    parser.add_argument("--split", choices=["val", "train"], default="val",
                        help="(note mode only) dataset split to sample dialogues from")
    parser.add_argument("--n", type=int, default=100, help="number of examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="write served predictions to this JSON file")
    parser.add_argument("--compare", type=str, default=None, help="metrics JSON to diff against")
    args = parser.parse_args()

    if args.app == "note":
        app_name, count, totals, predictions = run_note_mode(args)
        label = f"note app ({app_name}) | split={args.split} | dialogues={count} | seed={args.seed}"
    else:
        app_name, count, totals, predictions = run_section_mode(args)
        label = f"section app ({app_name}) | test csv | n={count} | seed={args.seed}"

    print(f"\nApp integration eval over {label} (decode=beam4_deterministic):")
    for k, v in totals.items():
        print(f"  {k}: {v:.4f}")

    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)["metrics"]
        print(f"\nDelta vs {args.compare}:")
        for k in totals:
            if k in baseline:
                print(f"  {k}: {totals[k] - baseline[k]:+.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "app": args.app,
                "n": count,
                "seed": args.seed,
                "decode": "beam4_deterministic",
                "metrics": totals,
                "predictions": predictions,
            }, f, indent=2)
        print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
