import argparse
import json

import pandas as pd
from rouge_score import rouge_scorer
from tqdm import tqdm

import eval as app_module

DATASET_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-TestSet-1-MEDIQA-Chat-2023.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Score the Flask app end-to-end with the same ROUGE protocol as compute_metrics.py"
    )
    parser.add_argument("--n", type=int, default=100, help="number of test examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="write served predictions to this JSON file")
    parser.add_argument("--compare", type=str, default=None, help="metrics JSON from compute_metrics.py to diff against")
    args = parser.parse_args()

    df = pd.read_csv(DATASET_CSV).sample(n=args.n, random_state=args.seed)

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

    metrics = {k: v / len(df) for k, v in totals.items()}
    print(f"\nApp integration eval over {len(df)} served examples (seed={args.seed}, decode=beam4_deterministic):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    if args.compare:
        with open(args.compare) as f:
            baseline = json.load(f)["metrics"]
        print(f"\nDelta vs {args.compare} (sampling decode):")
        for k in totals:
            print(f"  {k}: {metrics[k] - baseline[k]:+.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "n": len(df),
                "seed": args.seed,
                "decode": "beam4_deterministic",
                "metrics": metrics,
                "predictions": predictions,
            }, f, indent=2)
        print(f"Predictions written to {args.out}")


if __name__ == "__main__":
    main()
