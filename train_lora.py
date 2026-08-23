import argparse

import numpy as np
import pandas as pd
import torch
import evaluate
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy

TRAIN_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-TrainingSet.csv"
VAL_CSV = "./MTS-Dialog/Main-Dataset/MTS-Dialog-ValidationSet.csv"

SECTION_ORDER = [
    ("chief_complaint", "cc"),
    ("history_of_present_illness", "genhx"),
    ("past_medical_history", "pastmedicalhx"),
    ("past_surgeries", "pastsurgical"),
    ("medications", "medications"),
    ("allergies", "allergy"),
    ("social_history", "fam/sochx"),
    ("educational_courses", "edcourse"),
    ("review_of_systems", "ros"),
    ("physical_exam", "exam"),
    ("assessment", "assessment"),
    ("exam", "exam"),
    ("procedures", "procedures"),
    ("labs", "labs"),
    ("plan", "plan"),
    ("disposition", "disposition"),
]


def preprocess_data(df):
    structured = []
    grouped = df.groupby(df["ID"].astype(str))
    for dialogue_id, group in grouped:
        full_dialogue = " ".join(group["dialogue"].tolist())
        section_header = group["section_header"].tolist()
        section_header_name = ""
        for standard_name, header_name in SECTION_ORDER:
            if section_header[0].lower() == header_name:
                section_header_name = standard_name
        full_name = section_header_name.replace("_", " ")
        combined = [f"{full_name.strip()}: {str(t).strip()}" for t in group["section_text"]]
        structured.append({
            "input_text": f"Summarize the following doctor-patient dialogue into a detailed {full_name} clinical note: {full_dialogue}",
            "target_text": "\n".join(combined),
        })
    return pd.DataFrame(structured)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune mirroring the original notebook recipe")
    parser.add_argument("--base_model", default="GanjinZero/biobart-large")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--per_device_batch", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    short = args.base_model.split("/")[-1]
    output_dir = args.output_dir or f"./{short}-lora"
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=64,
        lora_dropout=0.01,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize(batch):
        inputs = tokenizer(
            batch["input_text"],
            max_length=args.max_input_length,
            padding="max_length",
            truncation=True,
        )
        targets = tokenizer(
            batch["target_text"],
            max_length=args.max_target_length,
            truncation=True,
        )
        labels = [
            [tid if tid != tokenizer.pad_token_id else -100 for tid in seq]
            for seq in targets["input_ids"]
        ]
        inputs["labels"] = labels
        return inputs

    train_data = preprocess_data(pd.read_csv(TRAIN_CSV))
    val_data = preprocess_data(pd.read_csv(VAL_CSV))
    train_dataset = Dataset.from_pandas(train_data, preserve_index=False)
    val_dataset = Dataset.from_pandas(val_data, preserve_index=False)
    print(f"train={len(train_dataset)} val={len(val_dataset)} device={device}")

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        vocab_size = len(tokenizer)
        predictions = np.where(
            (predictions >= 0) & (predictions < vocab_size),
            predictions,
            tokenizer.unk_token_id,
        )
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=f"{output_dir}/checkpoints",
        eval_strategy=IntervalStrategy.EPOCH,
        save_strategy=IntervalStrategy.EPOCH,
        learning_rate=args.lr,
        per_device_train_batch_size=args.per_device_batch,
        per_device_eval_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=128,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        logging_strategy="epoch",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names),
        eval_dataset=val_dataset.map(tokenize, batched=True, remove_columns=val_dataset.column_names),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding="max_length"),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
