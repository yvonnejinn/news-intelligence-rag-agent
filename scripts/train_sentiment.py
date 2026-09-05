"""Stratified training/validation entry point; validation is not a final test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from news_agent.preprocessing import clean_text, content_hash
from news_agent.evaluation import classification_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--model", default="ProsusAI/finbert")
    p.add_argument("--revision")
    p.add_argument("--output", required=True)
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, Trainer,
                              TrainingArguments, DataCollatorWithPadding, set_seed)
    frame = pd.read_csv(args.data)
    if not {"text", "label"}.issubset(frame) or frame[["text", "label"]].isna().any().any():
        raise ValueError("Nonmissing text,label CSV required")
    frame["text"] = frame["text"].map(clean_text)
    frame["content_sha256"] = frame["text"].map(content_hash)
    if frame["content_sha256"].duplicated().any() or frame["text"].eq("").any():
        raise ValueError("Remove duplicates and empty text before splitting")
    labels = {"negative": 0, "neutral": 1, "positive": 2}
    if set(frame["label"]) != set(labels) or frame["label"].value_counts().min() < 2:
        raise ValueError("All three classes with enough examples are required")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, revision=args.revision, num_labels=3)
    # Preserve a checkpoint's semantic label order (FinBERT's order differs from PhraseBank).
    existing = {str(label).lower(): int(i) for i, label in model.config.id2label.items()}
    mapping = existing if set(existing) == set(labels) else labels
    model.config.label2id, model.config.id2label = mapping, {i: name for name, i in mapping.items()}
    train, valid = train_test_split(np.arange(len(frame)), test_size=0.2,
                                    stratify=frame["label"], random_state=args.seed)
    frame["split"] = "train"
    frame.loc[valid, "split"] = "validation"
    frame[["content_sha256", "label", "split"]].to_csv(output / "split_manifest.csv", index=False)

    def prepare(indices):
        data = Dataset.from_dict({"text": frame.iloc[indices]["text"].tolist(),
                                  "labels": frame.iloc[indices]["label"].map(mapping).tolist()})
        return data.map(lambda batch: tokenizer(batch["text"], truncation=True, max_length=256),
                        batched=True, remove_columns=["text"])

    def compute(result):
        predictions = result.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        names = model.config.id2label
        return classification_metrics([names[int(i)] for i in result.label_ids],
                                      [names[int(i)] for i in predictions.argmax(axis=1)])

    training = TrainingArguments(output_dir=str(output / "checkpoints"), num_train_epochs=args.epochs,
        learning_rate=2e-5, per_device_train_batch_size=8, per_device_eval_batch_size=16,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, save_total_limit=1,
        seed=args.seed, data_seed=args.seed, report_to="none")
    trainer = Trainer(model=model, args=training, train_dataset=prepare(train), eval_dataset=prepare(valid),
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=compute)
    trainer.train()
    report = {"validation_metrics": trainer.evaluate(), "model": args.model,
              "requested_revision": args.revision, "resolved_revision": getattr(model.config, "_commit_hash", None),
              "label_mapping": mapping, "seed": args.seed,
              "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
              "scope": "Validation used for checkpoint selection; not an untouched final test. "
              "A sentiment checkpoint may already have seen PhraseBank: audit pretraining/fine-tuning provenance."}
    trainer.save_model(str(output / "model"))
    tokenizer.save_pretrained(output / "model")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
