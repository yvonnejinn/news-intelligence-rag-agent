import argparse
from pathlib import Path
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Label news using a saved sentiment checkpoint")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    from transformers import pipeline
    frame = pd.read_csv(args.data).fillna("")
    if "text" not in frame or frame.empty:
        raise ValueError("Nonempty text column required")
    classifier = pipeline("text-classification", model=args.model, tokenizer=args.model, device=-1)
    results = classifier(frame["text"].tolist(), truncation=True, max_length=256, batch_size=16)
    labels = [r["label"].lower() for r in results]
    if not set(labels).issubset({"negative", "neutral", "positive"}):
        raise ValueError("Checkpoint must define semantic sentiment labels")
    frame["predicted_label"] = labels
    frame["predicted_confidence"] = [r["score"] for r in results]
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(frame["predicted_label"].value_counts().to_string())


if __name__ == "__main__":
    main()
