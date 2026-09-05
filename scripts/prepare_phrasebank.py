import argparse
from pathlib import Path
import pandas as pd
from news_agent.preprocessing import clean_text, content_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="sentences_allagree")
    p.add_argument("--parquet")
    p.add_argument("--revision", help="Pin a dataset revision for repeatable online loading")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.parquet:
        frame = pd.read_parquet(args.parquet)
    else:
        from datasets import load_dataset
        try:
            frame = load_dataset("takala/financial_phrasebank", args.config,
                                 revision=args.revision, split="train").to_pandas()
        except Exception as error:
            raise RuntimeError("Dataset loading failed. Some releases do not support legacy dataset "
                               "scripts. Use a documented local Parquet export with --parquet.") from error
    frame = frame.rename(columns={"sentence": "text"})
    if not {"text", "label"}.issubset(frame):
        raise ValueError("Dataset needs sentence/text and label columns")
    frame = frame[["text", "label"]].copy()
    frame["text"] = frame["text"].map(clean_text)
    mapping = {0: "negative", 1: "neutral", 2: "positive", "0": "negative", "1": "neutral", "2": "positive"}
    frame["label"] = frame["label"].map(lambda value: mapping.get(value, value))
    if not set(frame["label"]).issubset({"negative", "neutral", "positive"}):
        raise ValueError("Unknown label; verify dataset label mapping")
    frame = frame[frame["text"].ne("")].copy()
    frame["hash"] = frame["text"].map(content_hash)
    if frame.groupby("hash")["label"].nunique().gt(1).any():
        raise ValueError("Conflicting labels for duplicate text require review")
    frame = frame.drop_duplicates("hash").drop(columns="hash")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Prepared {len(frame)} labeled sentences")


if __name__ == "__main__":
    main()
