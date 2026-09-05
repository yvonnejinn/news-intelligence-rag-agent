import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from news_agent.retrieval import VectorIndex, encoder_from_metadata
from news_agent.evaluation import recall_at_k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--output", default="artifacts/retrieval_metrics.json")
    args = p.parse_args()
    index = VectorIndex.load(args.index)
    encoder = encoder_from_metadata(index.metadata)
    frame = pd.read_csv(args.eval).fillna("")
    if not {"question", "relevant_chunk_ids"}.issubset(frame) or frame.empty:
        raise ValueError("Nonempty question,relevant_chunk_ids CSV required")
    known = {c["chunk_id"] for c in index.chunks}
    rows = []
    for item in frame.to_dict("records"):
        relevant = {v.strip() for v in item["relevant_chunk_ids"].split("|") if v.strip()}
        if not relevant or not relevant.issubset(known) or not item["question"].strip():
            raise ValueError("Every query needs a question and relevant IDs present in this index")
        start = time.perf_counter()
        retrieved = index.search(encoder.encode([item["question"]])[0], k=5)
        ids = [v["chunk_id"] for v in retrieved]
        rows.append({"question": item["question"], "retrieved_ids": ids,
                     "relevant_ids": sorted(relevant), "latency_seconds": time.perf_counter() - start,
                     **{f"recall_at_{k}": recall_at_k(ids, relevant, k) for k in (1, 3, 5)}})
    report = {"query_count": len(rows), "index_metadata": index.metadata, "engine": index.engine,
              "evaluation_sha256": hashlib.sha256(Path(args.eval).read_bytes()).hexdigest(),
              "metrics": {f"recall_at_{k}": float(np.mean([r[f"recall_at_{k}"] for r in rows])) for k in (1, 3, 5)},
              "queries": rows, "scope": "Demo fixture only" if index.metadata.get("demo") else
              "Supplied evaluation set; human review and independence must be documented separately"}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"]))


if __name__ == "__main__":
    main()
