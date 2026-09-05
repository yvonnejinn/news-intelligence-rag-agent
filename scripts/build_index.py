import argparse
import hashlib
from pathlib import Path
import pandas as pd
from news_agent.preprocessing import prepare_documents, chunk_documents
from news_agent.retrieval import HashEncoder, SentenceEncoder, VectorIndex, DEFAULT_MODEL


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--revision")
    p.add_argument("--chunk-size", type=int, default=160)
    p.add_argument("--overlap", type=int, default=30)
    p.add_argument("--demo", action="store_true", help="Lexical hash smoke test, not semantic evaluation")
    args = p.parse_args()
    frame = pd.read_csv(args.data).fillna("")
    if "text" not in frame:
        raise ValueError("Input CSV requires a text column")
    chunks = chunk_documents(prepare_documents(frame.to_dict("records")), args.chunk_size, args.overlap)
    if not chunks:
        raise ValueError("No usable text")
    encoder = HashEncoder() if args.demo else SentenceEncoder(args.model, args.revision)
    metadata = {"encoder": encoder.model_name, "revision": encoder.revision,
                "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "chunk_size": args.chunk_size, "overlap": args.overlap, "demo": args.demo}
    index = VectorIndex(encoder.encode([c["text"] for c in chunks]), chunks, metadata)
    index.save(args.output)
    print(f"Indexed {len(chunks)} chunks with {index.engine}; demo={args.demo}")


if __name__ == "__main__":
    main()
