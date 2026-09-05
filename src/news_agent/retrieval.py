"""Normalized vectors, portable non-pickle persistence, optional FAISS."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or not np.isfinite(vectors).all() or vectors.shape[1] < 1:
        raise ValueError("Embeddings must be a finite 2D array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.ascontiguousarray(vectors / np.maximum(norms, 1e-12))


class HashEncoder:
    """Deterministic lexical smoke-test encoder; not a semantic model."""
    model_name = "demo-hash-v1"
    revision = None

    def __init__(self, dimensions=256):
        self.dimensions = dimensions

    def encode(self, texts):
        values = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.casefold()):
                digest = hashlib.sha256(token.encode()).digest()
                values[row, int.from_bytes(digest[:4], "big") % self.dimensions] += 1
        return normalize(values)


class SentenceEncoder:
    def __init__(self, model_name=DEFAULT_MODEL, revision=None):
        from sentence_transformers import SentenceTransformer
        self.model_name, self.revision = model_name, revision
        self.model = SentenceTransformer(model_name, revision=revision)

    def encode(self, texts):
        return self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True,
                                 show_progress_bar=False)


def encoder_from_metadata(metadata):
    if metadata["encoder"] == "demo-hash-v1":
        return HashEncoder(metadata["dimensions"])
    return SentenceEncoder(metadata["encoder"], metadata.get("revision"))


class VectorIndex:
    def __init__(self, vectors, chunks, metadata=None, use_faiss=True):
        self.vectors = normalize(vectors)
        if len(chunks) != len(self.vectors) or not len(chunks):
            raise ValueError("A nonempty aligned chunk/vector collection is required")
        ids = [c["chunk_id"] for c in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate chunk IDs")
        if (np.linalg.norm(self.vectors, axis=1) == 0).any():
            raise ValueError("Empty document embeddings cannot be indexed")
        self.chunks, self.metadata = chunks, dict(metadata or {})
        self.metadata["dimensions"] = self.vectors.shape[1]
        self.engine, self._index = "numpy", None
        if use_faiss:
            try:
                import faiss
            except ImportError:
                pass
            else:
                self._index = faiss.IndexFlatIP(self.vectors.shape[1])
                self._index.add(self.vectors)
                self.engine = "faiss"

    def search(self, query_vector, k=5, min_score=None):
        if k < 1:
            raise ValueError("k must be positive")
        query = normalize(np.asarray(query_vector).reshape(1, -1))
        if query.shape[1] != self.vectors.shape[1]:
            raise ValueError("Query and index dimensions differ")
        if not query.any():
            return []
        k = min(k, len(self.chunks))
        if self._index is not None:
            scores, indices = self._index.search(query, k)
            indices, scores = indices[0], scores[0]
        else:
            all_scores = self.vectors @ query[0]
            indices = np.argsort(-all_scores, kind="stable")[:k]
            scores = all_scores[indices]
        return [{**self.chunks[int(i)], "score": float(score)} for i, score in zip(indices, scores)
                if min_score is None or score >= min_score]

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if any(path.iterdir()):
            raise ValueError("Index output directory must be empty")
        np.save(path / "vectors.npy", self.vectors, allow_pickle=False)
        payload = {"format_version": 1, "metadata": self.metadata, "chunks": self.chunks}
        (path / "index.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory, use_faiss=True):
        path = Path(directory)
        payload = json.loads((path / "index.json").read_text(encoding="utf-8"))
        if payload["format_version"] != 1:
            raise ValueError("Unsupported index format")
        return cls(np.load(path / "vectors.npy", allow_pickle=False), payload["chunks"],
                   payload["metadata"], use_faiss=use_faiss)
