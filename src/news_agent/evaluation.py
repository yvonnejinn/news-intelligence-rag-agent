import re
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(labels, predictions):
    classes = {"negative", "neutral", "positive"}
    if len(labels) != len(predictions) or not len(labels):
        raise ValueError("Aligned nonempty labels and predictions required")
    if not (set(labels) | set(predictions)).issubset(classes):
        raise ValueError("Unexpected sentiment class")
    return {"accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=sorted(classes),
                                       average="macro", zero_division=0))}


def recall_at_k(retrieved_ids, relevant_ids, k):
    relevant = set(relevant_ids)
    if k < 1 or not relevant:
        raise ValueError("Positive k and at least one relevant ID required")
    return len(set(retrieved_ids[:k]) & relevant) / len(relevant)


def extract_citations(answer):
    return re.findall(r"\[([^\[\]\n]+)\]", answer)


def citation_membership(answer, retrieved_ids):
    """ID membership is not factual entailment or human citation correctness."""
    citations = extract_citations(answer)
    allowed = set(retrieved_ids)
    invalid = sorted(set(citations) - allowed)
    return {"citation_count": len(citations), "invalid_citations": invalid,
            "citation_id_precision": sum(c in allowed for c in citations) / len(citations) if citations else None,
            "limitation": "Checks retrieved ID membership only, not whether evidence supports each claim"}
