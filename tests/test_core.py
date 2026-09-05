import tempfile
import unittest
import numpy as np
from news_agent.preprocessing import clean_text, prepare_documents, chunk_documents
from news_agent.retrieval import HashEncoder, VectorIndex
from news_agent.evaluation import recall_at_k, citation_membership, classification_metrics
from news_agent.rag import answer_question, REFUSAL


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [{"chunk_id": "d1:0", "text": "The company increased profit.", "title": "Profit"}]

    def test_cleaning(self):
        self.assertEqual(clean_text("<p>Hello</p><script>alert(1)</script><p>A &amp; B</p>"), "Hello A & B")

    def test_deduplication_and_conflicting_ids(self):
        documents = prepare_documents([{"text": "Same text"}, {"text": " same   TEXT "}])
        self.assertEqual(len(documents), 1)
        with self.assertRaises(ValueError):
            prepare_documents([{"document_id": "a", "text": "First"}, {"document_id": "a", "text": "Second"}])

    def test_overlap_and_stable_ids(self):
        docs = prepare_documents([{"document_id": "a", "text": "one two three four five"}])
        chunks = chunk_documents(docs, 3, 1)
        self.assertEqual([c["text"] for c in chunks], ["one two three", "three four five"])
        self.assertEqual(chunks[-1]["chunk_id"], "a:1")
        with self.assertRaises(ValueError):
            chunk_documents(docs, 2, 2)

    def test_numpy_retrieval_and_roundtrip(self):
        chunks = [{"chunk_id": "a:0", "text": "profit"}, {"chunk_id": "b:0", "text": "weather"}]
        index = VectorIndex([[1, 0], [0, 2]], chunks, {"encoder": "demo-hash-v1"}, use_faiss=False)
        self.assertEqual(index.search([4, 0], 1)[0]["chunk_id"], "a:0")
        self.assertEqual(index.search([0, 0]), [])
        with tempfile.TemporaryDirectory() as path:
            index.save(path)
            restored = VectorIndex.load(path, use_faiss=False)
            self.assertEqual(restored.search([0, 1], 1)[0]["chunk_id"], "b:0")

    def test_invalid_vectors(self):
        with self.assertRaises(ValueError):
            VectorIndex([[np.nan]], self.evidence)

    def test_hash_encoder_is_deterministic(self):
        np.testing.assert_array_equal(HashEncoder().encode(["Profit up"]), HashEncoder().encode(["Profit up"]))

    def test_recall_counts_all_relevant_items(self):
        self.assertEqual(recall_at_k(["a", "b"], ["a", "c"], 2), 0.5)
        with self.assertRaises(ValueError):
            recall_at_k(["a"], [], 1)

    def test_citation_membership_not_entailment(self):
        result = citation_membership("Profit increased [d1:0] [fake:0]", ["d1:0"])
        self.assertEqual(result["citation_id_precision"], 0.5)
        self.assertEqual(result["invalid_citations"], ["fake:0"])
        self.assertIsNone(citation_membership("No citations", ["d1:0"])["citation_id_precision"])

    def test_refusal_without_api(self):
        def fail(*args):
            self.fail("Should not call API")
        self.assertEqual(answer_question("Why?", [], fail)["answer"], REFUSAL)

    def test_invalid_or_missing_citations_are_rejected(self):
        for answer in ("Profit up", "Profit up [invented]", ""):
            with self.assertRaises(ValueError):
                answer_question("Why?", self.evidence, lambda *args: answer)

    def test_transient_retry_then_success(self):
        calls = []
        def generate(*args):
            calls.append(1)
            if len(calls) == 1:
                raise TimeoutError()
            return "Profit increased [d1:0]"
        result = answer_question("What changed?", self.evidence, generate, sleep=lambda _: None)
        self.assertEqual(result["api_calls"], 2)
        self.assertEqual(result["citation_check"]["citation_id_precision"], 1)

    def test_permanent_failure_not_retried(self):
        calls = []
        def generate(*args):
            calls.append(1)
            raise ValueError("Invalid configuration")
        with self.assertRaises(ValueError):
            answer_question("Why?", self.evidence, generate, sleep=lambda _: None)
        self.assertEqual(len(calls), 1)

    def test_classification_metrics(self):
        labels = ["negative", "neutral", "positive"]
        self.assertEqual(classification_metrics(labels, labels)["macro_f1"], 1)


if __name__ == "__main__":
    unittest.main()
