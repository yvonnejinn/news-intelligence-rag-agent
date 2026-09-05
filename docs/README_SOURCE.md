> Owner-supplied specification. Refer to the root README for current verified implementation and execution status.

# News Intelligence Analysis & Retrieval Agent

An evaluation-first NLP project for financial-news sentiment classification, semantic retrieval, source-grounded question answering, and lightweight analyst reporting.

> Project status: the complete Phase 1 code path and offline unit tests are implemented. FinBERT, retrieval, and Gemini quality metrics are intentionally not reported until the labeled dataset, model weights, and evaluation run are available.

## 1. Project overview

News research usually combines two different problems:

1. classify the tone of individual news passages; and
2. retrieve evidence before asking a language model to summarize an event.

This repository keeps those problems separate and evaluates them separately. FinBERT/RoBERTa handles sentiment classification. Sentence Transformer embeddings and FAISS handle retrieval. Gemini generates an answer only from retrieved passages and must cite their source IDs.

## 2. Implemented scope

- deterministic HTML cleanup and whitespace normalization;
- content-hash deduplication;
- overlapping word-based chunking with stable document and chunk IDs;
- FinBERT/RoBERTa fine-tuning entry point using Hugging Face Trainer;
- stratified sentiment evaluation using Accuracy and Macro-F1;
- Sentence Transformer embeddings and cosine-similarity retrieval;
- FAISS index with a NumPy fallback for offline tests;
- source-constrained Gemini prompts and exponential-backoff retries;
- Recall@K and citation-precision evaluation;
- RSS collection for a fresh-news retrieval corpus;
- Streamlit interfaces for search/Q&A and human label review.

FastAPI, LoRA, reranking, hybrid BM25 retrieval, and multi-tool agent orchestration are deferred until the Phase 1 metrics are established.

## 3. Architecture

```text
Financial PhraseBank ──> text,label ──> FinBERT/RoBERTa ──> Accuracy / Macro-F1

RSS feeds ──> clean ──> deduplicate ──> chunk ──> embeddings ──> FAISS
                                                                    │
user question ──────────────────────────────────────────────────────┘
                  ──> top-k evidence ──> Gemini prompt ──> cited answer
                                           │
                                           └──> Recall@K / citation checks / latency
```

## 4. Data

### Sentiment training baseline

- Dataset: [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank)
- Recommended configuration: `sentences_allagree`
- Schema after preparation: `text,label`
- Labels: `negative`, `neutral`, `positive`
- License: CC BY-NC-SA 3.0. Review the dataset license before commercial use.

The all-agreement subset is the quickest credible baseline because its labels were created by human annotators. The script keeps the original labels and converts only the column names.

### Fresh-news retrieval corpus

- Example source: [BBC Business RSS](https://feeds.bbci.co.uk/news/business/rss.xml)
- Output schema: `document_id,title,text,source,published`

RSS content is used for learning and demonstration. Keep source URLs, respect publisher terms, and do not redistribute full copyrighted articles.

### Human-reviewed evaluation data

Fresh RSS items should be manually reviewed in `label_app.py`. Use this set only as a fixed out-of-domain test set. Gemini-generated labels may help triage examples, but they must not be treated as final ground truth when Gemini is also part of the system being evaluated.

Raw and processed datasets are excluded from Git. Small schema examples are stored in `data/sample/`.

## 5. Repository layout

```text
.
├── app.py                         # Search and cited Q&A interface
├── label_app.py                   # Human sentiment-label review interface
├── data/sample/                   # Small input/evaluation examples
├── scripts/
│   ├── prepare_phrasebank.py      # Export Financial PhraseBank to text,label
│   ├── collect_rss.py             # Collect fresh RSS summaries
│   ├── train_sentiment.py         # Fine-tune FinBERT/RoBERTa
│   ├── build_index.py             # Build embeddings and vector index
│   └── evaluate_retrieval.py      # Calculate Recall@K
├── src/news_agent/
│   ├── preprocessing.py           # Clean, deduplicate, and chunk
│   ├── retrieval.py               # FAISS/NumPy vector index
│   ├── rag.py                     # Gemini prompt and retry logic
│   └── evaluation.py              # Classification/retrieval/citation metrics
└── tests/test_core.py             # Offline core tests
```

## 6. Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Create an environment variable only when running Gemini:

```bash
export GEMINI_API_KEY="your-key"
```

Never commit API keys or `.env` files.

## 7. End-to-end workflow

### Step 1: prepare sentiment data

Online Hugging Face loading:

```bash
python scripts/prepare_phrasebank.py \
  --config sentences_allagree \
  --output data/processed/sentiment.csv
```

Local Parquet fallback:

```bash
python scripts/prepare_phrasebank.py \
  --parquet data/raw/financial_phrasebank_allagree.parquet \
  --output data/processed/sentiment.csv
```

### Step 2: fine-tune the sentiment model

```bash
python scripts/train_sentiment.py \
  --data data/processed/sentiment.csv \
  --model ProsusAI/finbert \
  --output artifacts/finbert \
  --epochs 3
```

The script uses a stratified 80/20 split with seed 42 and stores the model plus `metrics.json`.

### Step 3: collect fresh news

```bash
python scripts/collect_rss.py \
  --feed https://feeds.bbci.co.uk/news/business/rss.xml \
  --output data/processed/articles.csv
```

Pass `--feed` repeatedly to combine multiple permitted RSS sources.

### Step 4: build the retrieval index

```bash
python scripts/build_index.py \
  --data data/processed/articles.csv \
  --output artifacts/index
```

### Step 5: create and evaluate retrieval questions

Prepare a CSV with `question,relevant_chunk_ids`, then run:

```bash
python scripts/evaluate_retrieval.py \
  --eval data/processed/eval_questions.csv \
  --index artifacts/index
```

### Step 6: review labels and open the demo

```bash
streamlit run label_app.py
streamlit run app.py
```

## 8. Evaluation design

| Component | Primary metric | Why it is used |
|---|---|---|
| Sentiment classification | Macro-F1 | Gives equal weight to negative, neutral, and positive classes |
| Sentiment classification | Accuracy | Easy-to-read secondary measure |
| Retrieval | Recall@K | Tests whether relevant evidence appears in the top K passages |
| Answer grounding | Citation precision | Checks whether cited IDs belong to retrieved evidence |
| Product performance | Latency and API calls | Measures usability and operating cost |

A final answer-quality set should also record human ratings for relevance, factual consistency, and citation correctness.

## 9. Current verified results

| Check | Result |
|---|---|
| Python syntax compilation | Passed |
| Cleaning/deduplication/chunking test | Passed |
| Vector retrieval test | Passed |
| Recall@K calculation test | Passed |
| Citation validation test | Passed |
| FinBERT Macro-F1 | Not run yet |
| Retrieval Recall@K on reviewed set | Not run yet |
| Gemini answer-quality evaluation | Not run yet |

The absence of model metrics is deliberate: the repository does not publish results that cannot be reproduced from committed code and a documented dataset version.

## 10. Limitations

- RSS summaries are shorter and less detailed than full articles.
- Financial PhraseBank is a sentence-level dataset and may not transfer perfectly to current long-form news.
- Vector similarity alone may miss exact names, tickers, dates, and numbers.
- Citation-format validation does not prove that every claim is factually supported.
- Gemini availability, latency, and cost vary by model and account settings.

## 11. Next milestones

1. Produce a fixed train/validation/test manifest and record dataset hashes.
2. Compare zero-shot FinBERT with fine-tuned FinBERT/RoBERTa.
3. Manually review at least 150 fresh-news examples for out-of-domain testing.
4. Add reranking and compare Recall@K/latency against the current FAISS baseline.
5. Publish a Streamlit screenshot and machine-readable evaluation report.

## 12. Resume evidence boundary

Safe to claim now: implementation of the preprocessing, training, vector retrieval, grounded prompting, evaluation, and Streamlit code paths.

Do not claim a FinBERT F1 score, Recall@K improvement, hallucination reduction percentage, production deployment, or API cost reduction until those results are generated and stored in the repository.
