# CLAUDE.md — RAG Evaluation Framework

This file gives Claude Code full context for this project. Read it before making any changes.

---

## What This Project Does

A modular evaluation harness for RAG (Retrieval-Augmented Generation) pipelines. Takes a QA dataset and a callable RAG pipeline, runs it through RAGAS metrics (faithfulness, answer_relevancy, context_recall, context_precision), and produces a scored DataFrame + Streamlit dashboard.

**Core abstraction**: `RAGPipeline` is an abstract base class with a single method `answer(question: str) -> RAGResponse`. Any RAG system — LangChain, LlamaIndex, custom — can be wrapped in a subclass and evaluated without changing the evaluation code.

---

## Repository Layout

```
rag-eval-framework/
├── src/rag_eval/
│   ├── models.py        # Pydantic: RAGSample, RAGResponse, EvalRecord
│   ├── dataset.py       # JSONL loader + HuggingFace datasets loader
│   ├── pipeline.py      # RAGPipeline (abstract), MockRAGPipeline, LangChainRAGAdapter
│   ├── evaluator.py     # RAGASEvaluator: collect() → evaluate() → pd.DataFrame
│   │                    # evaluate_mock() skips RAGAS for testing without LLM API
│   └── dashboard.py     # Streamlit app (run with: streamlit run src/rag_eval/dashboard.py)
├── tests/
│   └── test_evaluator.py    # MockRAGPipeline tests, JSONL loader tests
├── data/
│   └── sample_qa.jsonl  # 5-question demo dataset (RAG, NLI, embeddings topics)
└── pyproject.toml
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Evaluation metrics | `ragas` |
| Pipeline abstraction | `langchain`, `langchain-community` |
| Datasets | `datasets` (HuggingFace) |
| Data analysis | `pandas` |
| Visualization | `plotly` |
| Dashboard | `streamlit` |
| Data models | `pydantic` v2 |
| Tests | `pytest`, `pytest-asyncio` |

---

## Environment

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...   # RAGAS uses OpenAI by default for metric computation
# or configure RAGAS to use Anthropic
```

## Commands

```bash
# Launch dashboard
streamlit run src/rag_eval/dashboard.py

# Run tests (no API calls — uses evaluate_mock())
pytest

# Programmatic evaluation
python -c "
from rag_eval.dataset import load_jsonl
from rag_eval.evaluator import RAGASEvaluator
from rag_eval.pipeline import MockRAGPipeline
samples = load_jsonl('data/sample_qa.jsonl')
df = RAGASEvaluator().evaluate_mock(samples, MockRAGPipeline(samples))
print(df)
"
```

---

## Key Design Decisions

- **Abstract pipeline**: `RAGPipeline` base class means swapping from LangChain to LlamaIndex to a custom pipeline is one subclass, zero evaluator changes
- **`evaluate_mock()`**: runs the full pipeline without calling RAGAS (uses trivial scoring) — enables CI without LLM API key
- **`evaluate()`**: calls real RAGAS metrics — needs an LLM API key; produces the actual faithfulness/relevancy/recall/precision scores
- **Streamlit dashboard**: uses `evaluate_mock()` by default to keep demo instant; swap to `evaluate()` for real scores

---

## JSONL Dataset Schema

```json
{"question": "...", "ground_truth_answer": "...", "contexts": ["context 1", "context 2"]}
```

---

## Course Context

Built as part of the **UT Austin AI & Machine Learning** program (McCombs, 23-week executive program).
- **Course 03** — RAG pipeline architecture and LLM evaluation
- **Course 05** — Streamlit deployment
- **Tools**: LangChain, HuggingFace `datasets`, Streamlit

---

## Stretch Goals (not yet implemented)

- Async batch evaluation (`asyncio.gather` across questions)
- LlamaIndex adapter (`LlamaIndexRAGAdapter`)
- Cost tracking: log token counts + cost per eval run
- PDF summary report export
