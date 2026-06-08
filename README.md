# RAG Evaluation Framework

> A modular, production-ready harness for evaluating Retrieval-Augmented Generation (RAG) pipelines using RAGAS metrics — with an interactive Streamlit dashboard, CSV/JSON export, and a clean pipeline abstraction that lets you plug in any RAG system.

---

## What Is This?

**RAG** (Retrieval-Augmented Generation) is the dominant architecture for building LLM applications that need to answer questions from private documents, knowledge bases, or real-time data. Instead of relying solely on what the model learned during training, a RAG system:

1. **Retrieves** relevant documents from a vector database
2. **Feeds** them as context to an LLM
3. **Generates** an answer grounded in those documents

The problem: how do you know if your RAG system is actually good? Is it retrieving the right documents? Is the answer faithful to those documents? Is it actually answering the question?

**RAGAS** (RAG Assessment) is the industry-standard framework for answering those questions. This project wraps RAGAS in a clean, reusable harness with:
- A **dataset loader** for JSONL files and HuggingFace datasets
- A **pipeline abstraction** so you can swap any RAG system in with minimal code
- A **Streamlit dashboard** for visual, interactive evaluation
- **CSV/JSON export** for downstream analysis

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Evaluation Framework                      │
│                                                                   │
│   JSONL / HuggingFace ──► [Dataset Loader] ──► List[RAGSample]  │
│                                                        │          │
│                                                        ▼          │
│   Your RAG System ──► [RAGPipeline adapter] ──► RAGResponse      │
│   (LangChain / custom)    (swap in 5 lines)    (answer+contexts) │
│                                                        │          │
│                                                        ▼          │
│                              [RAGAS Evaluator]                    │
│                         ┌──────────────────────┐                  │
│                         │ faithfulness         │                  │
│                         │ answer_relevancy     │                  │
│                         │ context_recall       │                  │
│                         │ context_precision    │                  │
│                         └──────────────────────┘                  │
│                                    │                              │
│                                    ▼                              │
│              [Streamlit Dashboard] + CSV/JSON export              │
└─────────────────────────────────────────────────────────────────┘
```

### The Four RAGAS Metrics Explained

| Metric | Question It Answers | Why It Matters |
|--------|--------------------|--------------------|
| **Faithfulness** | Are all claims in the answer supported by the retrieved contexts? | Catches hallucination — the LLM inventing facts not in the documents |
| **Answer Relevancy** | Does the answer actually address the question asked? | Catches evasion — verbose answers that dodge the real question |
| **Context Recall** | Did the retriever find the documents needed to answer the question? | Catches retrieval failure — good LLM, bad retriever |
| **Context Precision** | Are the retrieved documents actually useful? | Catches retrieval noise — retrieving 10 docs when 1 was needed |

All metrics are 0–1, higher is better.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Evaluation metrics | RAGAS |
| Pipeline abstraction | Abstract base class + LangChain adapter |
| Dataset handling | HuggingFace `datasets` + JSONL |
| Data analysis | pandas |
| Visualization | Plotly |
| Dashboard | Streamlit |
| Data models | Pydantic v2 |
| Tests | pytest + pytest-asyncio |

---

## Installation

```bash
git clone https://github.com/RoUchiha/rag-eval-framework.git
cd rag-eval-framework
pip install -e ".[dev]"
```

Set your API key (RAGAS uses an LLM for metric computation):
```bash
export OPENAI_API_KEY=sk-...
# or
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick Start

### Run the Streamlit Dashboard

```bash
streamlit run src/rag_eval/dashboard.py
```

Open `http://localhost:8501`. Select the demo dataset, click **Run Evaluation**, and explore the results.

### Evaluate Programmatically

```python
from rag_eval.dataset import load_jsonl
from rag_eval.evaluator import RAGASEvaluator
from rag_eval.pipeline import MockRAGPipeline

samples = load_jsonl("data/sample_qa.jsonl")
pipeline = MockRAGPipeline(samples)          # swap in your real pipeline here
evaluator = RAGASEvaluator()
df = evaluator.evaluate(samples, pipeline)   # returns a pandas DataFrame

print(df[["question", "faithfulness", "answer_relevancy"]].to_string())
print("\nMean scores:")
print(df[["faithfulness", "answer_relevancy", "context_recall", "context_precision"]].mean())
```

### Plug In Your Own RAG Pipeline

```python
from rag_eval.pipeline import RAGPipeline
from rag_eval.models import RAGResponse

class MyRAGPipeline(RAGPipeline):
    def answer(self, question: str) -> RAGResponse:
        # your retrieval + generation logic here
        contexts = my_vector_db.search(question, top_k=5)
        answer = my_llm.generate(question, contexts)
        return RAGResponse(answer=answer, contexts=contexts)
```

That's it — pass `MyRAGPipeline()` to the evaluator.

---

## Dataset Format

JSONL file, one record per line:
```json
{"question": "What is RAG?", "ground_truth_answer": "RAG stands for...", "contexts": ["Context 1", "Context 2"]}
```

---

## Running Tests

```bash
pytest --cov=src/rag_eval
```

Tests use `MockRAGPipeline` — no LLM API calls required for CI.

---

## Dashboard Screenshots

The dashboard provides:
- **4 metric summary cards** with mean scores
- **Per-question heatmap** — immediately see which questions are problematic
- **Worst-performing questions table** — sortable, filterable
- **Score distribution histograms** — understand variance, not just averages
- **CSV/JSON export** — take results to your own analysis tools

---

## Project Structure

```
rag-eval-framework/
├── src/rag_eval/
│   ├── dataset.py      # JSONL + HuggingFace loader
│   ├── pipeline.py     # abstract base + Mock + LangChain adapters
│   ├── evaluator.py    # RAGAS wrapper → pandas DataFrame
│   ├── dashboard.py    # Streamlit app
│   └── models.py       # Pydantic models
├── tests/              # pytest suite (no API calls)
├── data/
│   └── sample_qa.jsonl # 5-question demo dataset
└── pyproject.toml
```

---

## Extending This

- **Add a LlamaIndex adapter**: subclass `RAGPipeline`, call your LlamaIndex query engine in `answer()`
- **Add cost tracking**: log token counts from each pipeline call; add a `cost_usd` column to the DataFrame
- **Async batch evaluation**: use `asyncio.gather` to run multiple questions in parallel for faster eval
- **Add ARES or TruLens**: swap the RAGAS evaluator for another framework by subclassing `RAGASEvaluator`
