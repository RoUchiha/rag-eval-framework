"""RAGAS evaluator — runs the pipeline and computes metrics."""

from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .models import EvalRecord, RAGSample
from .pipeline import RAGPipeline


class RAGASEvaluator:
    """Runs a RAGPipeline over a dataset and evaluates with RAGAS metrics."""

    def __init__(self, metrics: Optional[List[str]] = None):
        self.metrics = metrics or ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    def collect(self, samples: List[RAGSample], pipeline: RAGPipeline) -> List[EvalRecord]:
        records = []
        for sample in samples:
            response = pipeline.answer(sample.question)
            records.append(
                EvalRecord(
                    question=sample.question,
                    answer=response.answer,
                    contexts=response.contexts,
                    ground_truth=sample.ground_truth_answer,
                )
            )
        return records

    def evaluate(self, samples: List[RAGSample], pipeline: RAGPipeline) -> pd.DataFrame:
        """Run pipeline + RAGAS evaluation. Returns per-question DataFrame."""
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )
        from datasets import Dataset  # type: ignore

        records = self.collect(samples, pipeline)

        dataset = Dataset.from_list([
            {
                "question": r.question,
                "answer": r.answer,
                "contexts": r.contexts,
                "ground_truth": r.ground_truth,
            }
            for r in records
        ])

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        )
        df = result.to_pandas()
        return df

    def evaluate_mock(self, samples: List[RAGSample], pipeline: RAGPipeline) -> pd.DataFrame:
        """Evaluate without calling RAGAS (for testing without LLM API)."""
        records = self.collect(samples, pipeline)
        rows = []
        for r in records:
            # Trivially score mock pipeline: answer matches ground truth → 1.0
            score = 1.0 if r.answer == r.ground_truth else 0.5
            rows.append({
                "question": r.question,
                "answer": r.answer,
                "faithfulness": score,
                "answer_relevancy": score,
                "context_recall": score,
                "context_precision": score,
            })
        return pd.DataFrame(rows)
