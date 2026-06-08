"""Tests for the RAG evaluator."""

import pandas as pd
import pytest

from rag_eval.dataset import load_jsonl
from rag_eval.evaluator import RAGASEvaluator
from rag_eval.models import RAGSample
from rag_eval.pipeline import MockRAGPipeline


SAMPLES = [
    RAGSample(
        question="What is RAG?",
        ground_truth_answer="RAG stands for Retrieval-Augmented Generation.",
        contexts=["RAG combines retrieval and generation."],
    ),
    RAGSample(
        question="What is NLI?",
        ground_truth_answer="NLI stands for Natural Language Inference.",
        contexts=["NLI determines textual entailment."],
    ),
]


def test_mock_pipeline_returns_correct_answer():
    pipeline = MockRAGPipeline(SAMPLES)
    response = pipeline.answer("What is RAG?")
    assert response.answer == "RAG stands for Retrieval-Augmented Generation."
    assert len(response.contexts) > 0


def test_mock_pipeline_fallback():
    pipeline = MockRAGPipeline(SAMPLES)
    response = pipeline.answer("Unknown question?")
    assert "don't know" in response.answer.lower()


def test_evaluate_mock_returns_dataframe():
    pipeline = MockRAGPipeline(SAMPLES)
    evaluator = RAGASEvaluator()
    df = evaluator.evaluate_mock(SAMPLES, pipeline)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "faithfulness" in df.columns
    assert "answer_relevancy" in df.columns


def test_evaluate_mock_perfect_scores():
    pipeline = MockRAGPipeline(SAMPLES)
    evaluator = RAGASEvaluator()
    df = evaluator.evaluate_mock(SAMPLES, pipeline)
    # Mock pipeline returns exact ground truth → all scores 1.0
    assert (df["faithfulness"] == 1.0).all()


def test_load_jsonl(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(
        '{"question":"Q1","ground_truth_answer":"A1","contexts":["C1"]}\n'
        '{"question":"Q2","ground_truth_answer":"A2","contexts":["C2"]}\n'
    )
    samples = load_jsonl(f)
    assert len(samples) == 2
    assert samples[0].question == "Q1"
