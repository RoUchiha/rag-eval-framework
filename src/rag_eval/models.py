"""Data models for the RAG evaluation framework."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class RAGSample(BaseModel):
    question: str
    ground_truth_answer: str
    contexts: List[str]


class RAGResponse(BaseModel):
    answer: str
    contexts: List[str]
    metadata: Dict[str, Any] = {}


class EvalRecord(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_recall: Optional[float] = None
    context_precision: Optional[float] = None
