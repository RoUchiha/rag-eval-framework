"""Abstract RAG pipeline base class and concrete adapters."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List

from .models import RAGResponse, RAGSample


class RAGPipeline(ABC):
    """Base class for any RAG pipeline under evaluation."""

    @abstractmethod
    def answer(self, question: str) -> RAGResponse:
        """Return an answer and the retrieved contexts for a given question."""


class MockRAGPipeline(RAGPipeline):
    """Deterministic mock pipeline for testing — returns the ground truth answer."""

    def __init__(self, samples: List[RAGSample]):
        self._lookup = {s.question: s for s in samples}

    def answer(self, question: str) -> RAGResponse:
        sample = self._lookup.get(question)
        if sample:
            return RAGResponse(
                answer=sample.ground_truth_answer,
                contexts=sample.contexts,
                metadata={"source": "mock"},
            )
        return RAGResponse(
            answer="I don't know.",
            contexts=[],
            metadata={"source": "mock_fallback"},
        )


class LangChainRAGAdapter(RAGPipeline):
    """Adapter wrapping a LangChain retrieval chain."""

    def __init__(self, chain):
        self._chain = chain

    def answer(self, question: str) -> RAGResponse:
        result = self._chain.invoke({"query": question})
        contexts = [doc.page_content for doc in result.get("source_documents", [])]
        return RAGResponse(
            answer=result.get("result", ""),
            contexts=contexts,
            metadata={"chain_type": type(self._chain).__name__},
        )
