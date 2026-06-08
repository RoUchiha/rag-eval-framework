"""Dataset loader and validator for RAG evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Union

from .models import RAGSample


def load_jsonl(path: Union[str, Path]) -> List[RAGSample]:
    """Load a JSONL file with schema {question, ground_truth_answer, contexts}."""
    samples = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        try:
            samples.append(RAGSample(**raw))
        except Exception as e:
            raise ValueError(f"Invalid record at line {i+1}: {e}") from e
    return samples


def load_huggingface(dataset_name: str, split: str = "train") -> List[RAGSample]:
    """Load a dataset from HuggingFace hub."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(dataset_name, split=split)
    samples = []
    for row in ds:
        samples.append(
            RAGSample(
                question=row["question"],
                ground_truth_answer=row.get("ground_truth_answer", row.get("answer", "")),
                contexts=row.get("contexts", [row.get("context", "")]),
            )
        )
    return samples
