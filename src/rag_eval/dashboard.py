"""Streamlit dashboard for interactive RAG evaluation."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .dataset import load_jsonl
from .evaluator import RAGASEvaluator
from .pipeline import MockRAGPipeline

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
DEMO_DATA = Path(__file__).parent.parent.parent / "data" / "sample_qa.jsonl"


def run():
    st.set_page_config(page_title="RAG Eval Framework", layout="wide")
    st.title("RAG Evaluation Framework")
    st.caption("Powered by RAGAS metrics — evaluate any RAG pipeline in minutes.")

    # --- Sidebar: dataset selection ---
    with st.sidebar:
        st.header("Configuration")
        dataset_source = st.radio("Dataset source", ["Demo dataset", "Upload JSONL"])
        pipeline_mode = st.radio("Pipeline", ["Mock (no API)", "LangChain RAG"])
        run_btn = st.button("Run Evaluation", type="primary")

    samples = None
    if dataset_source == "Demo dataset" and DEMO_DATA.exists():
        samples = load_jsonl(DEMO_DATA)
        st.info(f"Loaded {len(samples)} questions from demo dataset.")
    elif dataset_source == "Upload JSONL":
        uploaded = st.file_uploader("Upload JSONL", type=["jsonl", "json"])
        if uploaded:
            with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
                f.write(uploaded.read())
                samples = load_jsonl(f.name)
            st.info(f"Loaded {len(samples)} questions.")

    if not run_btn or samples is None:
        st.info("Configure a dataset and click **Run Evaluation** to start.")
        return

    with st.spinner("Running evaluation…"):
        evaluator = RAGASEvaluator()
        pipeline = MockRAGPipeline(samples)
        df = evaluator.evaluate_mock(samples, pipeline)

    st.success("Evaluation complete!")

    # --- Metric summary cards ---
    cols = st.columns(4)
    for col, metric in zip(cols, METRICS):
        col.metric(metric.replace("_", " ").title(), f"{df[metric].mean():.3f}")

    # --- Per-question heatmap ---
    st.subheader("Per-Question Scores")
    heat_df = df[["question"] + METRICS].set_index("question")
    fig = px.imshow(heat_df.T, color_continuous_scale="RdYlGn", range_color=[0, 1],
                    labels={"color": "Score"}, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    # --- Worst performing questions ---
    st.subheader("Lowest-Scoring Questions")
    df["mean_score"] = df[METRICS].mean(axis=1)
    st.dataframe(
        df.nsmallest(10, "mean_score")[["question", "mean_score"] + METRICS],
        use_container_width=True,
    )

    # --- Score distributions ---
    st.subheader("Score Distributions")
    dist_cols = st.columns(4)
    for col, metric in zip(dist_cols, METRICS):
        fig = px.histogram(df, x=metric, nbins=10, range_x=[0, 1], title=metric)
        col.plotly_chart(fig, use_container_width=True)

    # --- Export ---
    st.subheader("Export Results")
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
    c2.download_button("Download JSON", df.to_json(orient="records", indent=2), "results.json", "application/json")


if __name__ == "__main__":
    run()
