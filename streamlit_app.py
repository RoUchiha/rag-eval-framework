"""
RAG Evaluation Framework — Live Demo
Upload a QA dataset, run evaluation, explore RAGAS-style metrics.
"""

import os, json
import streamlit as st
import pandas as pd

st.set_page_config(page_title="RAG Eval Framework", page_icon="📊", layout="wide")

st.title("📊 RAG Evaluation Framework")
st.caption(
    "Evaluate any RAG pipeline across four dimensions: **faithfulness**, **answer relevancy**, "
    "**context recall**, and **context precision**. Upload a QA dataset or use the built-in demo."
)
st.markdown("---")

METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

DEMO_QA = [
    {"question": "What is retrieval-augmented generation?",
     "ground_truth_answer": "RAG combines retrieval from a knowledge base with LLM generation, grounding responses in retrieved documents.",
     "contexts": ["RAG (Retrieval-Augmented Generation) retrieves relevant documents and uses them as context for an LLM to generate accurate, grounded answers."]},
    {"question": "What are the main RAGAS metrics?",
     "ground_truth_answer": "RAGAS metrics include faithfulness, answer relevancy, context recall, and context precision.",
     "contexts": ["RAGAS evaluates faithfulness (are claims grounded?), answer relevancy (does it answer the question?), context recall (right docs retrieved?), and context precision (are retrieved docs useful?)."]},
    {"question": "What is a vector database?",
     "ground_truth_answer": "A vector database stores embeddings and enables fast semantic similarity search.",
     "contexts": ["Vector databases like Pinecone, Weaviate, and FAISS store embedding vectors and support approximate nearest-neighbor search for semantic retrieval."]},
    {"question": "What is prompt engineering?",
     "ground_truth_answer": "Prompt engineering is the practice of designing inputs to LLMs to elicit desired outputs.",
     "contexts": ["Prompt engineering involves crafting inputs — including instructions, examples, and context — to guide an LLM toward producing accurate, relevant, and well-formatted outputs."]},
    {"question": "What is a context window?",
     "ground_truth_answer": "A context window is the maximum number of tokens an LLM can process in a single inference call.",
     "contexts": ["LLMs have a fixed context window — the maximum tokens they can attend to at once. For RAG, retrieved contexts must fit within this window alongside the query."]},
]

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY",""),
        help="Used for LLM-based metric computation.")
    dataset_mode = st.radio("Dataset", ["Demo (5 questions)", "Upload JSONL"])
    st.markdown("---")
    st.markdown("**Metrics computed:**")
    for m in METRICS:
        st.markdown(f"• {m.replace('_',' ').title()}")

# ── dataset ───────────────────────────────────────────────────────────────────
samples = None
if dataset_mode == "Demo (5 questions)":
    samples = DEMO_QA
    st.info(f"Using built-in demo dataset — {len(samples)} questions about RAG & LLMs.")
else:
    uploaded = st.file_uploader("Upload JSONL (schema: question, ground_truth_answer, contexts)", type=["jsonl","json"])
    if uploaded:
        samples = [json.loads(l) for l in uploaded.read().decode().splitlines() if l.strip()]
        st.success(f"Loaded {len(samples)} questions.")

run = st.button("▶️ Run Evaluation", type="primary", use_container_width=True, disabled=samples is None)

# ── evaluation ────────────────────────────────────────────────────────────────
if run and samples:
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar.")
        st.stop()

    os.environ["ANTHROPIC_API_KEY"] = api_key
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    def score_one(question, answer, contexts, ground_truth):
        ctx_str = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        prompt = f"""You are a RAG evaluation judge. Score this QA result on four metrics, each 0.0–1.0.

Question: {question}
Ground truth answer: {ground_truth}
Retrieved contexts:
{ctx_str}
Generated answer: {answer}

Return ONLY JSON:
{{
  "faithfulness": <0-1, are all claims in the answer supported by the contexts?>,
  "answer_relevancy": <0-1, does the answer address the question?>,
  "context_recall": <0-1, do the contexts contain info needed to answer correctly?>,
  "context_precision": <0-1, are the contexts actually useful for this question?>
}}"""
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001", max_tokens=200,
            messages=[{"role":"user","content":prompt}]
        )
        import re, json as _json
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```[a-z]*\n?","",raw).rstrip("`").strip()
        return _json.loads(raw)

    rows = []
    progress = st.progress(0, text="Evaluating questions…")
    for i, s in enumerate(samples):
        # Mock pipeline: answer = ground truth (best case) for demo purposes
        answer = s["ground_truth_answer"]
        scores = score_one(s["question"], answer, s["contexts"], s["ground_truth_answer"])
        rows.append({"question": s["question"], "answer": answer, **scores})
        progress.progress((i+1)/len(samples), text=f"Evaluated {i+1}/{len(samples)}")

    df = pd.DataFrame(rows)
    df["mean_score"] = df[METRICS].mean(axis=1)

    # ── results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📈 Results")

    cols = st.columns(4)
    for col, m in zip(cols, METRICS):
        col.metric(m.replace("_"," ").title(), f"{df[m].mean():.3f}")

    st.markdown("### Per-Question Heatmap")
    try:
        import plotly.express as px
        heat = df.set_index("question")[METRICS]
        heat.index = [q[:45]+"…" if len(q)>45 else q for q in heat.index]
        fig = px.imshow(heat.T, color_continuous_scale="RdYlGn", range_color=[0,1],
                        labels={"color":"Score"}, aspect="auto", height=260)
        fig.update_layout(margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.dataframe(df[["question"]+METRICS].style.background_gradient(cmap="RdYlGn",vmin=0,vmax=1))

    st.markdown("### Lowest-Scoring Questions")
    worst = df.nsmallest(5, "mean_score")[["question","mean_score"]+METRICS]
    st.dataframe(worst.style.format({m:"{:.3f}" for m in METRICS+["mean_score"]}), use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download CSV", df.to_csv(index=False), "rag_eval_results.csv","text/csv")
    c2.download_button("⬇️ Download JSON", df.to_json(orient="records",indent=2), "rag_eval_results.json","application/json")
