"""
RAGAS Evaluation Dashboard.

Displays eval_results.json as an interactive Streamlit page:
- Scorecard with color-coded pass/fail per metric
- Bar chart of aggregate scores vs targets
- Per-question detail table for debugging weak spots

This page loads pre-computed results. Run `python eval.py` first
to generate eval_results.json.
"""

import json
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="RAG Evaluation", page_icon="📊", layout="wide")
st.title("📊 RAG Evaluation Dashboard")

RESULTS_PATH = "eval_results.json"

# Friendly display names for RAGAS metric keys
METRIC_DISPLAY_NAMES = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision_with_reference": "Context Precision",
    "context_recall": "Context Recall",
}


def load_results() -> dict | None:
    """Load eval_results.json if it exists."""
    if not Path(RESULTS_PATH).exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


results = load_results()

if results is None:
    st.warning(
        "No evaluation results found. Run the evaluation first:\n\n"
        "```bash\npython eval.py\n```"
    )
    st.stop()

# --- Metadata ---
meta = results["metadata"]
model_info = meta.get("model", meta.get("model_rag", "unknown"))
eval_model = meta.get("model_eval", "")
model_str = f"RAG: {model_info}" + (f" | Eval: {eval_model}" if eval_model else "")
st.caption(
    f"Evaluated on {meta['timestamp'][:19]} | "
    f"{meta['num_questions']} questions | "
    f"{model_str}"
)

# --- Scorecard ---
st.subheader("Metric Scorecard")

scores = results["aggregate_scores"]
targets = results["targets"]

cols = st.columns(len(scores))
for col, (metric, score) in zip(cols, scores.items()):
    target = targets.get(metric, 0.7)
    display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
    passed = score >= target

    with col:
        # Color-coded metric card
        color = "#28a745" if passed else "#dc3545"
        status = "PASS" if passed else "FAIL"
        st.markdown(
            f"""
            <div style="
                border: 2px solid {color};
                border-radius: 10px;
                padding: 16px;
                text-align: center;
                margin-bottom: 8px;
            ">
                <div style="font-size: 14px; color: #888;">{display_name}</div>
                <div style="font-size: 36px; font-weight: bold; color: {color};">
                    {score:.2f}
                </div>
                <div style="font-size: 12px; color: #888;">
                    Target: {target:.1f} | {status}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Bar Chart ---
st.subheader("Scores vs Targets")

import pandas as pd
import altair as alt

chart_data = []
for metric, score in scores.items():
    display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
    target = targets.get(metric, 0.7)
    chart_data.append({"Metric": display_name, "Score": score, "Target": target})

chart_df = pd.DataFrame(chart_data)

# Reshape for grouped bar chart: one row per metric per type (Score / Target)
melted = chart_df.melt(id_vars="Metric", value_vars=["Score", "Target"],
                       var_name="Type", value_name="Value")

bars = (
    alt.Chart(melted)
    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
    .encode(
        x=alt.X("Metric:N", axis=alt.Axis(labelAngle=-20), title=None),
        y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 1.0]), title="Score"),
        xOffset="Type:N",  # side-by-side grouping
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(
                domain=["Score", "Target"],
                range=["#4C9AFF", "#FF6B6B"],
            ),
            legend=alt.Legend(title=None, orient="bottom"),
        ),
        tooltip=["Metric:N", "Type:N", alt.Tooltip("Value:Q", format=".3f")],
    )
    .properties(height=350)
)

st.altair_chart(bars, use_container_width=True)

# --- Per-Question Detail Table ---
st.subheader("Per-Question Results")

per_q = results.get("per_question", [])
if per_q:
    table_rows = []
    for item in per_q:
        row = {"Question": item["question"], "Answer": item["answer"][:100] + "..."}
        for metric_key, metric_score in item.get("scores", {}).items():
            display_name = METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
            row[display_name] = (
                round(metric_score, 3) if metric_score is not None else "N/A"
            )
        table_rows.append(row)

    detail_df = pd.DataFrame(table_rows)
    st.dataframe(detail_df, use_container_width=True)

    # Expandable detail view for each question
    st.subheader("Detailed View")
    for i, item in enumerate(per_q):
        with st.expander(f"Q{i + 1}: {item['question']}"):
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown(f"**Reference:** {item['reference']}")
            st.markdown("**Retrieved Contexts:**")
            for j, ctx in enumerate(item.get("contexts", [])):
                st.text_area(
                    f"Chunk {j + 1}",
                    ctx[:500],
                    height=100,
                    key=f"ctx_{i}_{j}",
                    disabled=True,
                )
            st.markdown("**Scores:**")
            for metric_key, metric_score in item.get("scores", {}).items():
                display_name = METRIC_DISPLAY_NAMES.get(metric_key, metric_key)
                if metric_score is not None:
                    target = targets.get(metric_key, 0.7)
                    icon = "✅" if metric_score >= target else "❌"
                    st.write(f"  {icon} {display_name}: {metric_score:.3f}")
                else:
                    st.write(f"  ⚠️ {display_name}: N/A")
else:
    st.info("No per-question results available.")
