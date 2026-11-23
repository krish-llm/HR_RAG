"""Page 4: Evaluation Dashboard — Run and Monitor RAG Quality Benchmarks."""

import json
from pathlib import Path
import subprocess
import sys
import streamlit as st

st.set_page_config(page_title="Evaluation Dashboard", page_icon="📊", layout="wide")

st.title("📊 Evaluation Dashboard")
st.markdown("Run automated RAG quality benchmarks measuring Retrieval (P@K, R@K, MRR, nDCG) and Generation (Completeness, Citation Validity, Conflict Detection).")

REPORTS_DIR = Path("data/evaluation/reports")
DATASET_PATH = Path("data/evaluation/dataset_v1.json")

# Benchmark Execution Panel
st.subheader("Run Evaluation Benchmark")

col_mode, col_btn = st.columns([2, 1])
with col_mode:
    eval_mode = st.radio(
        "Evaluation Target",
        options=["Full Production Pipeline (Live MongoDB, ChromaDB, Azure OpenAI)", "Offline Dry-Run (Mock Services)"],
        index=0,
    )
with col_btn:
    st.write("")
    st.write("")
    run_btn = st.button("🚀 Run Benchmark", type="primary", use_container_width=True)

if run_btn:
    cmd = [
        sys.executable,
        "-m",
        "app.modules.evaluation.cli",
        "--dataset",
        str(DATASET_PATH),
        "--output-dir",
        str(REPORTS_DIR),
    ]
    if "Offline" in eval_mode:
        cmd.append("--mock")

    with st.spinner("Executing evaluation dataset across pipeline..."):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 or proc.stdout:
            st.success("Benchmark Execution Completed!")
        else:
            st.error(f"Execution Error: {proc.stderr}")

st.divider()

# Report Selector
st.subheader("Benchmark Reports")
available_reports = sorted(list(REPORTS_DIR.glob("evaluation_report_*.json")), reverse=True) if REPORTS_DIR.exists() else []

if not available_reports:
    st.info("No benchmark reports found. Click 'Run Benchmark' above to generate an evaluation report.")
else:
    selected_report_path = st.selectbox(
        "Select Benchmark Report",
        options=available_reports,
        format_func=lambda p: p.name,
    )

    try:
        with open(selected_report_path, "r", encoding="utf-8") as f:
            report_data = json.load(f)

        # Overview Metrics Cards
        tot = report_data.get("total_cases", 0)
        passed = report_data.get("passed_cases", 0)
        failed = report_data.get("failed_cases", 0)
        pass_rate = report_data.get("pass_rate", 0.0)
        mean_lat = report_data.get("mean_latency_ms", 0.0)
        p95_lat = report_data.get("p95_latency_ms", 0.0)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Test Cases", tot)
        c2.metric("Pass Rate", f"{pass_rate * 100:.1f}%", delta=f"{passed} passed")
        c3.metric("Failed Cases", failed, delta=f"-{failed}" if failed else None, delta_color="inverse")
        c4.metric("Latency (Mean / P95)", f"{mean_lat:.1f} ms / {p95_lat:.1f} ms")

        # Retrieval Metrics Section
        st.subheader("🎯 Retrieval Quality Metrics (Phases 6 & 7)")
        ret = report_data.get("retrieval_summary", {})
        mrr = ret.get("mrr", 0.0)
        p_at_k = ret.get("mean_precision_at_k", {})
        r_at_k = ret.get("mean_recall_at_k", {})
        ndcg_at_k = ret.get("mean_ndcg_at_k", {})

        st.metric("Mean Reciprocal Rank (MRR)", f"{mrr:.4f}")

        k_keys = sorted([int(k) for k in p_at_k.keys()])
        ret_rows = []
        for k in k_keys:
            ret_rows.append({
                "Rank Cutoff": f"Top-{k}",
                "Precision@K": f"{p_at_k.get(str(k), 0.0):.4f}",
                "Recall@K": f"{r_at_k.get(str(k), 0.0):.4f}",
                "nDCG@K": f"{ndcg_at_k.get(str(k), 0.0):.4f}",
            })
        st.table(ret_rows)

        # Answer Generation Metrics Section
        ans = report_data.get("answer_summary")
        if ans:
            st.subheader("💬 Answer Generation & Grounding Metrics (Phase 8)")
            a1, a2, a3 = st.columns(3)
            a1.metric("Factual Point Coverage", f"{ans.get('mean_point_coverage', 0.0) * 100:.1f}%")
            a2.metric("Citation Validity", f"{ans.get('mean_citation_validity', 1.0) * 100:.1f}%")
            a3.metric("Citation Coverage", f"{ans.get('mean_citation_coverage', 0.0) * 100:.1f}%")

            a4, a5, a6 = st.columns(3)
            a4.metric("Insufficient Evidence Accuracy", f"{ans.get('insufficient_evidence_accuracy', 1.0) * 100:.1f}%")
            a5.metric("Conflict Detection Accuracy", f"{ans.get('conflicting_evidence_accuracy', 1.0) * 100:.1f}%")
            a6.metric("Unsupported Claims Rate", f"{ans.get('unsupported_claims_rate', 0.0) * 100:.1f}%", delta_color="inverse")

        # Granular Case Results Table
        st.subheader("📋 Granular Case Results")
        cases = report_data.get("case_results", [])
        case_rows = []
        for c in cases:
            case_rows.append({
                "Case ID": c.get("case_id"),
                "Name": c.get("name"),
                "Status": "✅ PASS" if c.get("passed") else "❌ FAIL",
                "Latency (ms)": f"{c.get('total_latency_ms', 0.0):.1f}",
                "Failure Reasons": " | ".join(c.get("failure_reasons", [])) or "None",
            })
        st.table(case_rows)

        # Expandable Details for Failed Test Cases
        failed_cases = [c for c in cases if not c.get("passed")]
        if failed_cases:
            st.subheader(f"⚠️ Failed Test Cases ({len(failed_cases)})")
            for fc in failed_cases:
                with st.expander(f"❌ {fc.get('case_id')}: {fc.get('name')}", expanded=True):
                    st.markdown(f"**Question**: {fc.get('question')}")
                    for reason in fc.get("failure_reasons", []):
                        st.error(f"Failure Reason: {reason}")
                    if fc.get("actual_answer"):
                        st.markdown(f"**Actual Answer**: {fc.get('actual_answer')}")
                    st.json({
                        "retrieval_metrics": fc.get("retrieval_metrics"),
                        "answer_metrics": fc.get("answer_metrics"),
                    })

        # Report Download Buttons
        st.divider()
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            with open(selected_report_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="⬇️ Download Full JSON Report",
                    data=f.read(),
                    file_name=selected_report_path.name,
                    mime="application/json",
                    use_container_width=True,
                )

        csv_path = selected_report_path.parent / selected_report_path.name.replace("evaluation_report_", "evaluation_cases_").replace(".json", ".csv")
        with dl_col2:
            if csv_path.exists():
                with open(csv_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="⬇️ Download Case-Level CSV Report",
                        data=f.read(),
                        file_name=csv_path.name,
                        mime="text/csv",
                        use_container_width=True,
                    )

    except Exception as exc:
        st.error(f"Failed to load evaluation report: {exc}")
