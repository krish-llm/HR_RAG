"""Page 1: Ask HR Policy — Grounded Q&A Playground."""

import time
import streamlit as st
from api_client import HRPolicyAPIClient

st.set_page_config(page_title="Ask HR Policy", page_icon="💬", layout="wide")

st.title("💬 Ask HR Policy")
st.markdown("Query the HR Policy RAG platform with grounded answers, strict citations, and conflict detection.")

client = HRPolicyAPIClient()

# Tenant Context Preview from Session State
school_ctx = {
    "school_id": st.session_state.get("school_id") or None,
    "group_id": st.session_state.get("group_id") or None,
    "branch_id": st.session_state.get("branch_id") or None,
    "state": st.session_state.get("state") or None,
}

with st.expander("⚙️ Request Scope & Employee Context", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**School Tenant Scope** (configure in sidebar)")
        st.json({k: v for k, v in school_ctx.items() if v})

    with col2:
        st.markdown("**Optional Employee Context**")
        staff_cat = st.selectbox(
            "Staff Category",
            options=["", "teaching", "non_teaching", "administrative"],
            index=0,
        )
        emp_type = st.selectbox(
            "Employment Type",
            options=["", "full_time", "part_time", "contract", "probationary"],
            index=0,
        )
        dept = st.text_input("Department", value="")

    col_k, col_tok = st.columns(2)
    with col_k:
        top_k = st.slider("Top-K Candidates", min_value=1, max_value=20, value=10)
    with col_tok:
        token_budget = st.slider("Context Token Budget", min_value=200, max_value=8000, value=2000, step=100)

emp_ctx = {}
if staff_cat:
    emp_ctx["staff_category"] = staff_cat
if emp_type:
    emp_ctx["employment_type"] = emp_type
if dept:
    emp_ctx["department"] = dept

# Question Input
question = st.text_area(
    "HR Policy Question",
    placeholder="e.g. How many days of casual leave are full-time teachers entitled to annually?",
    height=90,
)

if st.button("Submit Question", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving policies, reranking, and generating grounded answer..."):
            start_time = time.perf_counter()
            try:
                response = client.ask_question(
                    question=question.strip(),
                    school_context=school_ctx,
                    employee_context=emp_ctx if emp_ctx else None,
                    top_k=top_k,
                    token_budget=token_budget,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000
                data = response.get("data", {})

                st.divider()

                # Status Alerts
                is_insufficient = data.get("insufficient_evidence", False)
                is_conflicting = data.get("conflicting_evidence", False)
                limitations = data.get("limitations")

                if is_insufficient:
                    st.warning("⚠️ **Insufficient Evidence**: The available policy documents do not contain sufficient evidence to answer this question.")
                if is_conflicting:
                    st.error("🚨 **Conflicting Policies Detected**: Multiple active policies contain contradictory rules regarding this topic.")

                # Answer Display
                st.subheader("Answer")
                st.markdown(data.get("answer", "No answer generated."))

                # Limitations Box
                if limitations:
                    with st.expander("ℹ️ Policy Limitations & Boundary Notes", expanded=True):
                        st.info(limitations)

                # Citations Display
                citations = data.get("citations", [])
                st.subheader(f"📚 Citations ({len(citations)})")
                if citations:
                    for idx, cit in enumerate(citations, start=1):
                        label = cit.get("citation_label", f"[Source {idx}]")
                        doc_name = cit.get("document_name", "Unknown Policy")
                        version = cit.get("version", 1)
                        heading = cit.get("heading") or "General Policy"
                        p_start = cit.get("page_start", 1)
                        p_end = cit.get("page_end", 1)
                        pages_str = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"

                        with st.container():
                            st.markdown(
                                f"**{label}**: *{doc_name}* (v{version}) | **Section**: {heading} | **{pages_str}**"
                            )
                            st.caption(f"Doc ID: `{cit.get('document_id')}` | Reference: `{cit.get('citation_reference')}`")
                else:
                    st.write("No citations available.")

                # Metadata Footer
                st.divider()
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Model", data.get("model_name", "N/A"))
                m_col2.metric("Context Tokens", data.get("context_tokens", 0))
                m_col3.metric("Response Time", f"{latency_ms:.0f} ms")

            except Exception as exc:
                st.error(f"Failed to generate answer: {exc}")
