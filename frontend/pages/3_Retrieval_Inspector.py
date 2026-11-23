"""Page 3: Retrieval Inspector — Inspect Vector Retrieval, Reranking, and Context Assembly."""

import streamlit as st
from api_client import HRPolicyAPIClient

st.set_page_config(page_title="Retrieval Inspector", page_icon="🔍", layout="wide")

st.title("🔍 Retrieval Inspector")
st.markdown("Inspect raw semantic candidate retrieval, deterministic composite reranking, and context window assembly.")

client = HRPolicyAPIClient()

school_ctx = {
    "school_id": st.session_state.get("school_id") or None,
    "group_id": st.session_state.get("group_id") or None,
    "branch_id": st.session_state.get("branch_id") or None,
    "state": st.session_state.get("state") or None,
}

question = st.text_input(
    "Query / Question",
    value="How many days of sick leave require a medical certificate?",
)

col1, col2, col3 = st.columns(3)
with col1:
    top_k = st.slider("Candidate Top-K", min_value=1, max_value=20, value=5)
with col2:
    token_budget = st.slider("Context Token Budget", min_value=200, max_value=8000, value=2000, step=100)
with col3:
    staff_cat = st.selectbox("Staff Category Filter", options=["", "teaching", "non_teaching", "administrative"])

emp_ctx = {"staff_category": staff_cat} if staff_cat else None

tab_context, tab_semantic = st.tabs(["📑 Phase 7 Context & Reranking", "🎯 Phase 6 Raw Semantic Retrieval"])

# ==============================================================================
# TAB 1: Phase 7 Reranked Context & Manifest Inspector
# ==============================================================================
with tab_context:
    if st.button("Inspect Reranked Context", type="primary"):
        with st.spinner("Building reranked context window..."):
            try:
                res = client.inspect_context(
                    question=question,
                    school_context=school_ctx,
                    employee_context=emp_ctx,
                    top_k=top_k,
                    token_budget=token_budget,
                )
                data = res.get("data", {})

                # 1. Query Understanding
                qu = data.get("query_understanding", {})
                with st.expander("🧠 Query Understanding & Normalization", expanded=True):
                    q_col1, q_col2 = st.columns(2)
                    with q_col1:
                        st.markdown(f"**Normalized Query**: `{qu.get('normalized_query')}`")
                        st.markdown(f"**Identified Intent**: `{qu.get('intent')}`")
                    with q_col2:
                        st.markdown(f"**Extracted Keywords**: `{qu.get('extracted_keywords')}`")
                        st.markdown(f"**Requested Scope**: `{qu.get('requested_scope')}`")

                # 2. Token Budget & Packing Stats
                st.subheader("Context Budget & Selection Stats")
                b_col1, b_col2, b_col3, b_col4 = st.columns(4)
                b_col1.metric("Total Tokens", data.get("total_tokens", 0))
                b_col2.metric("Token Budget", data.get("token_budget", token_budget))
                b_col3.metric("Selected Chunks", data.get("selected_chunks_count", 0))
                b_col4.metric("Skipped Chunks", data.get("skipped_chunks_count", 0))

                # 3. Source Manifest
                manifest = data.get("source_manifest", [])
                st.subheader(f"📑 Source Manifest ({len(manifest)})")
                if manifest:
                    for item in manifest:
                        lbl = item.get("citation_label")
                        name = item.get("document_name")
                        ver = item.get("version", 1)
                        p_start = item.get("page_start")
                        p_end = item.get("page_end")
                        pages_str = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}–{p_end}"
                        h_path = " > ".join(item.get("heading_path") or []) or item.get("heading") or "General"

                        with st.expander(f"**{lbl}**: *{name}* (v{ver}) — {pages_str}", expanded=False):
                            st.markdown(f"**Heading Breadcrumbs**: `{h_path}`")
                            st.markdown(f"**Document ID**: `{item.get('document_id')}`")
                            st.markdown(f"**Chunk ID**: `{item.get('chunk_id')}`")
                            st.markdown(f"**Full Reference**: `{item.get('citation_reference')}`")

                # 4. Assembled Context Text
                st.subheader("Assembled Grounded Context Window")
                st.text_area("Full Context (Sent to LLM)", value=data.get("assembled_context", ""), height=250)

                # 5. Ranked Chunks Breakdown
                ranked_chunks = data.get("ranked_chunks", [])
                st.subheader(f"Composite Reranked Chunks ({len(ranked_chunks)})")
                for c in ranked_chunks:
                    r_rank = c.get("rerank_rank", "N/A")
                    o_rank = c.get("original_rank", "N/A")
                    r_score = c.get("rerank_score", 0.0)
                    s_score = c.get("similarity_score", 0.0)
                    lex_score = c.get("lexical_score", 0.0)
                    head_score = c.get("heading_score", 0.0)
                    title = f"Rank #{r_rank} (orig #{o_rank}) | Rerank Score: {r_score:.4f} | {c.get('document_name')}"

                    with st.expander(title, expanded=False):
                        sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
                        sc_col1.metric("Rerank Score", f"{r_score:.4f}")
                        sc_col2.metric("Semantic Sim", f"{s_score:.4f}")
                        sc_col3.metric("Lexical Match", f"{lex_score:.4f}")
                        sc_col4.metric("Heading Match", f"{head_score:.4f}")

                        st.caption(c.get("score_explanation", ""))
                        st.markdown(f"**Heading**: {c.get('heading')} | **Pages**: {c.get('page_start')}–{c.get('page_end')}")
                        st.text(c.get("content", ""))

            except Exception as exc:
                st.error(f"Context inspection failed: {exc}")


# ==============================================================================
# TAB 2: Phase 6 Raw Semantic Vector Retrieval Inspector
# ==============================================================================
with tab_semantic:
    if st.button("Inspect Raw Semantic Retrieval", type="secondary"):
        with st.spinner("Querying ChromaDB vector search and MongoDB validation..."):
            try:
                res = client.inspect_retrieval(
                    question=question,
                    school_context=school_ctx,
                    employee_context=emp_ctx,
                    top_k=top_k,
                )
                data = res.get("data", {})

                st.subheader("Retrieval Summary")
                s_col1, s_col2, s_col3 = st.columns(3)
                s_col1.metric("Raw Candidates Evaluated", data.get("total_candidates", 0))
                s_col2.metric("Valid Scoped Chunks", data.get("total_results", 0))
                s_col3.metric("Status", data.get("retrieval_status", "unknown").upper())

                chunks = data.get("chunks", [])
                st.subheader(f"Retrieved Chunks ({len(chunks)})")
                for idx, c in enumerate(chunks, start=1):
                    sim = c.get("similarity_score", 0.0)
                    with st.expander(f"#{idx} — Similarity: {sim:.4f} | {c.get('document_name')}", expanded=False):
                        st.markdown(f"**Doc ID**: `{c.get('document_id')}` | **Chunk ID**: `{c.get('chunk_id')}`")
                        st.markdown(f"**Heading**: {c.get('heading')}")
                        st.markdown(f"**Pages**: {c.get('page_start')}–{c.get('page_end')}")
                        st.text(c.get("content", ""))

            except Exception as exc:
                st.error(f"Retrieval inspection failed: {exc}")
