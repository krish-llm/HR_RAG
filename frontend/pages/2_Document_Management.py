"""Page 2: Document Management — Upload, Multi-Stage Ingestion Pipeline, and Library."""

import streamlit as st
from api_client import HRPolicyAPIClient

st.set_page_config(page_title="Document Management", page_icon="📁", layout="wide")

st.title("📁 Document Management")
st.markdown("Upload policy PDFs, trigger the automated ingestion pipeline, and monitor document lifecycle states.")

client = HRPolicyAPIClient()

tab_upload, tab_library = st.tabs(["📤 Upload & Index Document", "📚 Document Library"])

# ==============================================================================
# TAB 1: Upload & Multi-Stage Ingestion Pipeline
# ==============================================================================
with tab_upload:
    st.subheader("Upload Policy PDF")

    uploaded_file = st.file_uploader("Select PDF Policy Document", type=["pdf"])

    col1, col2 = st.columns(2)
    with col1:
        doc_name = st.text_input("Document Name / Title", placeholder="e.g. Leave Policy 2026")
        doc_type = st.selectbox("Document Classification", options=["policy", "amendment", "guideline", "handbook"], index=0)
        version = st.number_input("Version Number", min_value=1, value=1, step=1)
        is_global = st.checkbox("Is Global Group Policy", value=False)
        school_id = st.text_input("School ID", value=st.session_state.get("school_id", "sch-delhi-01"))
        group_id = st.text_input("Group ID", value=st.session_state.get("group_id", "grp-national-01"))

    with col2:
        branch_id = st.text_input("Branch ID", value=st.session_state.get("branch_id", ""))
        state = st.text_input("State Jurisdiction", value=st.session_state.get("state", "Delhi"))
        staff_category = st.selectbox("Target Staff Category", options=["", "teaching", "non_teaching", "administrative"])
        employment_type = st.selectbox("Employment Type", options=["", "full_time", "part_time", "contract"])
        department = st.text_input("Department Applicability", value="")

    if st.button("Start Ingestion Pipeline", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a PDF file first.")
        else:
            file_bytes = uploaded_file.read()
            metadata = {
                "document_name": doc_name.strip() if doc_name else uploaded_file.name,
                "document_type": doc_type,
                "version": version,
                "is_global": is_global,
                "school_id": school_id.strip() if school_id else None,
                "group_id": group_id.strip() if group_id else None,
                "branch_id": branch_id.strip() if branch_id else None,
                "state": state.strip() if state else None,
                "staff_category": staff_category.strip() if staff_category else None,
                "employment_type": employment_type.strip() if employment_type else None,
                "department": department.strip() if department else None,
            }

            st.write("---")
            progress_bar = st.progress(0, text="Starting pipeline...")

            try:
                # Stage 1: Upload & Register
                progress_bar.progress(20, text="Stage 1/5: Uploading PDF and registering document metadata...")
                upload_res = client.upload_document(file_bytes, uploaded_file.name, metadata)
                doc_data = upload_res.get("data", {})
                document_id = doc_data.get("document_id")
                st.success(f"Stage 1 Complete: Document Registered (`ID: {document_id}`)")

                # Stage 2: Extract Content
                progress_bar.progress(40, text="Stage 2/5: Extracting Markdown pages via PyMuPDF4LLM...")
                extract_res = client.extract_document(document_id)
                pages = extract_res.get("data", {}).get("total_pages", "N/A")
                st.success(f"Stage 2 Complete: Extracted {pages} pages")

                # Stage 3: Clean & Process
                progress_bar.progress(60, text="Stage 3/5: Cleaning text, normalizing, stripping repeated headers...")
                proc_res = client.process_document(document_id)
                sections = proc_res.get("data", {}).get("total_sections", "N/A")
                st.success(f"Stage 3 Complete: Processed {sections} structural sections")

                # Stage 4: Semantic Chunking
                progress_bar.progress(80, text="Stage 4/5: Generating heading-aware chunks with tiktoken...")
                chunk_res = client.chunk_document(document_id)
                total_chunks = chunk_res.get("data", {}).get("total_chunks", "N/A")
                st.success(f"Stage 4 Complete: Created {total_chunks} semantic chunks")

                # Stage 5: Embed & Vector Index
                progress_bar.progress(95, text="Stage 5/5: Generating embeddings and vector indexing in ChromaDB...")
                index_res = client.index_document(document_id)
                gen_id = index_res.get("data", {}).get("generation", 1)
                progress_bar.progress(100, text="Pipeline Completed Successfully!")
                st.balloons()
                st.success(f"Stage 5 Complete: Chunks indexed into ChromaDB (Active Generation: `g{gen_id}`)")

            except Exception as exc:
                progress_bar.empty()
                st.error(f"Pipeline Failed: {exc}")


# ==============================================================================
# TAB 2: Document Library & Status Monitoring
# ==============================================================================
with tab_library:
    st.subheader("Ingested Policy Documents")

    c_refresh, _ = st.columns([1, 4])
    with c_refresh:
        refresh = st.button("🔄 Refresh Library", use_container_width=True)

    try:
        docs_res = client.list_documents(page=1, page_size=50)
        doc_list = docs_res.get("data", {}).get("items", [])

        if not doc_list:
            st.info("No policy documents found in the database.")
        else:
            st.write(f"Total Registered Documents: **{len(doc_list)}**")

            for doc in doc_list:
                doc_id = doc.get("document_id")
                name = doc.get("document_name") or "Untitled Policy"
                ver = doc.get("version", 1)
                doc_status = doc.get("status", "unknown")
                proc_status = doc.get("processing_status", "unknown")
                pages = doc.get("total_pages", 0)

                with st.expander(f"📄 **{name}** (v{ver}) — `{doc_id}` | Status: `{doc_status}`", expanded=False):
                    d_col1, d_col2 = st.columns(2)
                    with d_col1:
                        st.markdown(f"**Document Type**: `{doc.get('document_type')}`")
                        st.markdown(f"**Processing Status**: `{proc_status}`")
                        st.markdown(f"**Pages**: `{pages}`")
                        st.markdown(f"**Created At**: `{doc.get('created_at')}`")

                    with d_col2:
                        st.markdown(f"**School ID**: `{doc.get('school_id')}`")
                        st.markdown(f"**Group ID**: `{doc.get('group_id')}`")
                        st.markdown(f"**Branch ID**: `{doc.get('branch_id')}`")
                        st.markdown(f"**Global Scope**: `{doc.get('is_global')}`")

                    btn_col1, btn_col2 = st.columns([1, 1])
                    with btn_col1:
                        if st.button("Check Indexing Status", key=f"status_{doc_id}"):
                            try:
                                status_res = client.get_indexing_status(doc_id)
                                st.json(status_res.get("data", {}))
                            except Exception as e:
                                st.error(f"Status check failed: {e}")
                    with btn_col2:
                        if st.button("Delete Document", key=f"del_{doc_id}", type="secondary"):
                            try:
                                client.delete_document(doc_id)
                                st.success(f"Document `{doc_id}` deleted. Please refresh.")
                            except Exception as e:
                                st.error(f"Delete failed: {e}")

    except Exception as exc:
        st.error(f"Failed to fetch documents: {exc}")
