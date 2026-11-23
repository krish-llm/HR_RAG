"""Streamlit Main Application — HR Policy RAG Playground."""

import os
import streamlit as st
from api_client import HRPolicyAPIClient

st.set_page_config(
    page_title="HR Policy RAG Playground",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State for Tenant Scoping
if "school_id" not in st.session_state:
    st.session_state.school_id = "sch-delhi-01"
if "group_id" not in st.session_state:
    st.session_state.group_id = "grp-national-01"
if "branch_id" not in st.session_state:
    st.session_state.branch_id = ""
if "state" not in st.session_state:
    st.session_state.state = "Delhi"

# Sidebar: Global Tenant Context Scoping
st.sidebar.title("🏫 Tenant Scoping")
st.sidebar.caption("Headers applied to all retrieval & answering requests")

st.session_state.school_id = st.sidebar.text_input("School ID", value=st.session_state.school_id)
st.session_state.group_id = st.sidebar.text_input("Group/Trust ID", value=st.session_state.group_id)
st.session_state.branch_id = st.sidebar.text_input("Branch/Campus ID", value=st.session_state.branch_id)
st.session_state.state = st.sidebar.text_input("State Jurisdiction", value=st.session_state.state)

st.sidebar.divider()
backend_url = os.getenv("BACKEND_API_URL", "http://localhost:8000/api/v1")
st.sidebar.caption(f"Backend API: `{backend_url}`")

# Header & Overview
st.title("📋 HR Policy RAG Playground")
st.markdown(
    "Interactive developer playground and evaluation dashboard for testing the **HR Policy RAG Platform** end-to-end."
)

# Backend Connectivity Check
client = HRPolicyAPIClient()
try:
    health = client.check_health()
    status_str = health.get("data", {}).get("status", "unknown") if "data" in health else health.get("status", "healthy")
    st.success(f"Backend Connected: `{backend_url}` (Status: **{status_str.upper()}**)")
except Exception as exc:
    st.error(f"⚠️ Backend Disconnected: {exc}")
    st.info("Ensure the FastAPI backend is running on `http://localhost:8000` via: `uvicorn app.main:app --reload`")

st.divider()

# Navigation Cards
col1, col2 = st.columns(2)

with col1:
    st.subheader("💬 Ask HR Policy")
    st.write("Submit natural language HR questions, inspect grounded answers, and verify structured citations.")
    st.page_link("pages/1_Ask_HR_Policy.py", label="Open Ask HR Policy", icon="💬")

    st.subheader("🔍 Retrieval Inspector")
    st.write("Inspect semantic vector candidate retrieval, composite reranking, and context window assembly.")
    st.page_link("pages/3_Retrieval_Inspector.py", label="Open Retrieval Inspector", icon="🔍")

with col2:
    st.subheader("📁 Document Management")
    st.write("Upload policy PDFs, trigger the ingestion-to-indexing pipeline, and inspect chunk status.")
    st.page_link("pages/2_Document_Management.py", label="Open Document Management", icon="📁")

    st.subheader("📊 Evaluation Dashboard")
    st.write("Run the benchmark dataset and measure Precision@K, Recall@K, MRR, point coverage, and citation validity.")
    st.page_link("pages/4_Evaluation_Dashboard.py", label="Open Evaluation Dashboard", icon="📊")
