import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import re
import os
from transformers import pipeline

st.set_page_config(page_title="CTA Analyzer", layout="wide")
st.title("📑 Clinical Trial Agreement Analyzer")
st.write("Upload one Clinical Trial Agreement at a time to receive a clause breakdown, AI summary, and risk flags.")

@st.cache_data
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join([page.extract_text() or "" for page in reader.pages])

def extract_key_clauses(text):
    return {
        "Sponsor": re.findall(r"Sponsor.*", text[:2000]),
        "Institution": re.findall(r"Institution.*", text[:2000]),
        "Investigator": re.findall(r"Investigator.*", text[:2000]),
        "Budget Amounts": re.findall(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?", text),
        "Includes Indemnification": "Indemnification" in text,
        "Includes Injury Clause": "Trial Participant Injury" in text,
        "Termination Rights": "terminate" in text.lower()
    }

def flag_risks(text):
    flags = []
    if "termination for convenience" in text.lower():
        flags.append("⚠️ Termination may favor sponsor")
    if "sole discretion" in text.lower():
        flags.append("⚠️ One-sided decision-making clause")
    if "no obligation to pay" in text.lower():
        flags.append("⚠️ Payment obligation is unclear")
    return flags

@st.cache_resource
def get_summarizer():
    hf_token = os.getenv("HF_API_KEY")  # Ensure this is set in Streamlit Secrets
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", use_auth_token=hf_token)

uploaded_file = st.file_uploader("Upload one CTA PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    clauses = extract_key_clauses(text)
    df = pd.DataFrame({
        "Clause": list(clauses.keys()),
        "Extracted Info": [str(val) for val in clauses.values()]
    })
    st.subheader("🔍 Clause Summary")
    st.dataframe(df, use_container_width=True)

    st.subheader("🧠 LLM Summary")
    summarizer = get_summarizer()
    sample = text[:1000]
    summary = summarizer(sample, max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
    st.info(summary)

    st.subheader("🚨 Risk Flags")
    risks = flag_risks(text)
    if risks:
        for r in risks:
            st.warning(r)
    else:
        st.success("✅ No major risks detected.")
else:
    st.info("Please upload a single PDF to begin analysis.")
