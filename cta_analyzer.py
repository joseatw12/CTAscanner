import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import pandas as pd
import re
import requests
import io
from datetime import datetime

st.set_page_config(page_title="CTA Analyzer", layout="wide")
st.title("📑 Clinical Trial Agreement Analyzer")
st.write("Upload one Clinical Trial Agreement at a time to receive a clause breakdown, AI summary, and risk flags.")

@st.cache_data
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    raw_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if len(raw_text.strip()) > 100:
        return raw_text
    st.warning("📷 Detected a scanned or image-based PDF. Running OCR...")
    images = convert_from_bytes(file.read())
    return "".join(pytesseract.image_to_string(img) for img in images)

def clean_text(text):
    clean_lines = []
    for line in text.splitlines():
        s = line.strip()
        if len(s) <= 10 or s.isdigit():
            continue
        lower = s.lower()
        if lower.startswith("whereas") or "confidential" in lower:
            continue
        clean_lines.append(s)
    return " ".join(clean_lines[:30]) or "No usable text found."

@st.cache_data
def extract_key_clauses(text):
    head = text[:2000]
    return {
        "Sponsor": re.findall(r"Sponsor.*", head),
        "Institution": re.findall(r"Institution.*", head),
        "Investigator": re.findall(r"Investigator.*", head),
        "Budget Amounts": re.findall(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?", text),
        "Includes Indemnification": "Indemnification" in text,
        "Includes Injury Clause": "Trial Participant Injury" in text,
        "Termination Rights": "terminate" in text.lower()
    }

def flag_risks(text_lower):
    risks = []
    if "termination for convenience" in text_lower:
        risks.append("⚠️ Termination may favor sponsor")
    if "sole discretion" in text_lower:
        risks.append("⚠️ One-sided decision-making clause")
    if "no obligation to pay" in text_lower:
        risks.append("⚠️ Payment obligation is unclear")
    return risks

# Compile milestone regexes once
MILESTONE_PATTERNS = {
    "First Subject In (FSI)": r"\b(first subject in|fsi)\b",
    "Last Subject Out (LSO)": r"\b(last subject out|lso)\b",
    "Enrollment Completion": r"\benrollment completion\b",
    "Study Closeout": r"\b(study close[- ]?out|close[- ]?out visit)\b",
    "IRB Approval": r"\birb (approval|submission)\b",
    "Site Activation": r"\b(site activation|activation date)\b"
}
COMPILED_MILESTONES = {
    label: re.compile(pat, re.IGNORECASE)
    for label, pat in MILESTONE_PATTERNS.items()
}

def extract_milestones(text):
    rows = []
    for label, regex in COMPILED_MILESTONES.items():
        m = regex.search(text)
        rows.append({
            "Milestone": label,
            "Mentioned?": "✅ Yes" if m else "❌ No",
            "Example": m.group(0) if m else ""
        })
    return pd.DataFrame(rows)

@st.cache_data
def summarize_with_api(text):
    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers={"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"},
        json={"inputs": text}
    )
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    raise ValueError(f"API Error {response.status_code}: {response.text}")

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload one CTA PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    text_lower = text.lower()

    # 🔍 Clause Summary
    st.subheader("🔍 Clause Summary")
    clauses = extract_key_clauses(text)
    clause_df = pd.DataFrame({
        "Clause": list(clauses.keys()),
        "Extracted Info": [str(v) for v in clauses.values()]
    })
    st.dataframe(clause_df, use_container_width=True)

    # 📆 Milestone Tracker
    st.subheader("📆 Clinical Trial Milestones")
    milestone_df = extract_milestones(text)
    st.dataframe(milestone_df, use_container_width=True)

    # 🧠 LLM Summary
    st.subheader("🧠 LLM Summary")
    try:
        cleaned = clean_text(text)
        summary_input = cleaned if cleaned != "No usable text found." else text.strip()[:1000]
        st.text_area("📝 Text sent to summarizer", summary_input, height=200)
        summary = summarize_with_api(summary_input)
        st.info(summary)

        summary_txt = f"LLM Summary - {datetime.now():%Y-%m-%d %H:%M}\n\n{summary}"
        st.download_button("Download Summary as TXT", summary_txt, file_name="cta_summary.txt")
    except Exception as e:
        st.error("❌ Hugging Face API summarization failed.")
        st.exception(e)

    # 🚨 Risk Flags
    st.subheader("🚨 Risk Flags")
    risks = flag_risks(text_lower)
    if risks:
        for r in risks:
            st.warning(r)
    else:
        st.success("✅ No major risks detected.")

    # 📥 Clause Report Download
    clause_bytes = io.BytesIO()
    clause_df.to_excel(clause_bytes, index=False, engine="openpyxl")
    st.download_button("Download Clause Report as Excel", clause_bytes.getvalue(), file_name="cta_clauses.xlsx")

else:
    st.info("Please upload a single PDF to begin analysis.")
