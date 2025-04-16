import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes
import pytesseract
import pandas as pd
import re
import os
import requests
import io
from datetime import datetime

st.set_page_config(page_title="CTA Analyzer", layout="wide")
st.title("📑 Clinical Trial Agreement Analyzer")
st.write("Upload one Clinical Trial Agreement at a time to receive a clause breakdown, AI summary, and risk flags.")

@st.cache_data
def extract_text_from_pdf(file):
    # First try PyPDF2 text extraction
    reader = PdfReader(file)
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    if len(text.strip()) > 100:
        return text
    else:
        # If text is too short, apply OCR
        st.warning("📷 Detected a scanned or image-based PDF. Running OCR...")
        images = convert_from_bytes(file.read())
        ocr_text = ""
        for img in images:
            ocr_text += pytesseract.image_to_string(img)
        return ocr_text

def clean_text(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]
    lines = [line for line in lines if not line.lower().startswith("whereas")]
    lines = [line for line in lines if "confidential" not in line.lower()]
    cleaned = " ".join(lines[:30])
    return cleaned if cleaned else "No usable text found."

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
def summarize_with_api(text):
    hf_token = st.secrets["HF_API_KEY"]
    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers={"Authorization": f"Bearer {hf_token}"},
        json={"inputs": text}
    )
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        raise ValueError(f"API Error {response.status_code}: {response.text}")

uploaded_file = st.file_uploader("Upload one CTA PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    # 🔍 Clause Extraction
    clauses = extract_key_clauses(text)
    df = pd.DataFrame({
        "Clause": list(clauses.keys()),
        "Extracted Info": [str(val) for val in clauses.values()]
    })
    st.subheader("🔍 Clause Summary")
    st.dataframe(df, use_container_width=True)

    # 🧠 LLM Summary
    st.subheader("🧠 LLM Summary")
    try:
        cleaned = clean_text(text)
        if cleaned == "No usable text found.":
            fallback = text.strip()[:1000]
            if not fallback:
                raise ValueError("The extracted PDF text is too short or empty.")
            summary_input = fallback
        else:
            summary_input = cleaned

        st.text_area("📝 Text sent to summarizer", summary_input, height=200)

        summary = summarize_with_api(summary_input)
        st.info(summary)

        # 📥 Download Summary
        summary_file = io.StringIO()
        summary_file.write("LLM Summary - Generated on {}\n\n".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
        summary_file.write(summary)
        st.download_button("Download Summary as TXT", summary_file.getvalue(), file_name="cta_summary.txt")

    except Exception as e:
        st.error("❌ Hugging Face API summarization failed.")
        st.exception(e)

    # 🚨 Risk Detection
    st.subheader("🚨 Risk Flags")
    risks = flag_risks(text)
    if risks:
        for r in risks:
            st.warning(r)
    else:
        st.success("✅ No major risks detected.")

    # 📥 Clause Report Download
    clause_bytes = io.BytesIO()
    df.to_excel(clause_bytes, index=False, engine="openpyxl")
    st.download_button("Download Clause Report as Excel", clause_bytes.getvalue(), file_name="cta_clauses.xlsx")

else:
    st.info("Please upload a single PDF to begin analysis.")
