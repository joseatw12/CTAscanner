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
    text = "\n".join([page.extract_text() or "" for page in reader.pages])
    if len(text.strip()) > 100:
        return text
    else:
        st.warning("📷 Detected a scanned or image-based PDF. Running OCR...")
        images = convert_from_bytes(file.read())
        return "".join([pytesseract.image_to_string(img) for img in images])

def clean_text(text):
    lines = text.splitlines()
    lines = [line.strip() for line in lines if len(line.strip()) > 10 and not line.strip().isdigit()]
    lines = [line for line in lines if not line.lower().startswith("whereas")]
    lines = [line for line in lines if "confidential" not in line.lower()]
    return " ".join(lines[:30]) or "No usable text found."

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
    risks = []
    if "termination for convenience" in text.lower():
        risks.append("⚠️ Termination may favor sponsor")
    if "sole discretion" in text.lower():
        risks.append("⚠️ One-sided decision-making clause")
    if "no obligation to pay" in text.lower():
        risks.append("⚠️ Payment obligation is unclear")
    return risks

@st.cache_resource
def summarize_with_api(text):
    response = requests.post(
        "https://api-inference.huggingface.co/models/facebook/bart-large-cnn",
        headers={"Authorization": f"Bearer {st.secrets['HF_API_KEY']}"},
        json={"inputs": text}
    )
    if response.status_code == 200:
        return response.json()[0]["summary_text"]
    else:
        raise ValueError(f"API Error {response.status_code}: {response.text}")

# --- Milestone Tracker ---
MILESTONE_PATTERNS = {
    "First Subject In (FSI)": r"\b(first subject in|fsi)\b",
    "Last Subject Out (LSO)": r"\b(last subject out|lso)\b",
    "Enrollment Completion": r"\benrollment completion\b",
    "Study Closeout": r"\b(study close[- ]?out|close[- ]?out visit)\b",
    "IRB Approval": r"\birb (approval|submission)\b",
    "Site Activation": r"\b(site activation|activation date)\b"
}

def extract_milestones(text):
    results = []
    for label, pattern in MILESTONE_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        results.append({
            "Milestone": label,
            "Mentioned?": "✅ Yes" if match else "❌ No",
            "Example": match.group(0) if match else ""
        })
    return pd.DataFrame(results)

# --- Payment Term Visualizer ---
PAYMENT_PATTERNS = {
    "Startup Fee": r"(startup|start-up) fee.*?\$\d[\d,]*(?:\.\d{2})?",
    "Per Visit Payment": r"\$\d[\d,]*(?:\.\d{2})?.*?(per visit|per subject)",
    "Closeout Payment": r"(closeout|close-out) fee.*?\$\d[\d,]*(?:\.\d{2})?",
    "IRB Fee": r"irb (submission|review).*?\$\d[\d,]*(?:\.\d{2})?",
    "Overhead / Admin Fee": r"(overhead|administrative).*?(fee|charge).*?\d{1,2} ?%"
}

def extract_payment_terms(text):
    results = []
    for label, pattern in PAYMENT_PATTERNS.items():
        matches = re.findall(pattern, text, re.IGNORECASE)
        results.append({
            "Payment Type": label,
            "Mentioned?": "✅ Yes" if matches else "❌ No",
            "Examples": "; ".join([match[0] if isinstance(match, tuple) else match for match in matches[:2]])
        })
    return pd.DataFrame(results)

def visualize_payments(df):
    chart_df = df[df["Mentioned?"] == "✅ Yes"].copy()
    chart_df["Amount (est.)"] = 1000  # Placeholder for now
    st.subheader("💰 Estimated Payment Structure (Visualized)")
    st.bar_chart(chart_df.set_index("Payment Type")["Amount (est.)"])

# --- Main App Logic ---
uploaded_file = st.file_uploader("Upload one CTA PDF", type=["pdf"])

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    # 🔍 Clause Summary
    st.subheader("🔍 Clause Summary")
    clauses = extract_key_clauses(text)
    df = pd.DataFrame({"Clause": list(clauses.keys()), "Extracted Info": [str(v) for v in clauses.values()]})
    st.dataframe(df, use_container_width=True)

    # 📆 Milestone Tracker
    st.subheader("📆 Clinical Trial Milestones")
    milestone_df = extract_milestones(text)
    st.dataframe(milestone_df, use_container_width=True)

    # 💵 Payment Term Analyzer
    st.subheader("💵 Payment Terms Summary")
    payment_df = extract_payment_terms(text)
    st.dataframe(payment_df, use_container_width=True)

    if not payment_df[payment_df["Mentioned?"] == "✅ Yes"].empty:
        visualize_payments(payment_df)
    else:
        st.info("No payment terms found to visualize.")

    # 🧠 LLM Summary
    st.subheader("🧠 LLM Summary")
    try:
        cleaned = clean_text(text)
        summary_input = cleaned if cleaned != "No usable text found." else text.strip()[:1000]
        st.text_area("📝 Text sent to summarizer", summary_input, height=200)
        summary = summarize_with_api(summary_input)
        st.info(summary)

        summary_file = io.StringIO()
        summary_file.write(f"LLM Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{summary}")
        st.download_button("Download Summary as TXT", summary_file.getvalue(), file_name="cta_summary.txt")
    except Exception as e:
        st.error("❌ Hugging Face API summarization failed.")
        st.exception(e)

    # 🚨 Risk Flags
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
