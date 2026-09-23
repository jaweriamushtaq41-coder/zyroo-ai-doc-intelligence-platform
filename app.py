"""
ZYROO AI/ML Internship - Week 3: AI Document Intelligence (Improved MVP)
Pipeline: Upload -> Read Text/OCR -> Clean -> Classify -> Extract -> Check Missing -> Show
Run: python -m streamlit run app.py   (run `python train_model.py` first)
"""
import io
import re
import joblib
import pandas as pd
import streamlit as st
import pymupdf  # PyMuPDF (replaces deprecated `fitz` import)
from PIL import Image, ImageOps, ImageFilter
import pytesseract

from train_model import MODEL_PATH, clean_text, rule_based_label, train_and_save

# If Tesseract is not on PATH (Windows), uncomment and fix the path:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ALLOWED_TYPES = ["pdf", "jpg", "jpeg", "png"]
NOT_FOUND = "Not Found"

st.set_page_config(page_title="AI Document Intelligence - Week 3", page_icon="📄", layout="wide")

DOC_TYPE_STYLE = {
    "Invoice": {"color": "#2563EB", "bg": "#EFF6FF", "icon": "🧾"},
    "Resume": {"color": "#16A34A", "bg": "#F0FDF4", "icon": "👤"},
    "Other": {"color": "#6B7280", "bg": "#F3F4F6", "icon": "📄"},
}

CUSTOM_CSS = """
<style>
.hero {background: linear-gradient(135deg,#4F46E5,#7C3AED); padding: 2rem 2.2rem;
       border-radius: 16px; color: white; margin-bottom: 1.5rem;}
.hero h1 {margin: 0; font-size: 2rem; font-weight: 800;}
.hero p {margin-top: .4rem; opacity: .9;}
.type-badge {display: inline-flex; gap: .4rem; padding: .35rem .9rem;
             border-radius: 999px; font-weight: 700;}
.small-note {color: #6B7280; font-size: .85rem; margin-top: .3rem;}
section[data-testid="stFileUploaderDropzone"] {border-radius: 14px;
       border: 2px dashed #C7D2FE; background: #F5F7FF;}
</style>
"""


# ---------------- OCR / text extraction (Step 3) ----------------
def preprocess_image_for_ocr(img: Image.Image) -> Image.Image:
    """Grayscale + upscale + auto-contrast + light noise reduction."""
    img = ImageOps.exif_transpose(img).convert("L")
    w, h = img.size
    if max(w, h) < 1800:
        s = 1800 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
    img = ImageOps.autocontrast(img)
    return img.filter(ImageFilter.MedianFilter(3))


def ocr_image(img: Image.Image) -> str:
    """OCR two versions (grayscale, thresholded) and keep the one with more text."""
    base = preprocess_image_for_ocr(img)
    binary = base.point(lambda p: 255 if p > 150 else 0)
    results = [pytesseract.image_to_string(x) for x in (base, binary)]
    return max(results, key=lambda t: sum(c.isalnum() for c in t))


def extract_text_from_pdf(file_bytes: bytes) -> str:
    chunks = []
    with pymupdf.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text = page.get_text().strip()
            if len(text) < 20:  # probably a scanned page -> OCR
                pix = page.get_pixmap(dpi=250)
                text = ocr_image(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            chunks.append(text)
    return "\n".join(chunks)


def extract_text(uploaded_file) -> str:
    ext = uploaded_file.name.split(".")[-1].lower()
    data = uploaded_file.getvalue()  # safe on Streamlit reruns (unlike .read())
    if ext == "pdf":
        return extract_text_from_pdf(data)
    if ext in ("jpg", "jpeg", "png"):
        return ocr_image(Image.open(io.BytesIO(data)))
    return ""


# ---------------- Classification (Step 4-6, 9) ----------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return train_and_save()[0]


def classify_document(text: str, model):
    """Returns (label, confidence_or_None, note). Confidence is shown only
    when the ML model itself made the decision and can give probabilities."""
    ml_pred = model.predict([text])[0]
    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = round(float(model.predict_proba([text])[0].max()) * 100, 1)

    rule = rule_based_label(text)
    if rule != "Other" and rule != ml_pred:
        return rule, None, f"Rule-based override (ML model predicted {ml_pred})"
    return ml_pred, confidence, ""


# ---------------- Information extraction (Step 7-8) ----------------
def find(patterns, text, group=1, flags=re.IGNORECASE):
    """Try patterns in order; return first match or 'Not Found'."""
    for p in patterns:
        m = re.search(p, text, flags)
        if m and m.group(group).strip():
            return m.group(group).strip()
    return NOT_FOUND


DATE = r"(\d{1,2}[-/. ]\d{1,2}[-/. ]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4})"
AMOUNT = r"((?:PKR|Rs\.?|USD|\$|€|£)?\s?\d[\d,]*(?:\.\d{1,2})?)"


def extract_invoice_fields(text: str) -> dict:
    return {
        # (?=[A-Z0-9-]*\d) forces at least one digit, so "Invoice Date" is never the number
        "Invoice Number": find([r"invoice\s*(?:no\.?|number|#|id|ref(?:erence)?)?\s*[:#\-]?\s*((?=[A-Z0-9\-]*\d)[A-Z0-9\-]{3,})"], text),
        "Date": find([r"(?:invoice\s*)?date(?:\s*issued)?\s*[:\-]?\s*" + DATE, DATE], text),
        "Company": find([r"(?:company(?:\s*name)?|bill\s*from|from|vendor)\s*[:\-]\s*([^\n:]{2,50})"], text),
        # (?<!sub) so "Subtotal" is not mistaken for "Total"
        "Total Amount": find([r"(?<!sub)(?:total(?:\s+(?:amount|payable|due))?|amount\s+due|grand\s+total)\s*[:\-]?\s*" + AMOUNT], text),
    }


NAME_BLOCKLIST = {
    "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective", "education",
    "experience", "skills", "projects", "contact", "bs", "bsc", "bscs", "ms", "msc", "bba",
    "bachelor", "bachelors", "master", "masters", "computer", "science", "engineering",
    "university", "college", "institute", "software", "developer", "engineer", "intern",
    "data", "phone", "email", "address", "web", "designer", "analyst",
}


def extract_name(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:12]:
        if re.search(r"[@\d/:]", line):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z.'-]*", line)
        if not 2 <= len(words) <= 4 or len(words) != len(line.split()):
            continue
        if any(w.lower().strip(".") in NAME_BLOCKLIST for w in words):
            continue
        if all(w[0].isupper() for w in words):
            return " ".join(w.title() if w.isupper() else w for w in words)
    return NOT_FOUND


SECTION_END = r"(?:education|work experience|experience|projects?|certifications?|references|languages|interests|objective|summary|profile)"


def extract_skills(text: str) -> str:
    m = re.search(r"skills?\s*[:\-]?\s*(.+?)(?=\b" + SECTION_END + r"\b\s*[:\n]|\Z)", text, re.I | re.S)
    if not m:
        return NOT_FOUND
    skills = re.sub(r"\s*[\n•▪●·]+\s*", ", ", m.group(1).strip(" ,-:•"))
    return skills[:250] if skills else NOT_FOUND


def extract_resume_fields(text: str) -> dict:
    return {
        "Name": extract_name(text),
        "Email": find([r"([\w.+-]+@[\w-]+(?:\.[\w-]+)+)"], text),
        "Phone": find([r"((?:\+92|0092|0)[\s-]?\d{2,3}[\s-]?\d{7})", r"(\+\d{1,3}[\s-]?\d{2,4}[\s-]?\d{6,8})"], text),
        "Skills": extract_skills(text),
    }


def extract_fields(doc_type: str, text: str) -> dict:
    if doc_type == "Invoice":
        return extract_invoice_fields(text)
    if doc_type == "Resume":
        return extract_resume_fields(text)
    return {}


# ---------------- UI ----------------
def render_doc_card(name, doc_type, confidence, note, fields, cleaned):
    style = DOC_TYPE_STYLE.get(doc_type, DOC_TYPE_STYLE["Other"])
    with st.container(border=True):  # real container: no more empty white box
        st.subheader(f"📁 {name}")
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            st.markdown(
                f'<span class="type-badge" style="color:{style["color"]};background:{style["bg"]};">'
                f'{style["icon"]} {doc_type}</span>', unsafe_allow_html=True)
            if confidence is not None:
                st.markdown(f'<div class="small-note">Confidence: {confidence}%</div>', unsafe_allow_html=True)
                st.progress(min(int(confidence), 100))
            else:
                st.markdown('<div class="small-note">Confidence: not available</div>', unsafe_allow_html=True)
            if note:
                st.caption(note)
        with col2:
            if fields:
                st.write("**Extracted Fields**")
                st.dataframe(pd.DataFrame(fields.items(), columns=["Field", "Result"]),
                             hide_index=True, use_container_width=True)
                missing = [k for k, v in fields.items() if v == NOT_FOUND]
                if missing:
                    st.warning("Missing fields: " + ", ".join(missing))
            else:
                st.info("No structured fields defined for this document type.")
        with st.expander("🔍 View cleaned extracted text"):
            st.text(cleaned)


def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>📄 AI Document Intelligence & Workflow Platform</h1>'
        "<p>Week 3 — Improved MVP with cleaner text, ML classification, and smarter extraction</p></div>",
        unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ About this app")
        st.write("Upload invoices, resumes, or other documents. The app reads, cleans, "
                 "classifies, and extracts key fields automatically.")
        st.markdown("**Supported types**")
        st.write("PDF · JPG · JPEG · PNG")
        st.markdown("**Pipeline**")
        st.write("Upload → OCR/Text → Clean → Classify → Extract → Result")

    try:
        model = load_model()
    except FileNotFoundError as e:
        st.error(f"⚠️ {e}. Put `document_dataset.csv` next to app.py, then run `python train_model.py`.")
        st.stop()

    files = st.file_uploader("Upload documents", type=ALLOWED_TYPES, accept_multiple_files=True)
    if not files:
        st.info("👆 Upload one or more documents to get started.")
        return

    for f in files:
        try:  # one bad file must never break the whole app
            with st.spinner(f"Processing {f.name}..."):
                cleaned = clean_text(extract_text(f))
                if not cleaned:
                    st.warning(f"⚠️ **{f.name}** — could not extract readable text.")
                    continue
                doc_type, conf, note = classify_document(cleaned, model)
                fields = extract_fields(doc_type, cleaned)
            render_doc_card(f.name, doc_type, conf, note, fields, cleaned)
        except Exception as e:
            st.error(f"❌ Could not process **{f.name}**: {e}")


if __name__ == "__main__":
    main()