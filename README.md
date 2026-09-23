# AI Document Intelligence & Workflow Platform

ZYROO AI/ML Internship, **Week 3: Improve Document Understanding**

A Streamlit app that reads documents (PDF, JPG, JPEG, PNG), cleans the text, classifies the document type (Invoice, Resume, Other), extracts key fields, and shows the result with a confidence score where one is available.

**Pipeline:** Upload → Read Text/OCR → Clean Text → Identify Type → Extract Fields → Check Missing Fields → Show Result

---

## What Changed from Week 2

| Area | Week 2 | Week 3 |
|---|---|---|
| Text cleaning | Basic text only | Removes extra spaces and blank lines, normalizes text, handles empty or very short text |
| OCR | Basic OCR | Grayscale, upscaling, auto-contrast and noise reduction. Two OCR versions are tried and the one with more text is kept |
| Classification | Keyword rules | TF-IDF + ML classifier. The rule-based approach is kept as a baseline |
| Model comparison | None | Logistic Regression vs Linear SVM vs Naive Bayes on the same data and the same cross-validation folds |
| Evaluation | None | Accuracy, precision, recall, F1-score and confusion matrix (saved to `evaluation_report.txt`) |
| Extraction | Simple regex | Improved rules (see below) |
| Missing fields | Could fail | Shows `Not Found` and a warning listing the missing fields. One bad file never crashes the app |
| Confidence | None | Shown only when the ML model decided and can give probabilities. Never invented |
| UI | Basic | Card layout per document, document-type badge, confidence bar |

### Extraction improvements
- **Invoice Number:** must contain a digit, so the word "Date" is never returned as a number.
- **Date:** looks for the date next to a "Date" label first, then falls back to any date format.
- **Total Amount:** ignores "Subtotal", supports currency symbols (PKR, Rs, $, etc.) and decimals.
- **Name (resume):** skips section headings and degree names (e.g. "BS Computer Science") and picks the first valid 2-4 word name line.
- **Phone:** supports Pakistani mobile formats (`0300-1234567`, `+92-321-7654321`) and general international numbers.
- **Skills:** reads the full skills section until the next heading, not only one line.

### Bug fixes
- Replaced `uploaded_file.read()` with `getvalue()` so files are read correctly when Streamlit reruns.
- Replaced the deprecated `fitz` import with `pymupdf`.
- Fixed the empty white box in the UI by using `st.container(border=True)`.

---

## Project Structure

```
├── app.py                  # Streamlit app (upload, OCR, classify, extract, display)
├── train_model.py          # Trains and compares models, writes the evaluation report
├── document_dataset.csv    # Training data (text,label)
├── evaluation_report.txt   # Generated: metrics and confusion matrices
├── requirements.txt
└── README.md
```

## Setup and Run

1. Install Python packages:
   ```
   pip install -r requirements.txt
   ```
2. Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki). On Windows, if it is not on PATH, set `pytesseract.pytesseract.tesseract_cmd` in `app.py`.
3. Train and compare the models (creates `doc_classifier.joblib` and `evaluation_report.txt`):
   ```
   python train_model.py
   ```
4. Start the app:
   ```
   python -m streamlit run app.py
   ```

---

## Classification Model

- **Features:** TF-IDF (unigrams and bigrams) on cleaned text.
- **Models compared:** Logistic Regression, Linear SVM, Naive Bayes, plus the Week 2 rule-based baseline.
- **Fair comparison:** the same dataset and the same stratified k-fold cross-validation for every model.
- **Selection:** the model with the highest macro F1-score is saved and used by the app.
- **Confidence:** shown only when the selected model provides probabilities (e.g. Logistic Regression). If a keyword rule overrides the ML prediction, the app says so and shows no confidence number.

### Evaluation Results

> Fill this in from `evaluation_report.txt` after running `python train_model.py`.

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| Rule-based baseline | | | | |
| Logistic Regression | | | | |
| Linear SVM | | | | |
| Naive Bayes | | | | |

**Selected model:** _(name)_

**Confusion matrix (selected model):**

```
(paste from evaluation_report.txt)
```

**Evaluation note:** _(what the model does well, and which classes it confuses)_

---

## Testing

> Fill this in after testing. Include invoices, resumes, scanned/image documents, and documents with missing fields.

| File | Expected type | Predicted type | Confidence | Missing / wrong fields |
|---|---|---|---|---|
| | | | | |

### Missing-field handling
When a field cannot be found, the result shows `Not Found` and a warning lists the missing fields. The application keeps running. Screenshot: _(add `screenshots/missing_fields.png`)_

### Screenshots
_(add screenshots of the upload page, an invoice result, a resume result, a scanned document, and a missing-field case)_

---

## Known Limitations
- The dataset is small, so accuracy on very different real-world documents may vary. Adding more real examples improves results.
- Extraction is regex/keyword based and may fail on unusual layouts.
- OCR quality depends on scan quality.

## Tech Stack
Python · Streamlit · scikit-learn · PyMuPDF · Tesseract OCR (pytesseract) · Pandas · NumPy · Regular Expressions

## Next Phase
Saving, searching, and organizing the processed documents.
