import pdfplumber

def extract_text_from_pdf(pdf_path):
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")
    return "\n\n".join(text_parts)

from pathlib import Path

def load_all_reports(data_dir="data/reports"):
    docs = []
    pdf_files = sorted(Path(data_dir).glob("*.pdf"))

    for pdf_path in pdf_files:
        print(f"Reading {pdf_path.name} ...")
        text = extract_text_from_pdf(pdf_path)
        docs.append({"source": pdf_path.name, "text": text})

    return docs
