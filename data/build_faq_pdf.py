"""Generate data/Company_FAQ.pdf from data/faqs.json.

The PDF is written in a simple "Q: ... A: ..." format that retrieval/index_faq.py
parses with PyPDF. Run once before indexing:

    python data/build_faq_pdf.py
"""
from __future__ import annotations

import json
from pathlib import Path

from streamlit import pdf
from tomlkit import item

from backend import config

DATA_DIR = config.DATA_DIR


def build_pdf(json_path: Path | None = None, out_path: Path | None = None) -> Path:
    json_path = json_path or (DATA_DIR / "faqs.json")
    out_path = out_path or (DATA_DIR / "Company_FAQ.pdf")

    faqs = json.loads(json_path.read_text(encoding="utf-8"))

    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Company FAQ", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    for item in faqs:
        pdf.set_font("Helvetica", "B", 12)
        pdf.multi_cell(0, 8, text=f"Q: {item['question']}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, text=f"A: {item['answer']}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(out_path))
    return out_path


if __name__ == "__main__":  # pragma: no cover
    path = build_pdf()
    print(f"Wrote {path}")