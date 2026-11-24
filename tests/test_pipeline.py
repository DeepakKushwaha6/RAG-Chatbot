"""End-to-end pipeline check.

Creates sample documents, indexes them, asks questions and prints the answer
plus the retrieved sources so correctness can be verified by eye.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_sections  # noqa: E402
from app.engine import RAGEngine  # noqa: E402
from app.loaders import load_document  # noqa: E402
from app.models import Document, new_doc_id  # noqa: E402
from app.store import Store  # noqa: E402

SAMPLES = ROOT / "samples"
WORK = ROOT / "data" / "_test"


def build_samples() -> list[Path]:
    SAMPLES.mkdir(parents=True, exist_ok=True)

    txt = SAMPLES / "acme_policy.txt"
    txt.write_text(
        "Acme Corporation Travel and Expense Policy\n"
        "Version 4.2, effective 1 March 2025.\n\n"
        "1. Reimbursement Limits\n"
        "Employees may claim up to 750 USD for hotel accommodation per night in standard cities, "
        "and up to 1200 USD per night in Tier 1 cities such as New York, London and Singapore. "
        "Any amount above these limits requires written approval from the Finance Director.\n"
        "Meal expenses are reimbursed up to 60 USD per day. Alcohol is not reimbursable except "
        "at client entertainment events, where the cap is 90 USD per head.\n\n"
        "2. Flight Booking\n"
        "All flights longer than 6 hours must be booked in Economy Premium class. Flight tickets "
        "must be booked at least 14 days in advance to qualify for full reimbursement. "
        "Flexible or refundable tickets are only reimbursed for flights cancelled due to illness "
        "or company emergency.\n\n"
        "3. Approval Process\n"
        "Every expense claim above 500 USD must be approved by the employee's line manager before "
        "submission to Accounts Payable. Claims above 5000 USD additionally require approval from a "
        "Director. Submission deadlines are 30 days after the expense date; claims older than 90 days "
        "are automatically rejected.\n\n"
        "4. Contact\n"
        "Questions about this policy should be directed to the Travel Desk at travel@acme.example "
        "or extension 4412. The Travel Desk operates Monday to Friday, 09:00 to 18:00 IST.\n",
        encoding="utf-8",
    )

    csv = SAMPLES / "acme_sales_2025.csv"
    csv.write_text(
        "quarter,region,revenue_usd,target_usd,growth_pct\n"
        "Q1,North America,1250000,1200000,4.2\n"
        "Q1,Europe,980000,1100000,-10.9\n"
        "Q1,Asia Pacific,1430000,1300000,10.0\n"
        "Q2,North America,1380000,1300000,6.2\n"
        "Q2,Europe,1120000,1150000,-2.6\n"
        "Q2,Asia Pacific,1595000,1400000,13.9\n"
        "Q3,North America,1520000,1450000,4.8\n"
        "Q3,Europe,1275000,1200000,6.3\n"
        "Q3,Asia Pacific,1760000,1500000,17.3\n",
        encoding="utf-8",
    )

    docx_path = SAMPLES / "acme_handbook.docx"
    try:
        import docx

        document = docx.Document()
        document.add_heading("Acme Corporation Employee Handbook", level=1)
        document.add_paragraph("Section 4 - Leave Policy. Approved 12 January 2025.")
        document.add_paragraph(
            "Annual leave accrues at 18 days per year for employees with less than 3 years of "
            "service, and increases to 24 days after 3 years and to 30 days after 8 years."
        )
        document.add_paragraph(
            "Sick leave is a separate entitlement of 12 days per year and cannot be carried over "
            "into the following calendar year."
        )
        document.add_heading("Probation and Notice", level=2)
        document.add_paragraph(
            "The probation period is 6 months. During probation either party may terminate the "
            "employment with 15 days written notice. After probation the notice period is 60 days "
            "for the employee and 90 days for the employer."
        )
        document.save(docx_path)
    except Exception as exc:  # noqa: BLE001
        print(f"! docx sample skipped: {exc}")

    pdf_path = SAMPLES / "acme_security.pdf"
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4
        page_one = [
            "Acme Corporation Information Security Standard",
            "Version 2.1, issued 8 April 2025.",
            "",
            "1. Password Policy",
            "All corporate passwords must be at least 14 characters long and must",
            "include a mix of uppercase letters, lowercase letters, digits and",
            "symbols. Passwords expire every 90 days and cannot be reused within",
            "the previous 12 passwords. Multi-factor authentication is mandatory",
            "for all systems that contain customer payment data.",
            "",
            "2. Data Classification",
            "Data is classified into four tiers: Public, Internal, Confidential",
            "and Restricted. Confidential data includes payroll records, customer",
            "identifiers and unreleased product roadmaps. Restricted data is",
            "limited to named individuals with a business need.",
            "",
            "3. Incident Response",
            "Suspected security incidents must be reported to the Security Desk",
            "within 1 hour of discovery. The Security Desk operates a 24x7",
            "on-call rota and guarantees an initial response within 30 minutes.",
        ]
        text_obj = pdf.beginText(40, height - 50)
        for line in page_one:
            text_obj.textLine(line)
        pdf.drawText(text_obj)
        pdf.showPage()
        page_two = [
            "4. Third-Party Assessments",
            "Any third party that stores Acme customer data must pass an",
            "independent SOC 2 Type II audit before contract signature and",
            "must renew it annually. Vendors that cannot produce a current",
            "SOC 2 report are restricted to synthetic, anonymised test data.",
            "",
            "5. Encryption Standards",
            "Data at rest must be encrypted with AES-256. Data in transit must",
            "use TLS 1.3 for all external endpoints and TLS 1.2 as the internal",
            "minimum. Cryptographic keys are rotated every 24 months.",
        ]
        text_obj = pdf.beginText(40, height - 50)
        for line in page_two:
            text_obj.textLine(line)
        pdf.drawText(text_obj)
        pdf.save()
    except Exception as exc:  # noqa: BLE001
        print(f"! pdf sample skipped: {exc}")

    pptx_path = SAMPLES / "acme_roadmap.pptx"
    try:
        from pptx import Presentation
        from pptx.util import Inches

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = "Product Roadmap Q4 2025"
        slide.placeholders[1].text = (
            "Apollo analytics launches 15 October 2025. "
            "Beta cohort of 40 enterprise customers."
        )
        slide2 = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide2.shapes.title.text = "Orion migration completes 20 December 2025"
        presentation.save(pptx_path)
    except Exception as exc:  # noqa: BLE001
        print(f"! pptx sample skipped: {exc}")

    return [p for p in (txt, csv, docx_path, pdf_path, pptx_path) if p.exists()]


def main() -> int:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True, exist_ok=True)

    store = Store(WORK / "index")
    engine = RAGEngine(store)

    print("engine:", engine.provider_status())

    for path in build_samples():
        doc_id = new_doc_id(path.name)
        sections = load_document(path)
        chunks = chunk_sections(sections, doc_id, path.name)
        doc = Document(
            doc_id=doc_id,
            filename=path.name,
            filetype=path.suffix.lstrip("."),
            size_bytes=path.stat().st_size,
            added_at=time.time(),
            pages=len(sections),
            chunks=len(chunks),
            characters=sum(len(s.text) for s in sections),
        )
        store.add_document(doc, chunks)
        print(f"indexed {path.name}: {len(sections)} sections -> {len(chunks)} chunks")

    questions = [
        "What is the maximum hotel reimbursement per night in a Tier 1 city?",
        "How many days of annual leave do I get after 8 years of service?",
        "Which region had the highest revenue in Q3 and how much was it?",
        "What is the deadline for submitting an expense claim?",
        "Who do I contact about travel policy questions?",
        "What is the minimum password length and how often do passwords expire?",
        "How quickly must a security incident be reported?",
        "Which encryption standard is required for data at rest?",
        "When does the Apollo product launch and what is the beta cohort size?",
        "What is the notice period during and after probation?",
        # Comparative question - must reach BOTH the handbook and the policy.
        "Compare the leave policy in the handbook with the travel policy: "
        "what notice periods and deadlines apply?",
        "What is the capital of France?",  # must not be answered from documents
    ]

    failures = 0
    for question in questions:
        print("\n" + "=" * 78)
        print("Q:", question)
        result = engine.answer(question)
        print("A:", result.answer)
        print(f"[mode={result.mode} grounded={result.grounded} {result.elapsed_ms}ms]")
        for source in result.sources:
            print(f"   [{source.index}] {source.citation} (score {source.score})")
            print(f"       {source.snippet[:180]!r}")

        if question.startswith("What is the capital"):
            if "could not find" not in result.answer.lower():
                print("!! FAIL: out-of-scope question was answered from documents")
                failures += 1
        elif question.startswith("Compare"):
            files = {s.filename for s in result.sources}
            if "acme_handbook.docx" not in files or "acme_policy.txt" not in files:
                print(f"!! FAIL: comparison did not surface both documents: {files}")
                failures += 1
            if "30 days" not in result.answer and "15 days" not in result.answer:
                print("!! FAIL: comparison missing the policy deadline")
                failures += 1
        else:
            if result.answer.lower().startswith("i could not find"):
                print("!! FAIL: in-scope question was not answered")
                failures += 1

    print("\n" + "=" * 78)
    print("FAILURES:", failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
