"""Document loaders.

Each loader returns a list of :class:`Section` objects so that every chunk can
keep a precise human-readable location (page number, sheet name, slide index)
that is later shown as a citation.
"""

from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path

from .models import Section

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".log",
    ".tex",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".c",
    ".h",
    ".cpp",
    ".cs",
    ".go",
    ".rb",
    ".php",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sql",
    ".html",
    ".htm",
    ".xml",
}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".csv",
    ".tsv",
    ".json",
}


class UnsupportedDocument(Exception):
    """Raised when a file type cannot be parsed."""


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- PDF
def _load_pdf(path: Path) -> list[Section]:
    sections: list[Section] = []
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                parts = []
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(text)
                for table in page.extract_tables() or []:
                    rows = []
                    for row in table:
                        cells = [(c or "").strip() for c in row]
                        if any(cells):
                            rows.append(" | ".join(cells))
                    if rows:
                        parts.append("[table]\n" + "\n".join(rows))
                body = _clean("\n\n".join(parts))
                if body:
                    sections.append(Section(text=body, page=page_no, order=page_no))
    except Exception:
        sections = []

    if not sections:
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            for page_no, page in enumerate(reader.pages, start=1):
                body = _clean(page.extract_text() or "")
                if body:
                    sections.append(Section(text=body, page=page_no, order=page_no))
        except Exception as exc:  # noqa: BLE001
            raise UnsupportedDocument(f"Could not read PDF: {exc}") from exc

    if not sections:
        raise UnsupportedDocument(
            "No extractable text found. The PDF may be a scanned image "
            "(OCR is not enabled in this build)."
        )
    return sections


# -------------------------------------------------------------------------- DOCX
def _load_docx(path: Path) -> list[Section]:
    try:
        import docx
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocument(f"python-docx unavailable: {exc}") from exc

    document = docx.Document(str(path))
    sections: list[Section] = []
    current_heading: str | None = None
    buffer: list[str] = []
    order = 0

    def flush() -> None:
        nonlocal buffer, order
        body = _clean("\n".join(buffer))
        if body:
            order += 1
            sections.append(Section(text=body, page=None, section=current_heading, order=order))
        buffer = []

    for para in document.paragraphs:
        text = (para.text or "").strip()
        style = (para.style.name or "") if para.style is not None else ""
        if not text:
            continue
        if style.lower().startswith("heading"):
            flush()
            current_heading = text
            continue
        buffer.append(text)

    flush()

    for t_index, table in enumerate(document.tables, start=1):
        rows = []
        for row in table.rows:
            cells = [(c.text or "").strip() for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            order += 1
            sections.append(
                Section(
                    text=_clean("[table]\n" + "\n".join(rows)),
                    page=None,
                    section=f"Table {t_index}",
                    order=order,
                )
            )

    if not sections:
        raise UnsupportedDocument("The document contains no readable text.")
    return sections


# ------------------------------------------------------------------------- PPTX
def _load_pptx(path: Path) -> list[Section]:
    try:
        from pptx import Presentation
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocument(f"python-pptx unavailable: {exc}") from exc

    presentation = Presentation(str(path))
    sections: list[Section] = []
    for slide_no, slide in enumerate(presentation.slides, start=1):
        parts: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                text = _clean(shape.text_frame.text)
                if text:
                    parts.append(text)
            if getattr(shape, "has_table", False):
                rows = []
                for row in shape.table.rows:
                    cells = [(c.text or "").strip() for c in row.cells]
                    if any(cells):
                        rows.append(" | ".join(cells))
                if rows:
                    parts.append("[table]\n" + "\n".join(rows))
        body = _clean("\n\n".join(parts))
        if body:
            sections.append(Section(text=body, page=slide_no, section=f"Slide {slide_no}", order=slide_no))
    if not sections:
        raise UnsupportedDocument("The presentation contains no readable text.")
    return sections


# ------------------------------------------------------------------------- XLSX
def _load_xlsx(path: Path) -> list[Section]:
    try:
        from openpyxl import load_workbook
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocument(f"openpyxl unavailable: {exc}") from exc

    workbook = load_workbook(str(path), data_only=True, read_only=True)
    sections: list[Section] = []
    order = 0
    for sheet in workbook.worksheets:
        rows: list[str] = []
        for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            values = ["" if v is None else str(v) for v in row]
            if not any(v.strip() for v in values):
                continue
            if row_index == 1:
                rows.append(" | ".join(values))
            else:
                rows.append(" | ".join(values))
        body = _clean("\n".join(rows))
        if body:
            order += 1
            sections.append(Section(text=body, page=None, section=f"Sheet: {sheet.title}", order=order))
    workbook.close()
    if not sections:
        raise UnsupportedDocument("The workbook contains no readable values.")
    return sections


# --------------------------------------------------------------------------- CSV
def _load_csv(path: Path, delimiter: str | None = None) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        raise UnsupportedDocument("The CSV file is empty.")
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except Exception:  # noqa: BLE001
        delimiter = delimiter or ","

    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows: list[str] = []
    header: str | None = None
    for row_index, row in enumerate(reader, start=1):
        cells = [c.strip() for c in row]
        if not any(cells):
            continue
        if header is None:
            header = " | ".join(cells)
            rows.append(header)
        else:
            rows.append(f"Row {row_index}: " + " | ".join(cells))
    body = _clean("\n".join(rows))
    if not body:
        raise UnsupportedDocument("The CSV file contains no rows.")
    return [Section(text=body, page=None, section="Rows", order=1)]


# ------------------------------------------------------------------------- HTML
def _load_html(path: Path) -> list[Section]:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocument(f"beautifulsoup4 unavailable: {exc}") from exc

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    title = soup.title.get_text(strip=True) if soup.title else None
    body = _clean(soup.get_text("\n"))
    if not body:
        raise UnsupportedDocument("The HTML file contains no readable text.")
    return [Section(text=body, page=None, section=title or "Page content", order=1)]


# ------------------------------------------------------------------------- JSON
def _load_json(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        raise UnsupportedDocument(f"Invalid JSON: {exc}") from exc

    lines: list[str] = []

    def walk(node: object, trail: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{trail}.{key}" if trail else str(key))
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")
        else:
            lines.append(f"{trail}: {node}")

    walk(data)
    body = _clean("\n".join(lines))
    if not body:
        raise UnsupportedDocument("The JSON file contains no readable values.")
    return [Section(text=body, page=None, section="Values", order=1)]


# --------------------------------------------------------------------------- TXT
def _load_text(path: Path) -> list[Section]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = _clean(raw)
    if not body:
        raise UnsupportedDocument("The file is empty.")

    sections: list[Section] = []
    blocks = re.split(r"\n(?=[A-Z0-9][^\n]{0,80}\n[-=]{3,}\n)", body)
    if len(blocks) <= 1:
        paragraphs = [p for p in body.split("\n\n") if p.strip()]
        if len(paragraphs) > 1:
            for index, para in enumerate(paragraphs, start=1):
                sections.append(Section(text=para, page=None, section=None, order=index))
            return sections
        return [Section(text=body, page=None, section=None, order=1)]

    for index, block in enumerate(blocks, start=1):
        heading = block.split("\n", 1)[0].strip()[:80]
        sections.append(Section(text=_clean(block), page=None, section=heading, order=index))
    return sections


# --------------------------------------------------------------------------- API
def load_document(path: Path) -> list[Section]:
    """Parse ``path`` into sections, dispatching on the file extension."""

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix == ".docx":
        return _load_docx(path)
    if suffix == ".pptx":
        return _load_pptx(path)
    if suffix == ".xlsx":
        return _load_xlsx(path)
    if suffix in {".csv", ".tsv"}:
        return _load_csv(path, "\t" if suffix == ".tsv" else None)
    if suffix in {".html", ".htm", ".xml"}:
        return _load_html(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix in TEXT_EXTENSIONS:
        return _load_text(path)
    raise UnsupportedDocument(
        f"Unsupported file type '{suffix or path.name}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
    )


def is_supported(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_EXTENSIONS
