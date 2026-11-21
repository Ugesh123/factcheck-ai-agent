"""PDF text extraction with page numbers, using PyMuPDF."""

from dataclasses import dataclass, field

import fitz  # PyMuPDF

MAX_FILE_MB = 15
MAX_PAGES = 40


class PDFReadError(Exception):
    """Raised with a user-facing message when a PDF can't be read."""


@dataclass
class Page:
    number: int  # 1-based
    text: str


@dataclass
class Document:
    pages: list[Page] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(p.text) for p in self.pages)

    @property
    def image_only_pages(self) -> list[int]:
        return [p.number for p in self.pages if not p.text.strip()]

    def full_text(self) -> str:
        """Text with page markers, so extracted claims can cite their page."""
        return "\n\n".join(
            f"[PAGE {p.number}]\n{p.text}" for p in self.pages if p.text.strip()
        )


def read_pdf(data: bytes) -> Document:
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        raise PDFReadError(f"This file is larger than {MAX_FILE_MB} MB. Please upload a smaller PDF.")

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupt or not a real PDF
        raise PDFReadError("This file couldn't be opened as a PDF. Please check the file and try again.") from exc

    if pdf.needs_pass:
        raise PDFReadError("This PDF is password-protected. Please upload an unlocked copy.")

    if pdf.page_count > MAX_PAGES:
        raise PDFReadError(f"This PDF has {pdf.page_count} pages; the limit is {MAX_PAGES}.")

    doc = Document()
    for i, page in enumerate(pdf, start=1):
        # sort=True keeps a natural reading order in multi-column layouts
        doc.pages.append(Page(number=i, text=page.get_text("text", sort=True)))
    pdf.close()

    if not any(p.text.strip() for p in doc.pages):
        raise PDFReadError(
            "No text could be extracted. This PDF looks like scanned images; "
            "please upload a text-based PDF."
        )
    return doc
