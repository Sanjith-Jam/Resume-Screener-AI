"""
PDF text extraction and preprocessing module.
Uses pdfplumber for robust PDF parsing.
"""

import pdfplumber
import re
from io import BytesIO


def extract_text_from_pdf(pdf_file) -> str:
    """Extract all text from a PDF file (file path or BytesIO object)."""
    text_parts = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = re.sub(r"[^\S\n]+", " ", text)       # collapse horizontal whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)       # limit consecutive newlines
    text = re.sub(r"[^\x20-\x7E\n]", "", text)  # strip non-printable chars
    return text.strip()


def extract_sections(text: str) -> dict:
    """
    Attempt to split a resume into common sections.
    Returns a dict with section names as keys. If no headings are found,
    returns {"full_text": text}.
    """
    heading_pattern = re.compile(
        r"^(SUMMARY|OBJECTIVE|EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|"
        r"EDUCATION|SKILLS|TECHNICAL SKILLS|CERTIFICATIONS|PROJECTS|"
        r"ACHIEVEMENTS|AWARDS|PUBLICATIONS|LANGUAGES|INTERESTS|REFERENCES|"
        r"CONTACT|PROFILE|ABOUT ME|QUALIFICATIONS)\b",
        re.IGNORECASE | re.MULTILINE,
    )

    matches = list(heading_pattern.finditer(text))
    if not matches:
        return {"full_text": text}

    sections = {}
    for i, match in enumerate(matches):
        section_name = match.group(0).strip().title()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[section_name] = text[start:end].strip()

    # Include any text before the first heading as "Header"
    if matches[0].start() > 0:
        sections["Header"] = text[: matches[0].start()].strip()

    return sections


def process_pdf(pdf_file) -> dict:
    """Full pipeline: extract -> clean -> section-split."""
    raw = extract_text_from_pdf(pdf_file)
    cleaned = clean_text(raw)
    sections = extract_sections(cleaned)
    return {
        "raw_text": raw,
        "cleaned_text": cleaned,
        "sections": sections,
        "char_count": len(cleaned),
        "word_count": len(cleaned.split()),
    }
