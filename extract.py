from pypdf import PdfReader
import re

def extract_all_text(file_name):
    reader = PdfReader(file_name)

    number_of_pages = len(reader.pages)
    all_text = ""

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:  # sometimes extract_text() returns None
            all_text += f"\n\n--- Page {page_num + 1} ---\n\n"
            all_text += text

    print(f"Number of pages: {number_of_pages}")
    return all_text

def clean_pdf_text(text):
    if not text:
        return ""

    # Remove hyphenation at line breaks
    text = re.sub(r"-\n", "", text)

    # Convert single newlines to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Convert multiple newlines to paragraph breaks
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()
