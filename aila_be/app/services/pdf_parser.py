from pypdf import PdfReader


def parse_pdf(file_path: str, source_name: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages: list[dict] = []

    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue

        pages.append(
            {
                "page": i,
                "source": source_name,
                "text": text,
            }
        )

    return pages
