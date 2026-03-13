from collections.abc import Iterable


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    chunks: list[str] = []

    for start in range(0, len(cleaned), step):
        chunk = cleaned[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)

    return chunks


def build_chunk_records(
    pages: Iterable[dict],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    records: list[dict] = []

    for page_item in pages:
        page_text = page_item["text"]
        chunks = chunk_text(page_text, chunk_size, chunk_overlap)

        for index, content in enumerate(chunks):
            records.append(
                {
                    "chunk_id": f"p{page_item['page']}-c{index}",
                    "content": content,
                    "source": page_item["source"],
                    "page": page_item["page"],
                }
            )

    return records
