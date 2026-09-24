from pathlib import Path
import fitz


def extract_pdf_chunks(
    pdf_path: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    """
    Extract text from a PDF and split it into overlapping chunks.

    Returns dictionaries containing:
      - text
      - source_file
      - page
      - chunk_id
    """

    path = Path(pdf_path)

    document = fitz.open(path)

    chunks = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text("text").strip()

        if not text:
            continue

        start = 0
        chunk_number = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "source_file": path.name,
                        "page": page_number,
                        "chunk_id": (
                            f"{path.stem}-"
                            f"page-{page_number}-"
                            f"chunk-{chunk_number}"
                        ),
                    }
                )

            if end >= len(text):
                break

            start = end - overlap
            chunk_number += 1

    document.close()

    return chunks