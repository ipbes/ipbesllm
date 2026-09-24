from pathlib import Path

import fitz


def _back_off_to_whitespace(
    text: str,
    end: int,
    min_end: int,
) -> int:
    """
    If `end` is not at a whitespace boundary, move it backwards
    to the nearest whitespace, but never below `min_end`.

    This avoids splitting mid-word or mid-sentence, which improves
    embedding quality on narrative PDFs.
    """
    if end >= len(text):
        return end

    # If the character right after `end` is whitespace, we're
    # already at a clean boundary.
    if end > 0 and text[end - 1].isspace():
        return end

    # Otherwise walk backwards to the previous whitespace.
    cursor = end
    while cursor > min_end and not text[cursor - 1].isspace():
        cursor -= 1

    # If we couldn't find any whitespace at all, keep `end`.
    if cursor <= min_end:
        return end

    return cursor


def extract_pdf_chunks(
    pdf_path: str,
    chunk_size: int = 2000,
    overlap: int = 300,
) -> list[dict]:
    """
    Extract text from a PDF and split it into overlapping chunks.

    Each chunk is a dictionary containing:
      - text
      - source_file
      - page
      - page_count
      - chunk_id
      - chunk_index   (index of this chunk within its page)
      - chunk_total   (number of chunks on this page)
      - total_chunks  (number of chunks across the whole document)

    Raises:
        ValueError: if `overlap` is not strictly less than `chunk_size`,
                    which would prevent the window from advancing.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be strictly less than "
            f"chunk_size ({chunk_size}); otherwise the sliding "
            f"window never advances."
        )

    path = Path(pdf_path)

    chunks: list[dict] = []

    with fitz.open(path) as document:

        page_count = document.page_count

        # Pull document-level metadata once.
        pdf_metadata = document.metadata or {}

        # We need `total_chunks` in every chunk, so do a first pass
        # to compute the per-page chunk lists, then flatten.
        per_page_chunks: list[list[str]] = []

        for page_number in range(page_count):

            page = document.load_page(page_number)
            text = page.get_text("text").strip()

            if not text:
                per_page_chunks.append([])
                continue

            page_chunks: list[str] = []

            start = 0

            while start < len(text):

                end = min(start + chunk_size, len(text))

                # Try to end on a whitespace boundary without
                # going below `start + chunk_size // 2`, so chunks
                # don't collapse to tiny fragments.
                if end < len(text):
                    end = _back_off_to_whitespace(
                        text,
                        end,
                        min_end=start + max(1, chunk_size // 2),
                    )

                chunk_text = text[start:end].strip()

                if chunk_text:
                    page_chunks.append(chunk_text)

                if end >= len(text):
                    break

                # Guaranteed to advance because we enforced
                # overlap < chunk_size above.
                start = end - overlap

            per_page_chunks.append(page_chunks)

        total_chunks = sum(len(c) for c in per_page_chunks)

        # Second pass: emit chunks with full metadata.
        for page_number, page_chunks in enumerate(
            per_page_chunks, start=1
        ):

            chunk_total = len(page_chunks)

            for chunk_index, chunk_text in enumerate(page_chunks):

                chunks.append(
                    {
                        "text": chunk_text,
                        "source_file": path.name,
                        "page": page_number,
                        "page_count": page_count,
                        "chunk_id": (
                            f"{path.stem}-"
                            f"page-{page_number}-"
                            f"chunk-{chunk_index}"
                        ),
                        "chunk_index": chunk_index,
                        "chunk_total": chunk_total,
                        "total_chunks": total_chunks,
                    }
                )

    return chunks