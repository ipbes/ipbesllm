from pdf_loader import extract_pdf_chunks


PDF_PATH = "data/pdf/2013-1.pdf"


def main():
    print(f"Extracting chunks from: {PDF_PATH}")
    print()

    chunks = extract_pdf_chunks(PDF_PATH)

    print(f"Number of chunks: {len(chunks)}")

    if not chunks:
        print()
        print(
            "No text was extracted. The PDF may be scanned/"
            "image-based, or chunk_size/overlap may be wrong."
        )
        return

    total_chars = sum(len(c["text"]) for c in chunks)

    print(f"Total characters: {total_chars}")
    print(
        f"Average chunk length: "
        f"{total_chars // len(chunks)}"
    )
    print()

    for chunk in chunks[:3]:
        print("=" * 80)
        print(f"Source: {chunk['source_file']}")
        print(f"Page: {chunk['page']} of {chunk['page_count']}")
        print(
            f"Chunk: {chunk['chunk_index']} "
            f"of {chunk['chunk_total']} "
            f"(doc total: {chunk['total_chunks']})"
        )
        print(f"ID: {chunk['chunk_id']}")
        print(f"Length: {len(chunk['text'])} characters")
        print()
        print(chunk["text"])


if __name__ == "__main__":
    main()