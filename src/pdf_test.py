from pdf_loader import extract_pdf_chunks


chunks = extract_pdf_chunks(
    "data/pdf/2013-1.pdf"
)

print(f"Number of chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print("=" * 80)
    print(f"Source: {chunk['source_file']}")
    print(f"Page: {chunk['page']}")
    print(f"ID: {chunk['chunk_id']}")
    print()
    print(chunk["text"])