from collections import Counter

from ttl_loader import parse_ttl_file


TTL_PATH = "data/ttl/LDR_v01.ttl"


def main():
    print(f"Extracting chunks from: {TTL_PATH}")
    print()

    chunks = parse_ttl_file(TTL_PATH)

    print(f"Number of chunks: {len(chunks)}")
    print()

    if not chunks:
        print("No chunks were extracted.")
        return

    counts = Counter(c["chunk_type"] for c in chunks)
    print("Chunks by type:")
    for chunk_type, n in counts.most_common():
        print(f"  {chunk_type:22s} {n}")
    print()

    # Show one example of each chunk type so the shape is visible.
    seen = set()
    for chunk in chunks:
        if chunk["chunk_type"] in seen:
            continue
        seen.add(chunk["chunk_type"])

        print("=" * 80)
        print(f"Type: {chunk['chunk_type']}")
        print(f"ID:   {chunk['chunk_id']}")
        print(f"eId:  {chunk['eId']}")
        print(f"Len:  {len(chunk['text'])} characters")
        print()
        print(chunk["text"][:1500])
        if len(chunk["text"]) > 1500:
            print("  [... truncated ...]")
        print()


if __name__ == "__main__":
    main()