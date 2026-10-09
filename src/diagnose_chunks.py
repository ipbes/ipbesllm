"""Inspect TTL chunks before indexing.

Reports chunk size per type, the longest chunks, duplicate chunk_ids, the
largest consecutive batch (what one Ollama request actually carries) and,
with --embed N, how long Ollama takes to embed the N longest chunks.

Run from the project root:
    python diagnose_chunks.py
    python diagnose_chunks.py --embed 5
"""
import argparse
import statistics
import time
from collections import Counter, defaultdict

from ttl_loader import parse_ttl_file
from ttl_index import TTL_DIR, EMBED_MODEL, assessment_id_from_path, make_batches


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(len(values) * q))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=0,
                    help="time embedding of the N longest chunks")
    args = ap.parse_args()

    chunks = []
    for path in sorted(TTL_DIR.glob("*.ttl")):
        for c in parse_ttl_file(str(path)):
            c["assessment"] = assessment_id_from_path(path)
            chunks.append(c)
    print(f"\nTotal chunks: {len(chunks)}\n")

    by_type = defaultdict(list)
    for c in chunks:
        by_type[c["chunk_type"]].append(len(c["text"]))
    print(f"{'type':<10}{'count':>7}{'median':>9}{'p95':>9}{'max':>9}  (chars)")
    for t, lens in sorted(by_type.items()):
        print(f"{t:<10}{len(lens):>7}{int(statistics.median(lens)):>9}"
              f"{pct(lens, .95):>9}{max(lens):>9}")

    print("\nLongest 10 chunks:")
    longest = sorted(chunks, key=lambda c: len(c["text"]), reverse=True)
    for c in longest[:10]:
        print(f"  {len(c['text']):>7}  {c['chunk_type']:<8} {c['chunk_id']}")

    dups = [i for i, n in Counter(c["chunk_id"] for c in chunks).items() if n > 1]
    print(f"\nDuplicate chunk_ids: {len(dups)}")
    for i in dups[:10]:
        print(f"  {i}")
    if dups:
        print("  (Chroma raises DuplicateIDError if two share a batch; "
              "otherwise upsert silently overwrites.)")

    batches = list(make_batches(chunks))
    worst = max(batches, key=lambda b: sum(len(c["text"]) for c in b))
    print(f"\nBatches at the indexer's default limits: {len(batches)}; heaviest has "
          f"{len(worst)} chunks / {sum(len(c['text']) for c in worst):,} chars")

    if args.embed:
        import ollama
        client = ollama.Client(timeout=600)
        print(f"\nWarming up {EMBED_MODEL}...")
        client.embed(model=EMBED_MODEL, input=["warmup"])
        print(f"Timing the {args.embed} longest chunks:")
        for c in longest[:args.embed]:
            t0 = time.perf_counter()
            try:
                client.embed(model=EMBED_MODEL, input=[c["text"]])
                status = "ok"
            except Exception as e:  # noqa: BLE001
                status = f"{type(e).__name__}: {e}"
            print(f"  {len(c['text']):>7} chars  {time.perf_counter() - t0:6.1f}s  "
                  f"{c['chunk_id']}  {status}")


if __name__ == "__main__":
    main()
