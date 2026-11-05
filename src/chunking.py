"""Split oversized TTL chunks so each embedding request stays small.

Hook into ttl_loader.py with two lines:

    from chunking import split_long_chunks      # top of file
    ...
    return split_long_chunks(chunks)            # replaces `return chunks`
"""
MAX_BODY_CHARS = 2000   # ~400-500 tokens; tune to taste
# Only these chunk types are split to MAX_BODY_CHARS. Messages (bgm/subm/key) stay
# whole so each stays a coherent unit; add "bgm" here if you want those split too.
SPLIT_TYPES = {"sch"}
# No chunk of any type may be longer than this: the embedding model takes at most
# 8192 tokens per input. 20,000 characters leaves room for text with fewer than the
# usual 4 characters per token (numbers, DOIs, non-Latin scripts).
MAX_CHUNK_CHARS = 20000


def too_long(chunks: list[dict], limit: int = MAX_CHUNK_CHARS) -> list[dict]:
    """The chunks the embedding model could reject for their length."""
    return [c for c in chunks if len(c["text"]) > limit]


def split_body(body: str, limit: int = MAX_BODY_CHARS) -> list[str]:
    """Pack paragraphs into parts <= limit chars; hard-split giant paragraphs."""
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    parts: list[str] = []
    cur = ""
    for p in paras:
        while len(p) > limit:
            cut = p.rfind(". ", 0, limit)
            cut = cut + 1 if cut > limit // 2 else limit
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(p[:cut].strip())
            p = p[cut:].strip()
        if not p:
            continue
        if cur and len(cur) + 2 + len(p) > limit:
            parts.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        parts.append(cur)
    return parts or [body]


def split_long_chunks(chunks: list[dict], limit: int = MAX_BODY_CHARS,
                      split_types: set[str] = SPLIT_TYPES) -> list[dict]:
    """Split chunks whose body exceeds `limit`, repeating the header on each part.
    Chunks of the other types are split only when longer than MAX_CHUNK_CHARS.

    The loader builds text as header lines + blank line + body, so the first
    blank line separates them.
    """
    out: list[dict] = []
    for chunk in chunks:
        header, sep, body = chunk["text"].partition("\n\n")
        if chunk["chunk_type"] in split_types:
            part_limit = limit
        elif len(chunk["text"]) > MAX_CHUNK_CHARS:
            part_limit = MAX_CHUNK_CHARS - len(header) - len(sep)
        else:
            out.append(chunk)
            continue
        if not sep or len(body) <= part_limit:
            out.append(chunk)
            continue
        parts = split_body(body, part_limit)
        if len(parts) == 1:
            out.append(chunk)
            continue
        for n, part in enumerate(parts, start=1):
            out.append({
                **chunk,
                "text": f"{header}\n\n{part}",
                "chunk_id": f"{chunk['chunk_id']}-p{n}",
                "paragraph": f"{n}/{len(parts)}",
            })
    return out
