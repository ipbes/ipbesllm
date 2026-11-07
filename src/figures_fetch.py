"""Download the figure packages of the IPBES assessments from Zenodo.

    python src/figures_fetch.py          # into .cache/zenodo/, then figures_build.py

config/figures_records.json lists one Zenodo record per assessment and part
(the file data/csv/figures_tables.csv gives one DOI for several chapters, while
a record holds one chapter). Downloads the zips and the loose SVG, JPEG and PNG
files; PDFs and source files (.ai, .indd, .eps) outside a zip are not needed.
Files already there with the right checksum are skipped, so it can be re-run.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDS = ROOT / "config" / "figures_records.json"
CACHE = ROOT / ".cache" / "zenodo"
WANTED = {".zip", ".svg", ".jpg", ".jpeg", ".png"}

KINDS = {"fig": "Figure", "figure": "Figure", "table": "Table", "box": "Box"}
_ITEM = re.compile(
    r"(figure|fig|table|box)[ _.]*(spm|sm)?[ _.]*([A-Z]?\d+(?:\.\d+)*[A-Za-z]?)(?![\d])",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Item:
    code: str
    part: str       # chapter number or SPM
    kind: str       # Figure, Table, Box
    number: str     # as the TTL writes it after the part: "9", "A1", "6"
    letter: str     # a printed part of the item: "A" in Table 2.6A


def parse_item(name: str, code: str, part: str) -> tuple[Item | None, str]:
    """The figure or table a file name means ("Figure 2.05.zip", "A_Table_2.6A.jpg",
    "IPBES_TCA_FIGURE SPM 12.zip"), and a note if it looks wrong. None for
    supplementary material, which is outside the report's numbering."""
    m = _ITEM.search(name)
    if not m:
        return None, ""
    kind = KINDS[m.group(1).lower()]
    if (m.group(2) or "").lower() == "sm":
        return None, ""
    raw = m.group(3)
    letter = ""
    if raw[-1].isalpha() and any(c.isdigit() for c in raw):
        raw, letter = raw[:-1], raw[-1].upper()
    note = ""
    if part == "SPM":
        number = raw.split(".")[-1]
        number = number if number[0].isalpha() else str(int(number))
    else:
        pieces = raw.split(".")
        if len(pieces) < 2:
            return None, ""
        if int(pieces[0]) != int(part):
            note = f"file says chapter {pieces[0]}, record says chapter {part}"
        number = ".".join(str(int(p)) for p in pieces[1:])
    return Item(code, part, kind, number, letter), note


def ttl_identifier(item: Item) -> str:
    """As dcterms:identifier in the TTL: Figure3.9, TableSPM.A1."""
    return (f"{item.kind}SPM.{item.number}" if item.part == "SPM"
            else f"{item.kind}{item.part}.{item.number}")


def item_dois(code: str) -> dict[str, str]:
    """{identifier: concept DOI} of the figures and tables of an assessment that
    have files in a Zenodo record, from the record's file names (no download
    needed). For pdf_to_ttl.py's ipbes:hasDoi."""
    if not RECORDS.exists():
        return {}
    dois = {}
    for record in json.loads(RECORDS.read_text()):
        if record["code"] != code or not record.get("conceptdoi"):
            continue
        for f in record["files"]:
            item, note = parse_item(f["key"], code, record["part"])
            if item and not note and item.kind in ("Figure", "Table"):
                dois.setdefault(ttl_identifier(item), record["conceptdoi"])
    return dois


def local_path(record_id: int, index: int, key: str) -> Path:
    """Each download in its own directory, under its name without any path."""
    return CACHE / str(record_id) / f"{index:03}" / Path(key).name


def has_checksum(path: Path, checksum: str | None) -> bool:
    if not path.exists():
        return False
    if not checksum:
        return True
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return f"md5:{digest.hexdigest()}" == checksum


def download(client, url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    for attempt in range(6):
        with client.stream("GET", url) as response:
            if response.status_code == 429 or response.status_code >= 500:
                wait = int(response.headers.get("Retry-After", 30 * (attempt + 1)))
                print(f"  HTTP {response.status_code}, waiting {wait} s", flush=True)
                time.sleep(wait)
                continue
            response.raise_for_status()
            with partial.open("wb") as f:
                for block in response.iter_bytes(1 << 20):
                    f.write(block)
        partial.replace(path)
        return
    raise RuntimeError(f"gave up on {url}")


def main() -> int:
    import httpx

    records = json.loads(RECORDS.read_text())
    failed = []
    fetched = skipped = 0
    with httpx.Client(follow_redirects=True, timeout=600,
                      headers={"User-Agent": "ipbes-figures/0.1"}) as client:
        for record in records:
            for index, item in enumerate(record["files"]):
                if Path(item["key"]).suffix.lower() not in WANTED:
                    continue
                path = local_path(record["id"], index, item["key"])
                if has_checksum(path, item.get("checksum")):
                    skipped += 1
                    continue
                try:
                    download(client, item["url"], path)
                    if not has_checksum(path, item.get("checksum")):
                        raise RuntimeError("checksum mismatch")
                    fetched += 1
                except (httpx.HTTPError, RuntimeError) as exc:
                    failed.append(f"{record['id']} {item['key']}: {exc}")
                    print(f"FAILED {failed[-1]}", flush=True)
                time.sleep(1.1)     # Zenodo limits requests without an account
            print(f"{record['code']} {record['part']} ({record['id']}): "
                  f"{fetched} fetched, {skipped} already there", flush=True)
    print(f"Done: {fetched} fetched, {skipped} already there, {len(failed)} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
