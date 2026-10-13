"""The data a deployed app needs, packed into one zip: the Chroma database
and the thesaurus and geography RDF. Neither is in git (chroma/ and data/
are git-ignored, and chroma.sqlite3 is over GitHub's 100 MB file limit).

    python src/data_bundle.py             # write dist/ipbes-data.zip

Upload the zip to private storage (e.g. an Azure Blob container) and give the
app a read-only link to it in DATA_BUNDLE_URL (e.g. a SAS URL). On startup,
ensure_data() downloads and unpacks it if the data is missing.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import httpx
from loguru import logger

from settings import CHROMA_DIR, GEO_RDF, PROJECT_ROOT, RDF_DIR, THESAURUS_RDF

BUNDLE_PATH = PROJECT_ROOT / "dist" / "ipbes-data.zip"
DATA_BUNDLE_URL = os.getenv("DATA_BUNDLE_URL", "")

# Zip entry prefixes -> where they are unpacked.
CHROMA_PREFIX = "chroma/"
RDF_PREFIX = "rdf/"


def data_present() -> bool:
    return all(p.exists() for p in (CHROMA_DIR / "chroma.sqlite3",
                                    THESAURUS_RDF, GEO_RDF))


def ensure_data() -> None:
    """Download and unpack DATA_BUNDLE_URL if the data is missing."""
    if data_present():
        return
    if not DATA_BUNDLE_URL:
        logger.warning("Search data missing and DATA_BUNDLE_URL not set.")
        return

    logger.info("Downloading the data bundle...")
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "bundle.zip"
        with httpx.stream("GET", DATA_BUNDLE_URL, timeout=600,
                          follow_redirects=True) as response:
            response.raise_for_status()
            with zip_path.open("wb") as f:
                for block in response.iter_bytes(1 << 20):
                    f.write(block)

        targets = {CHROMA_PREFIX: CHROMA_DIR, RDF_PREFIX: RDF_DIR}
        with zipfile.ZipFile(zip_path) as bundle:
            for member in bundle.infolist():
                prefix = next((p for p in targets if member.filename.startswith(p)), None)
                if prefix is None or member.is_dir():
                    continue
                target = (targets[prefix] / member.filename[len(prefix):]).resolve()
                if not target.is_relative_to(targets[prefix].resolve()):
                    raise ValueError(f"Unsafe path in bundle: {member.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
    logger.info("Data bundle unpacked.")


def build_bundle() -> Path:
    """Zip CHROMA_DIR and the two RDF files into BUNDLE_PATH."""
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLE_PATH, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(CHROMA_DIR.rglob("*")):
            if path.is_file():
                bundle.write(path, CHROMA_PREFIX + path.relative_to(CHROMA_DIR).as_posix())
        for path in (THESAURUS_RDF, GEO_RDF):
            bundle.write(path, RDF_PREFIX + path.name)
    return BUNDLE_PATH


if __name__ == "__main__":
    path = build_bundle()
    print(f"Wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")
