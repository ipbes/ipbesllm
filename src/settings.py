"""Project paths, resolved from the project root so scripts work from any
working directory.

CHROMA_DIR can be moved with the CHROMA_DIR environment variable.

Settings and secrets (e.g. AZURE_OPENAI_API_KEY) can also be put in a .env
file at the project root; it is git-ignored. Real environment variables win.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdf"
TTL_DIR = DATA_DIR / "ttl"
XML_DIR = DATA_DIR / "xml"
RDF_DIR = DATA_DIR / "rdf"   # thesaurus and geography RDF

THESAURUS_RDF = RDF_DIR / "ipbes-thesaurus.rdf"
# Local SKOS additions loaded on top of the thesaurus. Kept with the code,
# not in data/, so it cannot be left out of a data package.
THESAURUS_ENRICHMENT = PROJECT_ROOT / "src" / "enrichment.ttl"
GEO_RDF = RDF_DIR / "ipbes-geo.rdf"

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma")).resolve()
