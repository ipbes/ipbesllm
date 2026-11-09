"""File the figure and table images downloaded by figures_fetch.py and prepare
the spreadsheet that will enrich the TTL.

    python src/figures_build.py

Writes data/figures/<local name of the TTL URI>.<ext>, so the image of
http://ontology.ipbes.net/report/il/TCA24-Figure1.10 is TCA24-Figure1.10.svg
(several images of one item: _1, _2 ...; a part printed in the file name, as
Table 2.6A: _A) and data/figures/figures_enrichment.xlsx. Only SVG, JPEG and PNG
files are filed. Zips are read without unpacking them to disk, and nothing in
them is run or rendered. hasDoi is the concept DOI of the Zenodo record, as in
data/csv/figures_tables.csv.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import rdflib
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from figures_fetch import RECORDS, ROOT, Item, local_path, parse_item, ttl_identifier

OUT = ROOT / "data" / "figures"
WORKBOOK = OUT / "figures_enrichment.xlsx"
CSV_LINKS = ROOT / "data" / "csv" / "figures_tables.csv"
CSV_RESOLVED = ROOT / "config" / "figures_csv_records.json"
IPBES = "http://ontology.ipbes.net/report/"
IMAGE_EXTS = {".svg", ".jpg", ".jpeg", ".png"}
MAX_MEMBER = 500_000_000    # bytes; larger members of a zip are reported, not read
CLIPS_MANIFEST = ROOT / ".cache" / "clips" / "manifest.json"    # figures_clip.py
CODES = "AFA AMA APA BBA ECA GA1 IAS LDR NXS PPA SCM SUA TCA VA".split()

_LABEL = re.compile(
    r"^\s*(?:figure|fig\.?|table|box)\s+(?:spm[ .]*)?[A-Z]?\d*(?:\.\d+)*[A-Za-z]?\s*[.:\-–]?\s*",
    re.IGNORECASE,
)


def file_stem(item: Item, report_id: str) -> str:
    """The local name of the item's TTL URI (TCA24-Figure1.10), and _A for a
    part printed in the file name."""
    return f"{report_id}-{ttl_identifier(item)}" + (f"_{item.letter}" if item.letter else "")


def real_image(data: bytes, ext: str) -> bool:
    if ext == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if ext in (".jpg", ".jpeg"):
        return data.startswith(b"\xff\xd8\xff")
    return b"<svg" in data[:200_000]


def clean_caption(data: bytes) -> str:
    """The caption without its label ("Figure 1.9."), wrapped lines joined."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("cp1252", "replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n", text)]
    return _LABEL.sub("", "\n\n".join(p for p in paragraphs if p), count=1).strip()


def ignored(member: str) -> bool:
    return "__MACOSX" in member or Path(member).name.startswith((".", "~$"))


def collect(records: list[dict]):
    """(images, captions, items without an image, problems). Each image is
    (item, source package, member, ext, bytes, record)."""
    images: dict[Item, list[dict]] = defaultdict(list)
    captions: dict[Item, tuple[str, str, str]] = {}
    other: dict[Item, set[str]] = defaultdict(set)
    records_of: dict[Item, dict] = {}
    problems: list[str] = []

    def add(item, note, record, package, member, ext, data):
        if not real_image(data, ext):
            problems.append(f"{record['id']} {package} {member}: not a real {ext} file, skipped")
            return
        images[item].append(dict(package=package, member=member, ext=".jpg" if ext == ".jpeg" else ext,
                                 data=data, record=record, note=note))

    for record in records:
        code, part = record["code"], record["part"]
        for index, f in enumerate(record["files"]):
            key = f["key"]
            ext = Path(key).suffix.lower()
            item, note = parse_item(key, code, part)
            if item:
                records_of.setdefault(item, record)
            path = local_path(record["id"], index, key)
            if ext in IMAGE_EXTS:
                if not item:
                    problems.append(f"{record['id']} {key}: no figure or table number in the name")
                elif not path.exists():
                    problems.append(f"{record['id']} {key}: not downloaded")
                else:
                    add(item, note, record, key, "", ext, path.read_bytes())
            elif ext == ".zip":
                if not path.exists():
                    problems.append(f"{record['id']} {key}: not downloaded")
                    continue
                try:
                    bundle = zipfile.ZipFile(path)
                except zipfile.BadZipFile:
                    problems.append(f"{record['id']} {key}: not a valid zip")
                    continue
                with bundle:
                    for info in bundle.infolist():
                        member = info.filename
                        if info.is_dir() or ignored(member):
                            continue
                        member_ext = Path(member).suffix.lower()
                        member_item, member_note = item, note
                        if not member_item:
                            member_item, member_note = parse_item(Path(member).name, code, part)
                        if not member_item:
                            if member_ext in IMAGE_EXTS | {".txt"}:
                                problems.append(f"{record['id']} {key} {member}: no figure or table number")
                            continue
                        records_of.setdefault(member_item, record)
                        if info.file_size > MAX_MEMBER:
                            problems.append(f"{record['id']} {key} {member}: over the size limit, skipped")
                        elif member_ext in IMAGE_EXTS:
                            add(member_item, member_note, record, key, member, member_ext, bundle.read(info))
                        elif member_ext == ".txt":
                            captions[member_item] = (clean_caption(bundle.read(info)), key, member)
                        else:
                            other[member_item].add(member_ext.lstrip(".") or "other")
            elif item:
                other[item].add(ext.lstrip("."))
    return images, captions, other, records_of, problems


def load_ttl(code: str) -> tuple[dict, dict, str]:
    """({identifier: (uri, prefLabel, hasDescription, hasDoi)}, {section: uri}, report id)."""
    graph = rdflib.Graph()
    graph.parse(ROOT / "data" / "ttl" / f"{code}.ttl", format="turtle")
    items, sections = {}, {}
    report_id = ""
    for s in graph.subjects(rdflib.RDF.type, rdflib.URIRef(IPBES + "Illustration")):
        report_id = report_id or str(s).rsplit("/", 1)[-1].split("-", 1)[0]
        items[str(s).split("/il/")[-1].split("-", 1)[-1]] = (
            str(s),
            str(graph.value(s, rdflib.SKOS.prefLabel) or ""),
            str(graph.value(s, rdflib.URIRef(IPBES + "hasDescription")) or ""),
            str(graph.value(s, rdflib.URIRef(IPBES + "hasDoi")) or ""),
        )
    for s in graph.subjects(rdflib.RDF.type, rdflib.URIRef(IPBES + "SubChapter")):
        ident = str(graph.value(s, rdflib.DCTERMS.identifier) or "")
        sections[ident] = str(s)
    return items, sections, report_id


def load_placement(code: str) -> dict[str, tuple[str, str]]:
    """{identifier: (page, section)} from the converter's report <A>.md."""
    path = ROOT / "data" / "pdf_compare" / code / f"{code}.md"
    placed = {}
    if path.exists():
        for m in re.finditer(r"^### (Figure|Table|Box) (\S+) \(p(\d+)(?:, in ([^)]+))?\)",
                             path.read_text(encoding="utf-8"), re.MULTILINE):
            placed[f"{m.group(1)}{m.group(2)}"] = (m.group(3), m.group(4) or "")
    return placed


HEADER = Font(name="Arial", bold=True, color="FFFFFF", size=10)
HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
BODY = Font(name="Arial", size=10)


def write_sheet(book: Workbook, title: str, headers: list[tuple[str, int]], rows: list[list]):
    sheet = book.create_sheet(title)
    sheet.append([h for h, _ in headers])
    for column, (_, width) in enumerate(headers, 1):
        cell = sheet.cell(row=1, column=column)
        cell.font, cell.fill = HEADER, HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        sheet.column_dimensions[get_column_letter(column)].width = width
    for row in rows:
        sheet.append(row)
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            cell.font = BODY
            cell.alignment = Alignment(vertical="top")
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    return sheet


def main() -> None:
    records = json.loads(RECORDS.read_text())
    OUT.mkdir(exist_ok=True)
    images, captions, other, records_of, problems = collect(records)
    for old in OUT.iterdir():       # images of an earlier run, as names may have changed
        if old.suffix.lower() in IMAGE_EXTS:
            old.unlink()

    # The DOIs figures_tables.csv gives per assessment and chapter.
    csv_text = CSV_LINKS.read_text(encoding="utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    csv_dois: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in csv.DictReader(io.StringIO(csv_text)):
        part = (row.get("Chapter") or "").strip()
        doi = re.search(r"10\.5281/zenodo\.\d+", row.get("Link File") or "")
        if doi:
            csv_dois[((row.get("Document Symbol") or "").strip(), "SPM" if part in ("0", "SPM") else part)].add(doi.group(0))

    ttl = {code: load_ttl(code) for code in CODES}
    placement = {code: load_placement(code) for code in ttl}

    # Duplicates: one image bytes under one item, once.
    enrichment, used_names = [], set()
    for item in sorted(images, key=lambda i: (i.code, i.part.zfill(3), i.kind, i.number, i.letter)):
        unique, seen = [], set()
        for image in sorted(images[item], key=lambda x: (x["package"], x["member"])):
            if image["data"] not in seen:
                seen.add(image["data"])
                unique.append(image)
        identifier = ttl_identifier(item)
        items, sections, report_id = ttl[item.code]
        uri, label, description, _ = items.get(identifier, ("", "", "", ""))
        page, section = placement[item.code].get(identifier, ("", ""))
        caption, _, caption_member = captions.get(item, ("", "", ""))
        for number, image in enumerate(unique, 1):
            stem = file_stem(item, report_id) + (f"_{number}" if len(unique) > 1 else "")
            name = stem + image["ext"]
            if name in used_names:
                problems.append(f"{name}: name already used")
                continue
            used_names.add(name)
            (OUT / name).write_bytes(image["data"])
            record = image["record"]
            notes = [image["note"]] if image["note"] else []
            if record["conceptdoi"] not in csv_dois.get((item.code, item.part), {record["conceptdoi"]}):
                notes.append(f"figures_tables.csv gives {', '.join(sorted(csv_dois[(item.code, item.part)]))} "
                             f"for this chapter, which is another chapter's record")
            elif (item.code, item.part) not in csv_dois:
                notes.append("chapter not in figures_tables.csv: record found by title on Zenodo")
            if not uri:
                notes.append("not in the TTL")
            if len(unique) > 1:
                notes.append(f"image {number} of {len(unique)} for this item")
            if not caption:
                notes.append("no caption file in the package")
            enrichment.append([
                item.code, item.kind, item.part, identifier, uri, f"data/figures/{name}", image["ext"].lstrip("."),
                len(image["data"]), f"https://doi.org/{record['conceptdoi']}", record["id"], record.get("license") or "",
                image["package"], image["member"], section,
                sections.get(section, ""), page, label, caption, caption_member, description,
                "; ".join(notes), "Zenodo",
            ])

    # Figures and tables with no Zenodo image, clipped from the report PDFs
    # by figures_clip.py (which leaves out those with a Zenodo image).
    have = {(row[0], row[3]) for row in enrichment}
    clips = json.loads(CLIPS_MANIFEST.read_text()) if CLIPS_MANIFEST.exists() else {}
    clipped = set()
    for name, clip in sorted(clips.items()):
        code, identifier = clip["code"], clip["ident"]
        if not clip["file"] or (code, identifier) in have or code not in ttl:
            continue
        items, sections, report_id = ttl[code]
        uri, label, description, doi = items.get(identifier, ("", "", "", ""))
        _, section = placement[code].get(identifier, ("", ""))
        data = (CLIPS_MANIFEST.parent / clip["file"]).read_bytes()
        (OUT / clip["file"]).write_bytes(data)
        clipped.add((code, identifier))
        part = "SPM" if "SPM" in identifier else re.search(r"(\d+)\.", identifier).group(1)
        notes = ["clipped from the report PDF: check it"] + clip["flags"]
        enrichment.append([
            code, clip["kind"], part, identifier, uri, f"data/figures/{clip['file']}", "png", len(data),
            f"https://doi.org/{doi}" if doi else "", "", "", f"report PDF ({code})", f"page {clip['page']}",
            section, sections.get(section, ""), clip["page"], label, "", "", description,
            "; ".join(notes), "report PDF",
        ])

    # Items with files in Zenodo but no image in the wanted formats.
    no_image = []
    for item in sorted(set(other) | set(records_of), key=lambda i: (i.code, i.part.zfill(3), i.kind, i.number, i.letter)):
        if item in images or (item.code, ttl_identifier(item)) in clipped:
            continue
        identifier = ttl_identifier(item)
        record = records_of[item]
        caption = captions.get(item, ("",))[0]
        pdf_or_other = sorted(other.get(item, set()))
        section = placement[item.code].get(identifier, ("", ""))[1]
        no_image.append([item.code, item.kind, item.part, identifier,
                         ttl[item.code][0].get(identifier, ("",))[0], ", ".join(pdf_or_other),
                         f"https://doi.org/{record['conceptdoi']}", record["id"], section, caption])
    # Items figures_clip.py could not clip either: to be clipped by hand.
    listed = {(row[0], row[3]) for row in no_image}
    for clip in clips.values():
        code, identifier = clip["code"], clip["ident"]
        if clip["file"] or (code, identifier) in listed or code not in ttl:
            continue
        uri, _, _, doi = ttl[code][0].get(identifier, ("", "", "", ""))
        part = "SPM" if "SPM" in identifier else re.search(r"(\d+)\.", identifier).group(1)
        no_image.append([code, clip["kind"], part, identifier, uri,
                         f"not clipped from the report PDF (page {clip['page']}): {'; '.join(clip['flags'])}",
                         f"https://doi.org/{doi}" if doi else "", "",
                         placement[code].get(identifier, ("", ""))[1], ""])

    # Which record each row of figures_tables.csv points at.
    by_id = {r["id"]: r for r in records}
    resolved = json.loads(CSV_RESOLVED.read_text()) if CSV_RESOLVED.exists() else {}
    linked = []
    linked_ids = set()
    for row in csv.DictReader(io.StringIO(csv_text)):
        doi = (row.get("Link File") or "").strip()
        code = (row.get("Document Symbol") or "").strip()
        part = (row.get("Chapter") or "").strip()
        part = "SPM" if part in ("0", "SPM") else part
        old = re.search(r"zenodo\.(\d+)", doi).group(1) if "zenodo." in doi else ""
        target = resolved.get(old)
        record = by_id.get(target) if target else None
        linked_ids.add(target)
        match = "yes" if record and record["part"] == part and record["code"] == code else "NO"
        right = [r for r in records if r["code"] == code and r["part"] == part]
        linked.append([code, part, doi, record["title"] if record else "", match,
                       "; ".join(f"https://doi.org/{r['conceptdoi']}" for r in right) or "none found on Zenodo"])
    found = [[r["code"], r["part"], r["title"], f"https://doi.org/{r['conceptdoi']}",
              f"https://doi.org/{r['doi']}", r.get("license") or "",
              "linked in the CSV" if r["id"] in linked_ids else "found by title search on Zenodo"]
             for r in records]

    book = Workbook()
    readme = book.active
    readme.title = "Readme"
    lines = [
        ("IPBES figures and tables: data to enrich the TTL", True),
        ("", False),
        ("Sheets", True),
        ("Enrichment: one row per image filed in data/figures/. This is the sheet to load. source says where the "
         "image comes from: Zenodo, or the report PDF (clipped by figures_clip.py; check those, their flags are in notes).", False),
        ("No image: items on Zenodo with no SVG, JPEG or PNG (PDF, Word, .ai or .indd only) that could not be clipped either.", False),
        ("Coverage: per assessment, what the TTL lists against what was filed. Counts are formulas.", False),
        ("CSV links: each row of data/csv/figures_tables.csv, and whether its DOI is the chapter it says.", False),
        ("Records: the Zenodo records used, and how each was found.", False),
        ("Problems: anything skipped while filing.", False),
        ("", False),
        ("Enrichment columns", True),
        ("class: ipbes:Figure or ipbes:Table, subclasses of ipbes:Illustration (ontology v07, change 17). "
         "The files type them as ipbes:Illustration only, until they are converted again.", False),
        ("ttl_uri: the existing Illustration; empty if the TTL has no item with that identifier.", False),
        ("zenodo_doi: the value for ipbes:hasDoi, the concept DOI of the Zenodo record the image came from, the form "
         "data/csv/figures_tables.csv uses. It resolves to the latest version and identifies the chapter's record, not this item alone.", False),
        ("subchapter, subchapter_uri, page: from the converter's report data/pdf_compare/<A>/<A>.md.", False),
        ("hasDescription: the caption file in the package, without its label, wrapped lines joined.", False),
        ("ttl_prefLabel, ttl_hasDescription: what the TTL holds now, to compare.", False),
        ("", False),
        ("File names", True),
        ("The local name of the item's TTL URI: the image of http://ontology.ipbes.net/report/il/TCA24-Figure1.10 "
         "is TCA24-Figure1.10.svg, of .../il/IAS23-TableSPM.A1 IAS23-TableSPM.A1.svg.", False),
        ("_1, _2 ... mark several images of one item; _A, _B a part printed in the Zenodo file name (VA22-Table2.6_A.jpg).", False),
        ("An item missing from the TTL is named the same way, from its report code and number (see notes).", False),
    ]
    for line, bold in lines:
        readme.append([line])
        readme.cell(row=readme.max_row, column=1).font = Font(name="Arial", size=10 if not bold else 11, bold=bold)
    readme.column_dimensions["A"].width = 110

    write_sheet(book, "Enrichment", [
        ("assessment", 11), ("class", 8), ("chapter", 8), ("ttl_identifier", 15), ("ttl_uri", 34), ("image_file", 28),
        ("format", 7), ("bytes", 10), ("zenodo_doi", 36), ("zenodo_record", 12), ("license", 10), ("package", 26),
        ("package_member", 26), ("subchapter", 10), ("subchapter_uri", 34), ("page", 6), ("ttl_prefLabel", 40),
        ("hasDescription", 70), ("caption_file", 26), ("ttl_hasDescription", 50), ("notes", 34),
        ("source", 12)], enrichment)
    write_sheet(book, "No image", [
        ("assessment", 11), ("class", 8), ("chapter", 8), ("ttl_identifier", 15), ("ttl_uri", 34), ("formats_available", 16),
        ("zenodo_doi", 36), ("zenodo_record", 12), ("subchapter", 10), ("hasDescription", 70)], no_image)

    coverage = book.create_sheet("Coverage")
    headers = ["assessment", "class", "items_in_ttl", "images_filed", "items_with_image_in_ttl",
               "with_caption", "no_image_on_zenodo", "from_zenodo", "clipped_from_pdf"]
    coverage.append(headers)
    row = 2
    counts = {}
    for code, (items, _, _) in ttl.items():
        for kind in ("Figure", "Table"):
            counts[(code, kind)] = sum(i.startswith(kind) for i in items)
    for (code, kind), total in sorted(counts.items()):
        coverage.append([
            code, kind, total,
            f"=COUNTIFS(Enrichment!$A:$A,$A{row},Enrichment!$B:$B,$B{row})",
            f'=COUNTIFS(Enrichment!$A:$A,$A{row},Enrichment!$B:$B,$B{row},Enrichment!$E:$E,"?*")',
            f'=COUNTIFS(Enrichment!$A:$A,$A{row},Enrichment!$B:$B,$B{row},Enrichment!$R:$R,"?*")',
            f"=COUNTIFS('No image'!$A:$A,$A{row},'No image'!$B:$B,$B{row})",
            f'=COUNTIFS(Enrichment!$A:$A,$A{row},Enrichment!$B:$B,$B{row},Enrichment!$V:$V,"Zenodo")',
            f'=COUNTIFS(Enrichment!$A:$A,$A{row},Enrichment!$B:$B,$B{row},Enrichment!$V:$V,"report PDF")',
        ])
        row += 1
    coverage.append(["Total", ""] + [f"=SUM({c}2:{c}{row - 1})" for c in "CDEFGHI"])
    for r in coverage.iter_rows():
        for cell in r:
            cell.font = BODY
    for cell in coverage[1]:
        cell.font, cell.fill = HEADER, HEADER_FILL
        coverage.column_dimensions[cell.column_letter].width = 16
    for cell in coverage[row]:
        cell.font = Font(name="Arial", size=10, bold=True)
    coverage.freeze_panes = "A2"
    coverage.cell(row=row + 2, column=1, value="items_in_ttl is counted when this file was built; the others are formulas "
                  "over the Enrichment and No image sheets. images_filed can exceed items where one item has several images.").font = BODY

    write_sheet(book, "CSV links", [("assessment", 11), ("chapter", 8), ("doi_in_csv", 38), ("record_title_at_that_doi", 90),
                                    ("is_that_chapter", 12), ("right_record", 60)], linked)
    write_sheet(book, "Records", [("assessment", 11), ("chapter", 8), ("title", 90), ("concept_doi (hasDoi)", 36),
                                  ("latest_version_doi", 36), ("license", 10), ("found", 30)], found)
    write_sheet(book, "Problems", [("problem", 130)], [[p] for p in problems] or [["none"]])
    book.save(WORKBOOK)

    print(f"images filed: {len(enrichment)}  ({Counter(r[6] for r in enrichment)})")
    print(f"in the TTL: {sum(bool(r[4]) for r in enrichment)}; with caption: {sum(bool(r[17]) for r in enrichment)}; "
          f"with subchapter: {sum(bool(r[13]) for r in enrichment)}")
    print(f"items without an image: {len(no_image)}; problems: {len(problems)}")
    print(f"wrote {WORKBOOK}")


if __name__ == "__main__":
    main()
