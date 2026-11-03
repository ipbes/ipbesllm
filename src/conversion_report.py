"""A PDF report on one assessment's conversion by pdf_to_ttl.py, for the
assessment's technical support unit and the data team.

    PYTHONPATH=src python src/conversion_report.py BBA
    PYTHONPATH=src python src/conversion_report.py --all

It is built from what the conversion wrote to data/pdf_compare/<A>/:

  <A>_conversion.md   the counts, and what could not be read or matched
                      or where the report's own sources disagree
  <A>_compare.md      the comparison of the manual conversion
                      (data/ttl_original/<A>_v*.ttl) with the automatic one
                      (data/ttl/<A>_v00.ttl), and the fidelity of both to
                      the PDF; made here with ttl_compare.py when a manual
                      conversion exists

and written as <A>_conversion_report.pdf next to them. Run pdf_to_ttl.py
first. Every number comes from those files; nothing is written by hand.
"""
from __future__ import annotations

import argparse
import datetime
import html
import re
import sys
import tomllib
from pathlib import Path

import pymupdf

from pdf_to_ttl import DEFAULT_CONFIG, DEFAULT_OUT, ROOT

ORIGINAL_DIR = ROOT / "data" / "ttl_original"     # manual conversions
AUTOMATIC_DIR = ROOT / "data" / "ttl"               # automatic ones, <A>_v00.ttl, indexed
EXAMPLES = 4        # examples per finding in section 5

# The text classes whose fidelity the summary sums up.
TEXT_PROPERTIES = {"hasDescription"}

# What a count of zero means for the reader: the layout was not recognised.
ESSENTIAL = ["Chapters", "Sub-chapters", "Key messages", "People",
             "Reference entries in the PDF"]

CSS = """
* { font-family: sans-serif; }
body { font-size: 9.5pt; line-height: 1.35; color: #1f2328; }
h1 { font-size: 19pt; margin: 0 0 4pt 0; color: #037473; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt 0; color: #037473; }
h3 { font-size: 10.5pt; margin: 8pt 0 3pt 0; }
p { margin: 0 0 5pt 0; }
.kicker { font-size: 8pt; color: #59636e; letter-spacing: 1pt; }
.lede { font-size: 10.5pt; color: #3d444d; margin-bottom: 8pt; }
.warn { color: #9a3412; font-weight: bold; }
.note { font-size: 8.5pt; color: #59636e; }
table { border-collapse: collapse; width: 100%; margin: 3pt 0 8pt 0; }
th { background-color: #e6f2f2; text-align: left; font-size: 8.5pt; padding: 2pt 4pt; }
td { font-size: 8.5pt; padding: 2pt 4pt; border-bottom: 0.5pt solid #d1d9e0; vertical-align: top; }
td.n, th.n { text-align: right; }
ul { margin: 0 0 6pt 0; }
li { margin-bottom: 2pt; }
code { font-family: monospace; font-size: 8pt; }
"""


# ---------------------------------------------------------------------------
# Reading the conversion's files
# ---------------------------------------------------------------------------

def md_tables(text: str) -> dict[str, list[list[str]]]:
    """The first table under each "## " heading, as rows of cells (the
    header row first, the |---| row left out)."""
    tables: dict[str, list[list[str]]] = {}
    heading = None
    for line in text.splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
        elif line.startswith("|") and heading and not re.match(r"^\|[-:| ]+\|$", line):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            tables.setdefault(heading, []).append(cells)
        elif heading in tables and not line.strip():
            heading = None      # only the first table
    return tables


def read_log(path: Path) -> tuple[dict[str, str], list[tuple[str, list[str], dict[str, list[str]]]]]:
    """(counts, [(topic, items, {kind: items})]) from <A>_conversion.md, whose
    topics ("## ") group their lines by kind ("### ")."""
    counts: dict[str, str] = {}
    topics: list[tuple[str, list[str], dict[str, list[str]]]] = []
    kind = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if m := re.match(r"^\| ( *.+?)\s*\|\s*([\d.]+)\s*\|$", line):
            counts[m.group(1)] = m.group(2)
        elif line.startswith("## "):
            topics.append((re.sub(r"\s*\(\d+\)$", "", line[3:]), [], {}))
            kind = ""
        elif line.startswith("### ") and topics:
            kind = re.sub(r"\s*\(\d+\)$", "", line[4:])
        elif line.startswith("- ") and topics:
            topics[-1][1].append(line[2:])
            if kind:
                topics[-1][2].setdefault(kind, []).append(line[2:])
    return counts, topics


def original_ttl(assessment: str) -> Path | None:
    """The latest manual conversion, data/ttl_original/<A>_v<NN>.ttl."""
    files = sorted(ORIGINAL_DIR.glob(f"{assessment}_v*.ttl"),
                   key=lambda p: int(re.sub(r"\D", "", p.stem.split("_v")[-1]) or 0))
    return files[-1] if files else None


def run_compare(original: Path, ttl: Path, pdf: Path, out: Path) -> None:
    import ttl_compare
    ttl_compare.main([str(original), str(ttl), "--pdf", str(pdf), "--out", str(out)])


def percent(found: int, total: int) -> str:
    """Whole percentages, but never rounded up to 100% or down to 0%."""
    if not total:
        return "–"
    share = 100 * found / total
    if share == round(share) or 1 <= share <= 99:
        return f"{share:.0f}%"
    return f"{share:.1f}%"


def fraction(cell: str) -> tuple[int, int] | None:
    """ "39% (1569/4053)" -> (1569, 4053)."""
    m = re.search(r"\((\d+)/(\d+)\)", cell)
    return (int(m.group(1)), int(m.group(2))) if m else None


def number(value: str) -> str:
    """1992 -> "1,992"; other text as it is."""
    return f"{int(value):,}" if value.isdigit() else value


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def esc(text) -> str:
    return html.escape(str(text))


def table(header: list[str], rows: list[list], numeric_from: int = 1) -> str:
    def cell(tag, i, value):
        cls = ' class="n"' if i >= numeric_from else ""
        return f"<{tag}{cls}>{esc(value)}</{tag}>"
    out = ["<table><tr>", *(cell("th", i, h) for i, h in enumerate(header)), "</tr>"]
    for row in rows:
        out += ["<tr>", *(cell("td", i, v) for i, v in enumerate(row)), "</tr>"]
    out.append("</table>")
    return "".join(out)


def build_html(assessment: str, cfg: dict, folder: Path, today: str) -> str:
    pdf = ROOT / cfg["pdf"]
    ttl = folder / f"{assessment}_pdf.ttl"
    counts, topics = read_log(folder / f"{assessment}_conversion.md")
    compare_md = folder / f"{assessment}_compare.md"
    original = original_ttl(assessment)
    tables = md_tables(compare_md.read_text(encoding="utf-8")) if original and compare_md.exists() else {}
    with pymupdf.open(pdf) as doc:
        pages = doc.page_count

    c = {k.strip(): v for k, v in counts.items()}
    refs, matched, linked = (int(c.get(k, 0)) for k in
                             ("Reference entries in the PDF", "matched to Zotero",
                              "linked to a sub-chapter"))
    terms, known = int(c.get("Glossary terms (Annex I)", 0)), int(c.get("in the IPBES thesaurus", 0))
    missing = [k for k in ESSENTIAL if c.get(k, "0") == "0"]

    h = [f'<p class="kicker">IPBES KNOWLEDGE BASE · DATA CONVERSION</p>',
         f"<h1>{esc(assessment)}: conversion from PDF to TTL</h1>",
         f'<p class="lede">{esc(cfg.get("alt_label", assessment))} ({cfg.get("year", "")}), '
         f"converted by pdf_to_ttl.py from the published report.</p>"]
    h.append(table(["", ""], [
        ["Date", today],
        ["Source", f"{pdf.name} ({pages} pages)"],
        ["Report code", cfg.get("report_id", "")],
        ["Zotero group", cfg.get("zotero_group", "none")],
        ["Output", f"data/pdf_compare/{assessment}/{ttl.name}"],
        ["Compared with", f"data/ttl_original/{original.name} (manual conversion)"
         if original else "nothing: there is no earlier conversion"],
    ], numeric_from=9))

    # 1. Summary
    h.append("<h2>Summary</h2><ul>")
    if missing:
        h.append(f'<li class="warn">The conversion is incomplete: no {", no ".join(m.lower() for m in missing)} '
                 f"were read. The report's layout is not yet recognised; this file must not be indexed.</li>")
    h.append(f"<li>{number(c.get('Triples', '0'))} triples: {c.get('Chapters', 0)} chapters, "
             f"{c.get('Sub-chapters', 0)} sub-chapters, {c.get('Key messages', 0)} key messages, "
             f"{c.get('Background messages', 0)} background messages, "
             f"{c.get('Figures, tables and boxes', 0)} figures, tables and boxes, "
             f"{c.get('Knowledge gaps', 0)} knowledge gaps and {c.get('People', 0)} people.</li>")
    if refs:
        h.append(f"<li>References: {matched:,} of the {refs:,} reference-list entries ({percent(matched, refs)}) "
                 f"are matched to their Zotero item; {percent(linked, refs)} are linked to the "
                 f"sub-chapters that cite them.</li>")
    fid = tables.get("Fidelity to the PDF", [])
    if fid:
        sums = [[0, 0], [0, 0]]
        for row in fid[1:]:
            if row[1] in TEXT_PROPERTIES:
                for i, cell in enumerate(row[2:4]):
                    if f := fraction(cell):
                        sums[i][0] += f[0]
                        sums[i][1] += f[1]
        h.append(f"<li>Fidelity: {percent(*sums[1])} of the descriptive text in the automatic file is found word "
                 f"for word in the PDF, against {percent(*sums[0])} in the manual conversion "
                 f"(section 3).</li>")
    findings = sum(len(items) for _, items, _ in topics)
    if findings:
        h.append(f"<li>The conversion log lists {findings:,} points to check, most of them where the "
                 f"report's own sources disagree (section 5).</li>")
    if terms:
        h.append(f"<li>Glossary: {known} of the {terms} terms of the glossary annex are in the IPBES "
                 f"thesaurus.</li>")
    h.append("</ul>")
    if cfg.get("notes"):
        h.append("<h3>Notes on this assessment</h3><ul>")
        h += [f"<li>{esc(n)}</li>" for n in cfg["notes"]]
        h.append("</ul>")

    # 2. Counts
    h.append("<h2>1. What the automatic file contains</h2>")
    # The log indents sub-counts ("  matched to Zotero").
    h.append(table(["Item", "Count"], [["– of which " + k.strip() if k.startswith(" ") else k, number(v)]
                                        for k, v in counts.items()]))

    # 3 and 4. Against the manual conversion
    if tables:
        h.append("<h2>2. Comparison with the manual conversion</h2>")
        h.append('<p>Items are paired by their identifiers (sub-chapter 1.1, key message KM1, Zotero key), '
                 "people by name and sub-messages by their text.</p>")
        items = tables.get("Items", [])
        if items:
            h.append(table(["Class", "Manual", "Automatic", "In both", "Only manual", "Only automatic", "Within the other"], items[1:]))
        if fid:
            h.append("<h2>3. Fidelity to the published text</h2>")
            h.append("<p>For each text property, the share of sentences (40 characters or more) found word for "
                     "word in the PDF. A sentence is missed if a comma, a word or a character differs.</p>")
            h.append(table(["Class", "Property", "Manual", "Automatic"], fid[1:], numeric_from=2))
        props = tables.get("Properties of paired items", [])
        if props:
            h.append("<h2>4. Properties of paired items</h2>")
            h.append("<p>How many values are the same in both files, differ only in spaces, punctuation or "
                     "case, differ, or are in one file only. Every difference is listed in "
                     f"{esc(assessment)}_compare.csv.</p>")
            h.append(table(["Class", "Property", "Same", "Spaces/punct.", "Different", "Only manual",
                            "Only automatic"], props[1:], numeric_from=2))

    # 5. Findings
    h.append("<h2>5. Points to check</h2>")
    if topics:
        h.append(f"<p>From {esc(assessment)}_conversion.md, which lists each with its page. Most are in the "
                 "published report or the supporting data, not in the conversion, and are worth reporting to "
                 "the assessment's technical support unit.</p>")
        for topic, entries, kinds in topics:
            h.append(f"<h3>{esc(topic)} ({len(entries)})</h3><ul>")
            if kinds:
                # Each kind with its count and one example.
                h += [f"<li><b>{esc(k)} ({len(v)})</b>, e.g. {esc(v[0])}</li>" for k, v in kinds.items()]
            else:
                h += [f"<li>{esc(e)}</li>" for e in entries[:EXAMPLES]]
                if len(entries) > EXAMPLES:
                    h.append(f'<li class="note">… and {len(entries) - EXAMPLES} more in the log.</li>')
            h.append("</ul>")
    else:
        h.append("<p>None.</p>")

    # 6. Running it
    h.append("<h2>6. Running it</h2><p>")
    commands = [f"PYTHONPATH=src python src/pdf_to_ttl.py {assessment}"]
    if original:
        commands.append(f"PYTHONPATH=src python src/ttl_compare.py data/ttl_original/{original.name} "
                        f"data/ttl/{assessment}_v00.ttl --pdf {cfg['pdf']}")
    commands += [f"PYTHONPATH=src python src/ttl_audit.py data/pdf_compare/{assessment}/{ttl.name}",
                 f"PYTHONPATH=src python src/conversion_report.py {assessment}"]
    h.append("<br>".join(f"<code>{esc(cmd)}</code>" for cmd in commands) + "</p>")
    files = [[f"{assessment}_pdf.ttl", "The TTL"],
             [f"{assessment}.md", "The report as read from the PDF, with page numbers"],
             [f"{assessment}_conversion.md", "What could not be read or matched, and where the sources disagree"]]
    if tables:
        files.append([f"{assessment}_compare.md / .csv", "The manual file compared with the automatic one; the CSV lists "
                                                         "every difference"])
    files.append([f"{assessment}_zotero.json", "The Zotero library, cached"])
    h.append(table([f"File in data/pdf_compare/{assessment}/", "Content"], files, numeric_from=9))
    return "\n".join(h)


def render(body: str, path: Path, header: str) -> None:
    """HTML to an A4 PDF, with a running header and page numbers."""
    story = pymupdf.Story(html=f"<body>{body}</body>", user_css=CSS)
    page_rect = pymupdf.paper_rect("a4")
    frame = page_rect + (54, 60, -54, -54)
    tmp = path.with_suffix(".tmp.pdf")
    writer = pymupdf.DocumentWriter(str(tmp))
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(frame)
        story.draw(device)
        writer.end_page()
    writer.close()

    doc = pymupdf.open(tmp)
    for page in doc:
        page.insert_text((54, 36), header, fontsize=7.5, color=(0.35, 0.39, 0.43))
        page.insert_text((page_rect.width - 64, 36), str(page.number + 1), fontsize=7.5,
                         color=(0.35, 0.39, 0.43))
    doc.set_metadata({"title": header, "author": "IPBES knowledge base",
                      "subject": "Converting an assessment report PDF to TTL"})
    doc.save(path)
    doc.close()
    tmp.unlink()


def report(assessment: str, config: Path = DEFAULT_CONFIG, compare: bool = True) -> Path:
    with config.open("rb") as f:
        cfg = tomllib.load(f)["assessments"][assessment]
    folder = DEFAULT_OUT / assessment
    ttl = folder / f"{assessment}_pdf.ttl"
    if not ttl.exists():
        raise SystemExit(f"{assessment}: no {ttl.relative_to(ROOT)}; run pdf_to_ttl.py {assessment} first")
    original = original_ttl(assessment)
    compare_md = folder / f"{assessment}_compare.md"
    # Compared: the file that is indexed, data/ttl/<A>_v00.ttl, when it is there.
    automatic = AUTOMATIC_DIR / f"{assessment}_v00.ttl"
    automatic = automatic if automatic.exists() else ttl
    if compare and original and (not compare_md.exists()
                                 or compare_md.stat().st_mtime < automatic.stat().st_mtime):
        run_compare(original, automatic, ROOT / cfg["pdf"], folder / f"{assessment}_compare")
    today = datetime.date.today().strftime("%-d %B %Y")
    out = folder / f"{assessment}_conversion_report.pdf"
    render(build_html(assessment, cfg, folder, today), out,
           f"{assessment}: PDF to TTL conversion · {today}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("assessments", nargs="*", help="as in config/pdf_to_ttl.toml, e.g. BBA")
    ap.add_argument("--all", action="store_true", help="every assessment converted so far")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--no-compare", action="store_true",
                    help="use the existing <A>_compare.md, even if older than the TTL")
    args = ap.parse_args(argv)
    names = args.assessments
    if args.all:
        names = sorted(p.name for p in DEFAULT_OUT.iterdir() if (p / f"{p.name}_pdf.ttl").exists())
    if not names:
        ap.error("name an assessment, or use --all")
    for name in names:
        out = report(name, args.config, compare=not args.no_compare)
        print(f"{name} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
