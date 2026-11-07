"""Clip the figures and tables that have no image on Zenodo from the report PDFs.

    python src/figures_clip.py --extract AFA AMA ...   # slow: read the layout (pdf_to_ttl's parser)
    python src/figures_clip.py                         # clip every report with a layout
    python src/figures_build.py                        # file them with the Zenodo images

--extract keeps, per report, each figure's and table's page and caption box,
and the text blocks of those pages with their role, in .cache/clips/<A>_layout.json.
Clipping then reads the page's graphics: from the caption, it takes the side
(above or below) with the most graphics, up to the nearest body text, heading,
footnote or other caption, and clips the graphics and figure text in between.
The clips go to .cache/clips/<URI local name>.png (IAS23-Table1.1.png) with
.cache/clips/manifest.json; a contact sheet per report, .cache/clips/<A>_review.pdf,
shows each clip with its flags for review.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / ".cache" / "clips"
FIGURES = ROOT / "data" / "figures"
CODES = "AFA AMA APA BBA ECA GA1 IAS LDR NXS PPA SCM SUA TCA VA".split()
IPBES = "http://ontology.ipbes.net/report/"
# Text that bounds a figure: running text, headings, footnotes, other captions.
STOP_ROLES = {"body", "h2", "h3", "h4", "h4b", "title", "caption", "footnote"}
DPI = 150


def extract(code: str) -> None:
    """Run the converter's parser once and keep what clipping needs."""
    import pdf_to_ttl as p

    cfg, _ = p.load_config(p.DEFAULT_CONFIG, code)
    doc = pymupdf.open(p.ROOT / cfg["pdf"])
    paras = p.build_paras(p.read_lines(doc))
    rep = p.parse(doc, paras)
    items = [{"ident": il.ident, "kind": il.kind, "page": il.page, "bbox": list(il.bbox)}
             for il in rep.illustrations if il.kind in ("Figure", "Table") and il.bbox]
    pages = {i["page"] for i in items}
    blocks = [{"page": q.page, "role": q.role, "bbox": [q.x0, q.y0, q.x1, q.y1]}
              for q in paras if q.page in pages and q.lines]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{code}_layout.json").write_text(json.dumps(
        {"pdf": cfg["pdf"], "report_id": cfg["report_id"], "items": items, "blocks": blocks}))
    print(f"{code}: {len(items)} figures and tables on {len(pages)} pages", flush=True)


def missing_items(code: str, report_id: str) -> set[str]:
    """Identifiers of the TTL's figures and tables with no Zenodo image in
    data/figures/. The clips filed there by an earlier run do not count."""
    import rdflib

    graph = rdflib.Graph()
    graph.parse(ROOT / "data" / "ttl" / f"{code}_v00.ttl", format="turtle")
    manifest = OUT / "manifest.json"
    clipped = {v["file"] for v in json.loads(manifest.read_text()).values()} if manifest.exists() else set()
    have = {re.sub(r"(_[A-Z0-9]+)?\.(svg|jpg|png)$", "", f.name) for f in FIGURES.glob(f"{report_id}-*")
            if f.suffix in (".svg", ".jpg", ".png") and f.name not in clipped}
    out = set()
    for kind in ("Figure", "Table"):
        for s in graph.subjects(rdflib.RDF.type, rdflib.URIRef(IPBES + kind)):
            local = str(s).rsplit("/", 1)[-1]
            if local not in have:
                out.add(local.split("-", 1)[1])
    return out


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _prose(page: pymupdf.Page, cap: pymupdf.Rect) -> list[pymupdf.Rect]:
    """Text blocks of running prose: three lines or more, of full-length lines
    (a legend, a box's text). Figure labels are short."""
    out = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        lines = ["".join(s["text"] for s in line["spans"]).strip() for line in block["lines"]]
        lines = [x for x in lines if x]
        r = pymupdf.Rect(block["bbox"])
        if len(lines) >= 3 and r.width > 140 and sum(len(x) for x in lines) / len(lines) > 35 \
                and not r.intersects(cap):
            out.append(r)
    return out


def region_for(page: pymupdf.Page, cap: pymupdf.Rect, blocks: list[dict], kind: str = "Figure"):
    """(region, side, flags) of the figure or table captioned `cap`."""
    W, H = page.rect.width, page.rect.height
    top, bottom = 0.045 * H, 0.955 * H          # running header and footer
    rects = [(b["role"], pymupdf.Rect(b["bbox"])) for b in blocks]
    stops = [r for role, r in rects if role in STOP_ROLES and not r.intersects(cap)]
    others = [r for role, r in rects if role not in STOP_ROLES and not r.intersects(cap)]

    drawings = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
    images = [pymupdf.Rect(i["bbox"]) & page.rect for i in page.get_image_info()]

    def keep(g: pymupdf.Rect) -> bool:
        """Not a page background, header or footer, a tab at the page's edge
        (chapter tabs, page-number blocks), nor part of the caption (its label
        badge); a photo may touch the caption by a point or two."""
        overlap = (g & cap).get_area() if g.intersects(cap) else 0.0
        edge_tab = (g.x0 < 0.02 * W or g.x1 > 0.98 * W) and g.width < 0.2 * W
        return (g.width * g.height < 0.8 * W * H and (g.width > 2 or g.height > 2) and not edge_tab
                and g.y1 > top and g.y0 < bottom and overlap <= 0.3 * max(g.get_area(), 1e-6))

    drawings, images = [g for g in drawings if keep(g)], [g for g in images if keep(g)]
    below_from = cap.y1     # where the band below starts: after the caption and its legend
    if kind == "Figure":
        # A figure is graphics and their labels. Prose near it (the legend of
        # the figure above, a box's text) bounds it, and a shaded background
        # behind such prose is not part of it. Its own legend, printed right
        # under the caption (often on the figure's panel, AFA), belongs to the
        # caption: the band below starts after it. Tables are text in cells,
        # so they keep the text between the caption and the next stop.
        prose = _prose(page, cap)
        # The rest of a title printed on two lines, which the caption box from
        # the converter may leave out (older reports): any text right under it.
        lines = [pymupdf.Rect(b["bbox"]) for b in page.get_text("dict")["blocks"] if b.get("type") == 0]
        follow = True
        while follow:
            follow = next((r for r in lines if below_from - 2 <= r.y0 <= below_from + 4 and r.y1 > below_from
                           and _overlap(r.x0, r.x1, cap.x0, cap.x1) > 0.5 * min(r.width, cap.width)
                           and r not in prose), None)
            if follow:
                below_from = follow.y1
                lines.remove(follow)
        legend = True
        while legend:
            legend = next((p for p in prose if below_from - 3 <= p.y0 <= below_from + 14
                           and _overlap(p.x0, p.x1, cap.x0, cap.x1) > 0), None)
            if legend:
                below_from = legend.y1
                prose.remove(legend)
        # Only a vector shape can be a background; a photo's box can reach under
        # nearby text when the photo is cropped.
        drawings = [g for g in drawings if not (g.width * g.height > 0.02 * W * H
                                                and any(g.contains(p.tl + (p.br - p.tl) * 0.5) for p in prose))]
        stops = stops + prose
        others = [r for r in others if not any(r.intersects(p) for p in prose) and r.y0 >= below_from - 1
                  or r.y1 <= cap.y0 + 1]
    graphics = drawings + images

    def band(side: str) -> pymupdf.Rect:
        """From the caption to the nearest stopping text in the caption's
        columns (text beside a half-width figure does not stop it)."""
        in_column = [r for r in stops if _overlap(r.x0, r.x1, cap.x0, cap.x1) > 0.2 * min(r.width, cap.width)]
        if side == "below":
            ends = [r.y0 for r in in_column if r.y0 >= below_from - 1]
            return pymupdf.Rect(0, below_from, W, min(ends, default=bottom))
        ends = [r.y1 for r in in_column if r.y1 <= cap.y0 + 1]
        return pymupdf.Rect(0, max(ends, default=top), W, cap.y0)

    def contents(b: pymupdf.Rect) -> tuple[list, float]:
        inside = [g & b for g in graphics if g.intersects(b)]
        inside = [g for g in inside if not g.is_empty]
        return inside, sum(g.width * g.height for g in inside)

    below, above = band("below"), band("above")
    (g_below, a_below), (g_above, a_above) = contents(below), contents(above)
    flags = []
    if a_below == a_above == 0 and kind == "Figure":
        return None, "", ["no graphics next to the caption"]    # printed on another page
    if a_below == a_above == 0:
        # A table of text only: its rows follow the caption.
        side, b, inside = "below", below, []
        flags.append("no graphics")
    elif a_below >= a_above:
        side, b, inside = "below", below, g_below
    else:
        side, b, inside = "above", above, g_above
    texts = [r & b for r in others if r.intersects(b)]
    if kind == "Figure" and inside:
        # Only labels next to the graphics, not stray text elsewhere in the band.
        hull = pymupdf.Rect(inside[0])
        for g in inside[1:]:
            hull |= g
        texts = [r for r in texts if r.intersects(hull + (-12, -12, 12, 12))]
    parts = [r for r in inside + texts if not r.is_empty]
    if not parts:
        return None, side, ["nothing found next to the caption"]
    region = pymupdf.Rect(parts[0])
    for r in parts[1:]:
        region |= r
    region &= b
    area = region.width * region.height / (W * H)
    if area < 0.03 and kind == "Figure":
        return None, side, ["only a sliver next to the caption"]   # printed on another page
    if area < 0.03:
        flags.append("very small")
    if side == "below" and b.y1 >= bottom - 1 and region.y1 > bottom - 30:
        flags.append("reaches the foot of the page: may go on overleaf")
    return region, side, flags


def overleaf(doc: pymupdf.Document, page_number: int,
             captioned: set[int]) -> tuple[pymupdf.Page, pymupdf.Rect] | None:
    """A figure printed on the page facing its caption: the graphics at the top
    of that page, above any text, if they fill a good part of it. The previous
    page first (NXS prints the caption at the top of the right-hand page and the
    figure on the left), then the next. A page with a figure or table caption of
    its own, or already given to another figure, is not one."""
    for other in (page_number - 1, page_number + 1):
        if not 1 <= other <= doc.page_count or other in captioned:
            continue
        page = doc[other - 1]
        W, H = page.rect.width, page.rect.height
        prose = _prose(page, pymupdf.Rect())
        first_text = min((p.y0 for p in prose), default=0.955 * H)
        parts = [pymupdf.Rect(d["rect"]) for d in page.get_drawings()]
        parts += [pymupdf.Rect(i["bbox"]) & page.rect for i in page.get_image_info()]
        parts = [g for g in parts if g.y0 >= 0.045 * H and g.y1 <= first_text + 2
                 and g.width * g.height < 0.9 * W * H and (g.width > 2 or g.height > 2)
                 and not ((g.x0 < 0.02 * W or g.x1 > 0.98 * W) and g.width < 0.2 * W)]
        if not parts:
            continue
        region = pymupdf.Rect(parts[0])
        for g in parts[1:]:
            region |= g
        if region.width * region.height > 0.15 * W * H:
            return page, region
    return None


def clip(code: str, manifest: dict) -> None:
    layout = json.loads((OUT / f"{code}_layout.json").read_text())
    rid = layout["report_id"]
    wanted = missing_items(code, rid)
    doc = pymupdf.open(ROOT / layout["pdf"])
    review = pymupdf.open()
    done = 0
    used_pages: set[int] = set()     # pages given to a figure captioned on another page
    for item in layout["items"]:
        if item["ident"] not in wanted:
            continue
        page = doc[item["page"] - 1]
        cap = pymupdf.Rect(item["bbox"])
        blocks = [b for b in layout["blocks"] if b["page"] == item["page"]]
        region, side, flags = region_for(page, cap, blocks, item["kind"])
        page_number = item["page"]
        if region is None and item["kind"] == "Figure":
            found = overleaf(doc, item["page"], {i["page"] for i in layout["items"]} | used_pages)
            if found:
                page, region = found
                page_number = page.number + 1
                used_pages.add(page_number)
                side = "other page"
                flags = [f"on page {page_number}, not the caption's page {item['page']}"]
                cap = pymupdf.Rect()
        name = f"{rid}-{item['ident']}"
        entry = {"code": code, "ident": item["ident"], "kind": item["kind"], "page": page_number,
                 "side": side, "flags": flags, "file": ""}
        if region is not None:
            pix = page.get_pixmap(clip=(region + (-4, -4, 4, 4)) & page.rect, dpi=DPI)
            pix.save(OUT / f"{name}.png")
            entry.update(file=f"{name}.png", bbox=[round(v, 1) for v in region])
            done += 1
            # Review page: the page with the clip (red) and caption (blue), and the clip.
            sheet = review.new_page(width=842, height=595)
            sheet.insert_text((20, 24), f"{name}  p{page_number}  {side}  {'; '.join(flags)}", fontsize=11)
            thumb = pymupdf.open()
            thumb.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
            tp = thumb[0]
            tp.draw_rect(region, color=(1, 0, 0), width=2)
            tp.draw_rect(cap, color=(0, 0, 1), width=1.5)
            sheet.insert_image(pymupdf.Rect(20, 40, 380, 580), pixmap=tp.get_pixmap(dpi=60))
            sheet.insert_image(pymupdf.Rect(400, 40, 822, 580), pixmap=pix)
        manifest[name] = entry
    missing = wanted - {i["ident"] for i in layout["items"]}
    for ident in sorted(missing):
        manifest[f"{rid}-{ident}"] = {"code": code, "ident": ident, "kind": re.match(r"[A-Za-z]+", ident).group(0).replace("SPM", ""),
                                      "page": None, "side": "", "flags": ["caption not found by the parser"], "file": ""}
    if review.page_count:
        review.save(OUT / f"{code}_review.pdf")
    print(f"{code}: {len(wanted)} without a Zenodo image, {done} clipped, {len(missing)} without a caption box")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extract", action="store_true", help="read the layout of the reports named")
    ap.add_argument("codes", nargs="*", help="assessments (default: all)")
    args = ap.parse_args()
    codes = [c.upper() for c in args.codes] or CODES
    if args.extract:
        for code in codes:
            extract(code)
        return 0
    path = OUT / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    for code in codes:
        if not (OUT / f"{code}_layout.json").exists():
            print(f"{code}: no layout; run --extract {code} first")
            continue
        for key in [k for k, v in manifest.items() if v["code"] == code]:
            manifest.pop(key)
        clip(code, manifest)
    path.write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
