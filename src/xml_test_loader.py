"""Regression tests for xml_loader.parse_akn_file.

Run from the project root, either way:

    PYTHONPATH=src python src/test_xml_loader.py
    PYTHONPATH=src python -m pytest src/test_xml_loader.py

Each test parses a small Akoma Ntoso snippet, so no files in data/ are needed.
"""
import tempfile
from pathlib import Path

from xml_loader import parse_akn_file


def _parse(body: str) -> list[dict]:
    xml = (
        '<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">'
        '<debateReport name="report"><debateBody>'
        f"{body}"
        "</debateBody></debateReport></akomaNtoso>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.xml"
        path.write_text(xml, encoding="utf-8")
        return parse_akn_file(str(path))


def _of_type(chunks, chunk_type):
    return [c for c in chunks if c["chunk_type"] == chunk_type]


def _row_data(chunk) -> str:
    return chunk["text"].split("Row data: ", 1)[1]


# ---------------------------------------------------------------------------
# Issue 1: text after an <authorialNote> must be kept
# ---------------------------------------------------------------------------

def test_text_after_footnote_is_kept():
    chunks = _parse(
        '<paragraph eId="p1"><num>1.</num><content>'
        "<p>Germany<authorialNote><p>Footnote text.</p></authorialNote>"
        " contributed USD 400,000.</p>"
        "</content></paragraph>"
    )
    (para,) = _of_type(chunks, "paragraph")
    assert para["text"].endswith("Germany contributed USD 400,000."), para["text"]
    assert "Footnote text" not in para["text"]


def test_footnote_inside_inline_markup():
    chunks = _parse(
        '<paragraph eId="p1"><content>'
        "<p>The <i>Platform<authorialNote><p>fn</p></authorialNote> plenary</i>"
        " met in Bonn.</p>"
        "</content></paragraph>"
    )
    (para,) = _of_type(chunks, "paragraph")
    assert para["text"].endswith("The Platform plenary met in Bonn."), para["text"]


# ---------------------------------------------------------------------------
# Issue 2: no duplicate standalone chunks
# ---------------------------------------------------------------------------

def test_intro_and_subparagraph_not_duplicated():
    chunks = _parse(
        '<paragraph eId="p2"><num>2.</num><content>'
        "<intro><p>The Plenary decided:</p></intro>"
        "<subparagraph><content><p>(a) to adopt X;</p></content></subparagraph>"
        "</content></paragraph>"
    )
    (para,) = _of_type(chunks, "paragraph")
    assert para["text"].endswith("The Plenary decided: (a) to adopt X;")
    assert _of_type(chunks, "standalone_p") == []


def test_table_cells_are_not_standalone_chunks():
    chunks = _parse(
        '<table eId="t1">'
        "<tr><td><p>Germany</p></td><td><p>400,000</p></td></tr>"
        "</table>"
    )
    assert _of_type(chunks, "standalone_p") == []
    assert len(_of_type(chunks, "table_row")) == 1


def test_table_inside_paragraph_not_mixed_into_paragraph_text():
    chunks = _parse(
        '<paragraph eId="p3"><content><p>See the table below.</p>'
        '<table eId="t1"><tr><td><p>Germany</p></td><td><p>400,000</p></td></tr></table>'
        "</content></paragraph>"
    )
    (para,) = _of_type(chunks, "paragraph")
    assert para["text"].endswith("See the table below."), para["text"]
    assert len(_of_type(chunks, "table_row")) == 1


def test_real_standalone_p_still_indexed():
    chunks = _parse(
        "<p>Opening statement by the Chair.</p>"
        "<p><authorialNote><p>note only</p></authorialNote></p>"
    )
    standalone = _of_type(chunks, "standalone_p")
    assert [c["text"] for c in standalone] == ["Opening statement by the Chair."]


# ---------------------------------------------------------------------------
# Issue 3: table headers
# ---------------------------------------------------------------------------

def test_th_header_labels_every_value():
    chunks = _parse(
        '<subdivision eId="s1"><heading>Table 2 In-kind contributions</heading>'
        '<table eId="t1">'
        "<tr><th><p>Country</p></th><th><p>2012</p></th></tr>"
        "<tr><td><p>Germany</p></td><td><p>400,000</p></td></tr>"
        "<tr><td><p>Norway</p></td><td><p>100,000</p></td></tr>"
        "</table></subdivision>"
    )
    rows = _of_type(chunks, "table_row")
    assert [_row_data(r) for r in rows] == [
        "Country: Germany | 2012: 400,000",
        "Country: Norway | 2012: 100,000",
    ]
    for r in rows:
        assert "Table header: Country | 2012" in r["text"]
        assert "Table title: Table 2 In-kind contributions" in r["text"]
    # Row numbers are positions in the table (the header is row 1).
    assert [r["table_row"] for r in rows] == [2, 3]


def test_two_header_rows_with_colspan():
    chunks = _parse(
        '<table eId="t1">'
        '<tr><th rowspan="2"><p>Country</p></th><th colspan="2"><p>Cash</p></th></tr>'
        "<tr><th><p>2012</p></th><th><p>2013</p></th></tr>"
        "<tr><td><p>Germany</p></td><td><p>1,000</p></td><td><p>2,000</p></td></tr>"
        "</table>"
    )
    (row,) = _of_type(chunks, "table_row")
    assert _row_data(row) == (
        "Country: Germany | Cash – 2012: 1,000 | Cash – 2013: 2,000"
    ), _row_data(row)


def test_rowspan_in_data_rows():
    chunks = _parse(
        '<table eId="t1">'
        "<tr><th><p>Year</p></th><th><p>Country</p></th><th><p>Amount</p></th></tr>"
        '<tr><td rowspan="2"><p>2012</p></td><td><p>Germany</p></td><td><p>400</p></td></tr>'
        "<tr><td><p>Norway</p></td><td><p>100</p></td></tr>"
        "</table>"
    )
    assert [_row_data(r) for r in _of_type(chunks, "table_row")] == [
        "Year: 2012 | Country: Germany | Amount: 400",
        "Year: 2012 | Country: Norway | Amount: 100",
    ]


def test_table_without_th_keeps_previous_behaviour():
    chunks = _parse(
        '<table eId="t1">'
        "<tr><td><p>Country</p></td><td><p>2012</p></td></tr>"
        "<tr><td><p>Germany</p></td><td><p>400,000</p></td></tr>"
        "</table>"
    )
    rows = _of_type(chunks, "table_row")
    assert [_row_data(r) for r in rows] == ["Country | 2012", "Germany | 400,000"]
    assert "Table header" not in rows[0]["text"]
    assert "Table header: Country | 2012" in rows[1]["text"]


def test_nested_table_rows_not_duplicated():
    chunks = _parse(
        '<table eId="outer"><tr><td><p>A</p></td><td>'
        '<table eId="inner"><tr><td><p>B</p></td></tr></table>'
        "</td></tr></table>"
    )
    rows = _of_type(chunks, "table_row")
    assert sorted(r["chunk_id"].split("-table-")[1] for r in rows) == [
        "inner-row-1",
        "outer-row-1",
    ]
    assert len({r["chunk_id"] for r in chunks}) == len(chunks)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
