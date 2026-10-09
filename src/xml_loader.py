from pathlib import Path
import re
import xml.etree.ElementTree as ET


AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"

# Elements whose text should never appear in a chunk's main content.
# These are typically metadata/footnotes rather than narrative text.
EXCLUDED_TEXT_ELEMENTS = {
    "authorialNote",
    "authorialNotes",
    "note",
}

# Table structure. <p> inside these are indexed as table rows, never as
# paragraph or standalone text.
TABLE_ELEMENTS = {"table", "tr", "td", "th"}


def local_name(tag: str) -> str:
    """Return the local XML element name without the namespace."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def clean_text(text: str | None) -> str:
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_inside_excluded(
    element: ET.Element,
    parent_map: dict,
) -> bool:
    """
    Return True if `element` is a descendant of any element whose
    local name is in EXCLUDED_TEXT_ELEMENTS.
    """
    current = parent_map.get(element)

    while current is not None:
        if local_name(current.tag) in EXCLUDED_TEXT_ELEMENTS:
            return True
        current = parent_map.get(current)

    return False


def element_text(
    element: ET.Element,
    parent_map: dict | None = None,
) -> str:
    """
    Extract visible text from an XML element, including inline
    elements such as <i>, <b>, <sup>, etc.

    If `parent_map` is supplied, any text belonging to an excluded
    subtree (e.g. <authorialNote>) is skipped.
    """
    if parent_map is None:
        return clean_text(" ".join(element.itertext()))

    if local_name(element.tag) in EXCLUDED_TEXT_ELEMENTS:
        return ""

    parts: list[str] = []
    _collect_text(element, parts)
    return clean_text(" ".join(parts))


def _collect_text(node: ET.Element, parts: list[str]) -> None:
    """
    Append the visible text of `node` (excluding its own tail) to `parts`,
    skipping excluded subtrees such as <authorialNote>.

    A child's `tail` is the text after the child's closing tag, so it
    belongs to `node`'s narrative. It is kept even when the child itself
    is excluded:

        <p>Germany<authorialNote>...</authorialNote> contributed USD 400,000.</p>
                                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^ tail
    """
    if node.text:
        parts.append(node.text)

    for child in node:
        if local_name(child.tag) not in EXCLUDED_TEXT_ELEMENTS:
            _collect_text(child, parts)
        if child.tail:
            parts.append(child.tail)


def direct_children(element: ET.Element, name: str):
    return [
        child
        for child in list(element)
        if local_name(child.tag) == name
    ]


def first_child(element: ET.Element, name: str):
    for child in list(element):
        if local_name(child.tag) == name:
            return child
    return None


def find_descendant(element: ET.Element, name: str):
    for child in element.iter():
        if local_name(child.tag) == name:
            return child
    return None


def build_xpath(element: ET.Element, parent_map: dict) -> str:
    """
    Build a simple structural XPath-like location.
    Namespace prefixes are omitted to make the result readable.
    """
    parts = []
    current = element

    while current is not None:
        parent = parent_map.get(current)

        current_name = local_name(current.tag)

        if parent is None:
            parts.append(current_name)
            break

        siblings = [
            child
            for child in list(parent)
            if local_name(child.tag) == current_name
        ]

        if len(siblings) == 1:
            parts.append(current_name)
        else:
            index = siblings.index(current) + 1
            parts.append(f"{current_name}[{index}]")

        current = parent

    return "/" + "/".join(reversed(parts))


def get_frbr_metadata(root: ET.Element) -> dict:
    """
    Extract document-level metadata from the Akoma Ntoso
    identification / FRBR structures.
    """

    metadata = {
        "document_type": "",
        "title": "",
        "date": "",
        "language": "",
        "country": "",
        "subtype": "",
        "number": "",
    }

    debate_report = find_descendant(root, "debateReport")

    if debate_report is not None:
        metadata["document_type"] = debate_report.attrib.get("name", "")

    frbr_work = find_descendant(root, "FRBRWork")

    if frbr_work is not None:
        alias = find_descendant(frbr_work, "FRBRalias")
        date = find_descendant(frbr_work, "FRBRdate")
        country = find_descendant(frbr_work, "FRBRcountry")
        subtype = find_descendant(frbr_work, "FRBRsubtype")
        number = find_descendant(frbr_work, "FRBRnumber")

        if alias is not None:
            metadata["title"] = alias.attrib.get("value", "")

        if date is not None:
            metadata["date"] = date.attrib.get("date", "")

        if country is not None:
            metadata["country"] = country.attrib.get("value", "")

        if subtype is not None:
            metadata["subtype"] = subtype.attrib.get("value", "")

        if number is not None:
            metadata["number"] = number.attrib.get("value", "")

    frbr_expression = find_descendant(root, "FRBRExpression")

    if frbr_expression is not None:
        language = find_descendant(frbr_expression, "FRBRlanguage")

        if language is not None:
            metadata["language"] = language.attrib.get("language", "")

    return metadata


def get_heading_context(element: ET.Element, parent_map: dict):
    """
    Walk upward through the XML tree and collect the most recent
    division/subdivision headings.
    """

    divisions = []
    subdivisions = []

    current = parent_map.get(element)

    while current is not None:

        name = local_name(current.tag)

        if name in {"division", "subdivision"}:

            num_element = first_child(current, "num")
            heading_element = first_child(current, "heading")

            number = (
                element_text(num_element, parent_map)
                if num_element is not None
                else ""
            )

            heading = (
                element_text(heading_element, parent_map)
                if heading_element is not None
                else ""
            )

            value = " ".join(
                part for part in [number, heading] if part
            )

            if name == "division":
                divisions.append(value)

            elif name == "subdivision":
                subdivisions.append(value)

        current = parent_map.get(current)

    divisions.reverse()
    subdivisions.reverse()

    return divisions, subdivisions


def make_chunk_id(
    source_file: str,
    element: ET.Element,
    kind: str,
    index: int,
) -> str:

    e_id = element.attrib.get("eId")

    if e_id:
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", e_id)
    else:
        safe_id = f"element-{index}"

    source_stem = Path(source_file).stem

    return f"{source_stem}-{kind}-{safe_id}"


def paragraph_text(
    paragraph: ET.Element,
    parent_map: dict,
) -> str:
    """
    Extract the main paragraph text.

    We deliberately focus on <content>/<p> and <content>/<intro> rather
    than calling itertext() over the entire paragraph, so that:

      * authorial notes do not get mixed into the main paragraph text;
      * the <num> element's text is not duplicated into the body.

    Akoma Ntoso paragraphs have this shape:

        <paragraph>
          <num>1.</num>
          <content>
            <p>...</p>
          </content>
        </paragraph>

    Structured paragraphs use <intro> followed by nested
    <subparagraph> blocks:

        <paragraph>
          <num>22.</num>
          <content>
            <intro><p>...</p></intro>
            <subparagraph>...</subparagraph>
          </content>
        </paragraph>
    """

    parts: list[str] = []

    for p in paragraph_p_elements(paragraph, parent_map):

        text = element_text(p, parent_map)

        if text:
            parts.append(text)

    return clean_text(" ".join(parts))


def _has_ancestor(
    element: ET.Element,
    parent_map: dict,
    names: set[str],
    stop: ET.Element | None = None,
) -> bool:
    """True if an ancestor of `element` (below `stop`) has a local name in `names`."""
    current = parent_map.get(element)

    while current is not None and current is not stop:
        if local_name(current.tag) in names:
            return True
        current = parent_map.get(current)

    return False


def paragraph_p_elements(
    paragraph: ET.Element,
    parent_map: dict,
) -> list[ET.Element]:
    """
    The <p> elements whose text makes up a paragraph chunk.

    Walks each direct <content> child in document order. This includes <p>
    under <intro> and nested <subparagraph> blocks, and excludes <p> that
    sit inside an excluded subtree (e.g. <authorialNote>) or inside a table:
    tables get their own row-level chunks with header context.

    The standalone-<p> pass uses the same list to avoid indexing these
    <p> a second time.
    """
    elements: list[ET.Element] = []

    for content in direct_children(paragraph, "content"):

        for child in content.iter():

            if local_name(child.tag) != "p":
                continue

            if _is_inside_excluded(child, parent_map):
                continue

            if _has_ancestor(child, parent_map, TABLE_ELEMENTS, stop=content):
                continue

            elements.append(child)

    return elements


def _span(cell: ET.Element, attribute: str) -> int:
    """colspan/rowspan as a positive int (1 when missing or invalid)."""
    try:
        return max(1, int(cell.attrib.get(attribute, "1")))
    except ValueError:
        return 1


def _nearest_table(element: ET.Element, parent_map: dict):
    current = parent_map.get(element)
    while current is not None:
        if local_name(current.tag) == "table":
            return current
        current = parent_map.get(current)
    return None


def table_grid(
    table: ET.Element,
    parent_map: dict,
) -> list[tuple[list[str], bool]]:
    """
    Read an Akoma Ntoso table into a grid of rows.

    Returns [(cells, is_header), ...] in document order, where:
      * both <td> and <th> cells are read;
      * a cell with colspan=N fills N columns, and one with rowspan=N is
        repeated in the same column of the next N-1 rows, so every row's
        cells line up with the header columns;
      * `is_header` is True when every cell that starts in the row is a <th>.

    Rows of nested tables are left to those tables; empty rows are dropped.
    """
    grid: list[tuple[list[str], bool]] = []

    # column -> [text, rows still to fill] for cells spanning several rows
    pending: dict[int, list] = {}

    for tr in table.iter():

        if local_name(tr.tag) != "tr":
            continue

        if _nearest_table(tr, parent_map) is not table:
            continue

        cells: list[str] = []
        own_kinds: list[str] = []

        def fill_carried() -> None:
            # Place cells carried down from a rowspan above.
            while len(cells) in pending:
                column = len(cells)
                text, rows_left = pending[column]
                cells.append(text)
                if rows_left <= 1:
                    del pending[column]
                else:
                    pending[column][1] = rows_left - 1

        carried_before = dict(pending)

        for cell in list(tr):

            kind = local_name(cell.tag)

            if kind not in {"td", "th"}:
                continue

            fill_carried()

            text = element_text(cell, parent_map)
            rowspan = _span(cell, "rowspan")

            for _ in range(_span(cell, "colspan")):
                column = len(cells)
                cells.append(text)
                if rowspan > 1:
                    pending[column] = [text, rowspan - 1]

            own_kinds.append(kind)

        # Carried cells to the right of the last cell in this row.
        for column in sorted(c for c in carried_before if c >= len(cells)):
            while len(cells) < column:
                cells.append("")
            fill_carried()

        if any(cells):
            is_header = bool(own_kinds) and all(k == "th" for k in own_kinds)
            grid.append((cells, is_header))

    return grid


def _column_labels(header_rows: list[list[str]]) -> list[str]:
    """
    One label per column from one or more header rows, e.g.

        | In-kind contribution  |   (colspan=2)
        | 2012     | 2013       |
    ->  ["In-kind contribution – 2012", "In-kind contribution – 2013"]
    """
    width = max((len(row) for row in header_rows), default=0)
    labels = []

    for column in range(width):
        parts: list[str] = []
        for row in header_rows:
            text = row[column] if column < len(row) else ""
            if text and text not in parts:
                parts.append(text)
        labels.append(" – ".join(parts))

    return labels


def format_row(cells: list[str], labels: list[str]) -> str:
    """
    'Country: Germany | 2012: 400,000' when the header lines up with the
    row; plain 'Germany | 400,000' otherwise.
    """
    if labels and len(labels) == len(cells):
        return " | ".join(
            f"{label}: {value}" if label else value
            for label, value in zip(labels, cells)
            if value
        )

    return " | ".join(cells)


def table_rows(
    table: ET.Element,
    parent_map: dict,
):
    """
    Plain row strings ('cell | cell | ...') for every row, header rows
    included. Kept for scripts that only need the raw rows; parse_akn_file
    uses table_grid().
    """
    return [" | ".join(cells) for cells, _ in table_grid(table, parent_map)]


def get_table_context(
    table: ET.Element,
    parent_map: dict,
) -> dict:
    """
    Extract useful contextual information surrounding
    an Akoma Ntoso table.
    """

    context = {
        "table_title": "",
        "table_description": "",
        "division": "",
        "subdivision": "",
    }

    divisions, subdivisions = get_heading_context(
        table,
        parent_map,
    )

    if divisions:
        context["division"] = divisions[-1]

    if subdivisions:
        context["subdivision"] = subdivisions[-1]

    # The table normally lives inside a subdivision.
    parent = parent_map.get(table)

    if parent is not None:

        # Look for a heading associated with the table's
        # containing subdivision.
        heading = first_child(parent, "heading")

        if heading is not None:
            context["table_title"] = element_text(
                heading,
                parent_map,
            )

        # Look for descriptive <p> elements before the table.
        for child in list(parent):

            if child is table:
                break

            if local_name(child.tag) == "content":

                for p in child.iter():

                    if local_name(p.tag) == "p":

                        text = element_text(p, parent_map)

                        if text:
                            context["table_description"] = text

                            # Usually the first descriptive
                            # paragraph is enough.
                            break

                if context["table_description"]:
                    break

    return context


def parse_akn_file(xml_path: str) -> list[dict]:
    """
    Parse an Akoma Ntoso 3.0 document into retrieval-oriented chunks.

    Chunk types:
      - paragraph
      - standalone_p
      - table_row
    """

    path = Path(xml_path)

    tree = ET.parse(path)
    root = tree.getroot()

    parent_map = {
        child: parent
        for parent in root.iter()
        for child in parent
    }

    document_metadata = get_frbr_metadata(root)

    chunks = []
    chunk_index = 0

    # <p> elements already included in a paragraph chunk; the standalone
    # pass skips them so the same text is not indexed twice.
    covered_p: set[ET.Element] = set()

    # ------------------------------------------------------------
    # Paragraph chunks
    # ------------------------------------------------------------

    for paragraph in root.iter():

        if local_name(paragraph.tag) != "paragraph":
            continue

        covered_p.update(paragraph_p_elements(paragraph, parent_map))

        text = paragraph_text(paragraph, parent_map)

        if not text:
            continue

        divisions, subdivisions = get_heading_context(
            paragraph,
            parent_map,
        )

        number_element = first_child(paragraph, "num")

        paragraph_number = (
            element_text(number_element, parent_map)
            if number_element is not None
            else ""
        )

        e_id = paragraph.attrib.get("eId", "")

        xpath = build_xpath(
            paragraph,
            parent_map,
        )

        section = divisions[-1] if divisions else ""
        subsection = (
            subdivisions[-1]
            if subdivisions
            else ""
        )

        context_lines = []

        if document_metadata["title"]:
            context_lines.append(
                f"Document: {document_metadata['title']}"
            )

        if document_metadata["date"]:
            context_lines.append(
                f"Date: {document_metadata['date']}"
            )

        if section:
            context_lines.append(
                f"Section: {section}"
            )

        if subsection:
            context_lines.append(
                f"Subsection: {subsection}"
            )

        if paragraph_number:
            context_lines.append(
                f"Paragraph: {paragraph_number}"
            )

        context_lines.append("")
        context_lines.append(text)

        chunk_text = "\n".join(context_lines)

        chunks.append(
            {
                "text": chunk_text,
                "source_file": path.name,
                "chunk_id": make_chunk_id(
                    path.name,
                    paragraph,
                    "paragraph",
                    chunk_index,
                ),
                "chunk_type": "paragraph",
                "document_type": document_metadata["document_type"],
                "title": document_metadata["title"],
                "date": document_metadata["date"],
                "language": document_metadata["language"],
                "country": document_metadata["country"],
                "subtype": document_metadata["subtype"],
                "number": document_metadata["number"],
                "division": section,
                "subdivision": subsection,
                "paragraph": paragraph_number,
                "eId": e_id,
                "xpath": xpath,
            }
        )

        chunk_index += 1

    # ------------------------------------------------------------
    # Standalone <p> chunks
    #
    # These occur in places such as:
    #   <mainBody><p>...</p>
    #
    # Skipped:
    #   * <p> already part of a paragraph chunk (including <intro> and
    #     <subparagraph> text);
    #   * <p> inside a table cell (indexed as table rows below);
    #   * <p> inside an excluded subtree, e.g. an <authorialNote>.
    # ------------------------------------------------------------

    for element in root.iter():

        if local_name(element.tag) != "p":
            continue

        if element in covered_p:
            continue

        if parent_map.get(element) is None:
            continue

        if _has_ancestor(element, parent_map, TABLE_ELEMENTS):
            continue

        if _is_inside_excluded(element, parent_map):
            continue

        text = element_text(element, parent_map)

        if not text:
            continue

        divisions, subdivisions = get_heading_context(
            element,
            parent_map,
        )

        xpath = build_xpath(
            element,
            parent_map,
        )

        section = divisions[-1] if divisions else ""
        subsection = subdivisions[-1] if subdivisions else ""

        chunk_text = "\n".join(
            part
            for part in [
                f"Section: {section}" if section else "",
                f"Subsection: {subsection}" if subsection else "",
                "",
                text,
            ]
            if part != ""
        )

        chunks.append(
            {
                "text": chunk_text,
                "source_file": path.name,
                "chunk_id": make_chunk_id(
                    path.name,
                    element,
                    "p",
                    chunk_index,
                ),
                "chunk_type": "standalone_p",
                "document_type": document_metadata["document_type"],
                "title": document_metadata["title"],
                "date": document_metadata["date"],
                "language": document_metadata["language"],
                "country": document_metadata["country"],
                "subtype": document_metadata["subtype"],
                "number": document_metadata["number"],
                "division": section,
                "subdivision": subsection,
                "paragraph": "",
                "eId": element.attrib.get("eId", ""),
                "xpath": xpath,
            }
        )

        chunk_index += 1

    # ------------------------------------------------------------
    # Table chunks
    # ------------------------------------------------------------

    for table in root.iter():

        if local_name(table.tag) != "table":
            continue

        xpath = build_xpath(
            table,
            parent_map,
        )

        table_context = get_table_context(
            table,
            parent_map,
        )

        grid = table_grid(table, parent_map)

        if not grid:
            continue

        # --------------------------------------------------------
        # Header rows.
        #
        # Leading rows made of <th> cells are the header. Their column
        # labels are attached to every value in the data rows
        # ("Country: Germany | 2012: 400,000"), and they are not indexed
        # as rows of their own unless the table has nothing else.
        #
        # Without <th> cells the first row is still shown as a probable
        # header (as before), but values are not labelled with it,
        # because it may well be data.
        # --------------------------------------------------------

        header_count = 0

        while header_count < len(grid) and grid[header_count][1]:
            header_count += 1

        numbered_rows = list(enumerate(grid, start=1))

        if header_count:
            labels = _column_labels(
                [cells for cells, _ in grid[:header_count]]
            )
            header = " | ".join(labels)
            data_rows = numbered_rows[header_count:] or numbered_rows
        else:
            labels = []
            header = " | ".join(grid[0][0])
            data_rows = numbered_rows

        # --------------------------------------------------------
        # Stable table identifier: prefer the table's eId, fall back
        # to the running chunk index only if no eId exists.
        # --------------------------------------------------------

        table_e_id = table.attrib.get("eId")

        if table_e_id:
            safe_table_id = re.sub(
                r"[^A-Za-z0-9_.-]", "_", table_e_id
            )
        else:
            safe_table_id = f"table-{chunk_index}"

        # --------------------------------------------------------
        # Create one chunk per row, but repeat the table context
        # in EVERY row.
        # --------------------------------------------------------

        for row_number, (cells, is_header_row) in data_rows:

            row_text = format_row(
                cells,
                [] if is_header_row else labels,
            )

            context_lines = []

            if document_metadata["title"]:
                context_lines.append(
                    f"Document: "
                    f"{document_metadata['title']}"
                )

            if document_metadata["date"]:
                context_lines.append(
                    f"Date: "
                    f"{document_metadata['date']}"
                )

            if table_context["division"]:
                context_lines.append(
                    f"Section: "
                    f"{table_context['division']}"
                )

            if table_context["subdivision"]:
                context_lines.append(
                    f"Table: "
                    f"{table_context['subdivision']}"
                )

            if table_context["table_title"]:
                context_lines.append(
                    f"Table title: "
                    f"{table_context['table_title']}"
                )

            if table_context["table_description"]:
                context_lines.append(
                    f"Table description: "
                    f"{table_context['table_description']}"
                )

            context_lines.append(
                f"Table row: {row_number}"
            )

            # Include column/header information with every row.
            if header and not (header_count == 0 and row_number == 1):
                context_lines.append(
                    f"Table header: {header}"
                )

            context_lines.append("")
            context_lines.append(
                f"Row data: {row_text}"
            )

            chunks.append(
                {
                    "text": "\n".join(
                        context_lines
                    ),

                    "source_file": path.name,

                    # Stable across runs: depends only on the
                    # table's eId and the row number within it.
                    "chunk_id": (
                        f"{path.stem}-table-"
                        f"{safe_table_id}-row-{row_number}"
                    ),

                    "chunk_type": "table_row",

                    "document_type":
                        document_metadata[
                            "document_type"
                        ],

                    "title":
                        document_metadata["title"],

                    "date":
                        document_metadata["date"],

                    "language":
                        document_metadata["language"],

                    "country":
                        document_metadata["country"],

                    "subtype":
                        document_metadata["subtype"],

                    "number":
                        document_metadata["number"],

                    "division":
                        table_context["division"],

                    "subdivision":
                        table_context["subdivision"],

                    "paragraph": "",

                    "eId":
                        table.attrib.get(
                            "eId",
                            "",
                        ),

                    "xpath": xpath,

                    "table_row": row_number,

                    "table_title":
                        table_context["table_title"],
                }
            )

            chunk_index += 1

    return chunks