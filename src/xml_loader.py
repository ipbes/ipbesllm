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

    parts: list[str] = []

    for node in element.iter():
        # Skip anything inside an excluded subtree.
        if local_name(node.tag) in EXCLUDED_TEXT_ELEMENTS:
            continue
        if _is_inside_excluded(node, parent_map):
            continue

        if node.text:
            parts.append(node.text)

        # `tail` is the text that appears *after* the closing tag
        # of `node` but still inside its parent. It belongs to the
        # parent's narrative, not to any excluded subtree.
        if node.tail and not _is_inside_excluded(node, parent_map):
            parts.append(node.tail)

    return clean_text(" ".join(parts))


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

    for content in direct_children(paragraph, "content"):

        # Walk the content subtree in document order and pick up
        # narrative <p> elements. This naturally includes <p> under
        # <intro> and any top-level <p> under <content>.
        for child in content.iter():

            if local_name(child.tag) != "p":
                continue

            # Skip <p> that live inside an excluded subtree, e.g.
            # an <authorialNote>.
            if _is_inside_excluded(child, parent_map):
                continue

            text = element_text(child, parent_map)

            if text:
                parts.append(text)

    return clean_text(" ".join(parts))


def table_rows(
    table: ET.Element,
    parent_map: dict,
):
    """
    Convert an Akoma Ntoso table into row-level text chunks.
    """

    rows = []

    for row in table.iter():

        if local_name(row.tag) != "tr":
            continue

        cells = []

        for cell in list(row):

            if local_name(cell.tag) != "td":
                continue

            cell_text = element_text(cell, parent_map)

            cells.append(cell_text)

        if any(cells):
            rows.append(" | ".join(cells))

    return rows


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

    # ------------------------------------------------------------
    # Paragraph chunks
    # ------------------------------------------------------------

    for paragraph in root.iter():

        if local_name(paragraph.tag) != "paragraph":
            continue

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
    # They are not inside <paragraph>.
    # ------------------------------------------------------------

    for element in root.iter():

        if local_name(element.tag) != "p":
            continue

        parent = parent_map.get(element)

        if parent is None:
            continue

        if local_name(parent.tag) == "content":
            grandparent = parent_map.get(parent)

            if (
                grandparent is not None
                and local_name(grandparent.tag) == "paragraph"
            ):
                continue

        # Skip <p> nested inside an excluded subtree, e.g. an
        # <authorialNote>.
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

        rows = table_rows(table, parent_map)

        if not rows:
            continue

        # --------------------------------------------------------
        # Extract the first row as a header when appropriate.
        # --------------------------------------------------------

        header = rows[0]

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

        for row_number, row_text in enumerate(
            rows,
            start=1,
        ):

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
            if row_number != 1:
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