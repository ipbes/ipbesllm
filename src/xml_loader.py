from pathlib import Path
import re
import xml.etree.ElementTree as ET


AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"


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


def element_text(element: ET.Element) -> str:
    """
    Extract visible text from an XML element, including inline
    elements such as <i>, <b>, <sup>, etc.
    """
    return clean_text(" ".join(element.itertext()))


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
                element_text(num_element)
                if num_element is not None
                else ""
            )

            heading = (
                element_text(heading_element)
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


def paragraph_text(paragraph: ET.Element) -> str:
    """
    Extract the main paragraph text.

    We deliberately focus on <content>/<p> instead of calling
    itertext() over the entire paragraph so that authorial notes
    do not get accidentally mixed into the main paragraph text.
    """

    parts = []

    for content in direct_children(paragraph, "content"):

        for child in content.iter():

            if local_name(child.tag) != "p":
                continue

            # Skip p elements belonging to authorial notes.
            ancestor_is_note = False
            # We don't have a parent map here, so identify notes
            # through the paragraph's immediate content structure.
            # Nested notes are uncommon in the main p structure.

            text = element_text(child)

            if text:
                parts.append(text)

    return clean_text(" ".join(parts))


def table_rows(table: ET.Element):
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

            cell_text = element_text(cell)

            cells.append(cell_text)

        if any(cells):
            rows.append(" | ".join(cells))

    return rows


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

        text = paragraph_text(paragraph)

        if not text:
            continue

        divisions, subdivisions = get_heading_context(
            paragraph,
            parent_map,
        )

        number_element = first_child(paragraph, "num")

        paragraph_number = (
            element_text(number_element)
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

        text = element_text(element)

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

        divisions, subdivisions = get_heading_context(
            table,
            parent_map,
        )

        section = divisions[-1] if divisions else ""
        subsection = subdivisions[-1] if subdivisions else ""

        rows = table_rows(table)

        for row_number, row_text in enumerate(rows, start=1):

            context_lines = []

            if document_metadata["title"]:
                context_lines.append(
                    f"Document: {document_metadata['title']}"
                )

            if section:
                context_lines.append(
                    f"Section: {section}"
                )

            if subsection:
                context_lines.append(
                    f"Subsection: {subsection}"
                )

            context_lines.append(
                f"Table row: {row_number}"
            )

            context_lines.append("")
            context_lines.append(row_text)

            chunks.append(
                {
                    "text": "\n".join(context_lines),
                    "source_file": path.name,
                    "chunk_id": (
                        f"{path.stem}-"
                        f"table-{chunk_index}-"
                        f"row-{row_number}"
                    ),
                    "chunk_type": "table_row",
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
                    "eId": table.attrib.get("eId", ""),
                    "xpath": xpath,
                    "table_row": row_number,
                }
            )

            chunk_index += 1

    return chunks