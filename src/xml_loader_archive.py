from pathlib import Path
import xml.etree.ElementTree as ET


def clean_text(text: str | None) -> str:
    """Normalize whitespace."""
    if not text:
        return ""

    return " ".join(text.split())


def element_text(element: ET.Element) -> str:
    """
    Return all text contained inside an XML element,
    including descendant elements.
    """
    return clean_text(
        " ".join(
            text
            for text in element.itertext()
            if text
        )
    )


def build_xpath(element: ET.Element, parent_map: dict) -> str:
    """
    Build an approximate XPath-like location for an element.
    """

    parts = []

    current = element

    while current is not None:

        parent = parent_map.get(current)

        if parent is None:
            parts.append(current.tag)
            break

        siblings = [
            child
            for child in list(parent)
            if child.tag == current.tag
        ]

        if len(siblings) == 1:
            parts.append(current.tag)
        else:
            index = siblings.index(current) + 1
            parts.append(
                f"{current.tag}[{index}]"
            )

        current = parent

    return "/" + "/".join(reversed(parts))


def parse_xml_file(xml_path: str) -> list[dict]:
    """
    Parse an XML document into structured chunks.

    Each chunk contains:

        text
        source_file
        xpath
        element
        attributes
    """

    path = Path(xml_path)

    tree = ET.parse(path)
    root = tree.getroot()

    parent_map = {
        child: parent
        for parent in root.iter()
        for child in parent
    }

    chunks = []

    for element in root.iter():

        text = element_text(element)

        if not text:
            continue

        xpath = build_xpath(
            element,
            parent_map,
        )

        attributes = dict(element.attrib)

        chunks.append(
            {
                "text": text,
                "source_file": path.name,
                "xpath": xpath,
                "element": element.tag,
                "attributes": attributes,
            }
        )

    return chunks