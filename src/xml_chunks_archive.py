from pathlib import Path
import xml.etree.ElementTree as ET

from xml_loader_archive import (
    clean_text,
    build_xpath,
)


def make_structured_chunks(
    xml_path: str,
) -> list[dict]:

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

        # Ignore elements that have no children
        # and no useful text.
        text = clean_text(
            " ".join(
                element.itertext()
            )
        )

        if not text:
            continue

        children = list(element)

        # We primarily want meaningful structural
        # elements rather than every tiny XML node.
        if not children:
            continue

        xpath = build_xpath(
            element,
            parent_map,
        )

        attributes = dict(
            element.attrib
        )

        lines = []

        lines.append(
            f"XML element: {element.tag}"
        )

        if attributes:
            lines.append(
                "Attributes: "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in attributes.items()
                )
            )

        lines.append("")
        lines.append(text)

        chunk_text = "\n".join(lines)

        chunks.append(
            {
                "text": chunk_text,
                "source_file": path.name,
                "xpath": xpath,
                "element": element.tag,
                "attributes": attributes,
            }
        )

    return chunks