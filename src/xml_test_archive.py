from pathlib import Path

from xml_loader_archive import parse_xml_file


XML_DIR = Path("data/xml")


def main():

    xml_files = list(
        XML_DIR.glob("*.xml")
    )

    print(
        f"XML files found: {len(xml_files)}"
    )

    if not xml_files:
        print(
            "No XML files found in "
            f"{XML_DIR.resolve()}"
        )
        return

    for xml_file in xml_files:

        print()
        print("=" * 80)
        print(f"FILE: {xml_file.name}")
        print("=" * 80)

        chunks = parse_xml_file(
            str(xml_file)
        )

        print(
            f"Chunks: {len(chunks)}"
        )

        for chunk in chunks[:10]:

            print()
            print(
                f"Element: {chunk['element']}"
            )

            print(
                f"XPath: {chunk['xpath']}"
            )

            print(
                f"Attributes: "
                f"{chunk['attributes']}"
            )

            print(
                f"Text: {chunk['text'][:500]}"
            )


if __name__ == "__main__":
    main()