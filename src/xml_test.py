from pathlib import Path

from xml_loader import parse_akn_file


XML_DIR = Path("data/xml")


def main():

    xml_files = list(XML_DIR.glob("*.xml"))

    print(f"XML files found: {len(xml_files)}")

    if not xml_files:
        print(f"No XML files found in {XML_DIR.resolve()}")
        return

    for xml_file in xml_files:

        print()
        print("=" * 80)
        print(f"FILE: {xml_file.name}")
        print("=" * 80)

        chunks = parse_akn_file(str(xml_file))

        print(f"Chunks: {len(chunks)}")

        paragraph_count = sum(
            1
            for chunk in chunks
            if chunk["chunk_type"] == "paragraph"
        )

        standalone_count = sum(
            1
            for chunk in chunks
            if chunk["chunk_type"] == "standalone_p"
        )

        table_count = sum(
            1
            for chunk in chunks
            if chunk["chunk_type"] == "table_row"
        )

        print(f"Paragraph chunks: {paragraph_count}")
        print(f"Standalone p chunks: {standalone_count}")
        print(f"Table-row chunks: {table_count}")

        print()
        print("FIRST CHUNKS")
        print("-" * 80)

        for chunk in chunks[:10]:

            print()
            print(f"Type: {chunk['chunk_type']}")
            print(f"ID: {chunk['chunk_id']}")
            print(f"Section: {chunk['division']}")
            print(f"Subsection: {chunk['subdivision']}")
            print(f"Paragraph: {chunk['paragraph']}")
            print(f"eId: {chunk['eId']}")
            print(f"XPath: {chunk['xpath']}")
            print()
            print(chunk["text"][:1000])


if __name__ == "__main__":
    main()