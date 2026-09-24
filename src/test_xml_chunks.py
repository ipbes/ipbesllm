from pathlib import Path

from xml_chunks import make_structured_chunks


XML_DIR = Path("data/xml")


def main():

    xml_files = list(
        XML_DIR.glob("*.xml")
    )

    if not xml_files:
        print("No XML files found.")
        return

    for xml_file in xml_files:

        chunks = make_structured_chunks(
            str(xml_file)
        )

        print()
        print("=" * 80)
        print(xml_file.name)
        print("=" * 80)

        print(
            f"Structured chunks: "
            f"{len(chunks)}"
        )

        for chunk in chunks[:5]:

            print()
            print("-" * 80)

            print(
                f"XPath: {chunk['xpath']}"
            )

            print(
                f"Element: "
                f"{chunk['element']}"
            )

            print(
                f"Text:\n{chunk['text'][:1000]}"
            )


if __name__ == "__main__":
    main()