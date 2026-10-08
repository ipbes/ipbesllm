# src/rdf_test.py
from src.rdf_graph import ThesaurusGraph
from src.rdf_query import HybridThesaurusRetriever

RDF_PATH = "data/rdf/ipbes-thesaurus.rdf"

def main() -> None:
    g = ThesaurusGraph(RDF_PATH)
    print(f"Total triples: {len(g.graph)}")
    print(f"Total concepts: {len(g.all_concept_uris())}")

    c = g.get_concept("https://ibok.ipbes.net/thesaurus/9195")
    print("\nSample concept:", c.pref_label)
    print("Definition:", c.definitions)
    print("Broader:", [g.pref_label_of(u) for u in c.broader])

    r = HybridThesaurusRetriever(g, n_results=3)
    hits = r.retrieve("What is ecological civilization?")
    for h in hits:
        print(f"\n→ {h.pref_label}  (score={h.score:.3f})")
        print(f"  ancestors={h.ancestors}")
        print(f"  siblings={h.siblings[:5]}")

if __name__ == "__main__":
    main()