import os
import ollama


MODEL = os.getenv(
    "OLLAMA_EMBED_MODEL",
    "nomic-embed-text",
)


text = "The employee is entitled to annual leave."


response = ollama.embed(
    model=MODEL,
    input=text,
)


embedding = response["embeddings"][0]

print("Embedding model:", MODEL)
print("Dimensions:", len(embedding))
print("First 10 values:", embedding[:10])