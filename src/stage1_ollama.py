import os
import ollama


MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")


def ask_ollama(question: str) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Answer the user's question clearly and concisely."
                ),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )

    return response["message"]["content"]


def main():
    print(f"Using Ollama model: {MODEL}")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Question: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        answer = ask_ollama(question)

        print("\nAnswer:")
        print(answer)
        print()


if __name__ == "__main__":
    main()