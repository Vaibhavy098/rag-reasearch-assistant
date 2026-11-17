
import ollama


MODEL_NAME = "qwen2.5:3b"

SYSTEM_MESSAGE = """
You are a helpful research assistant.

Answer questions clearly and accurately.
If you are uncertain, say so rather than inventing facts.
When the user provides research material, use that material
as the primary evidence for your answer.
"""


def generate_answer(
    prompt: str,
    system_message: str = SYSTEM_MESSAGE,
) -> str:
    """Send a prompt to the locally running Ollama model."""

    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_message.strip(),
            },
            {
                "role": "user",
                "content": prompt.strip(),
            },
        ],
        options={
            "temperature": 0.2,
        },
    )

    return response.message.content.strip()


def main() -> None:
    print(f"Local LLM: {MODEL_NAME}")
    print("Ask a question. Type 'exit' to quit.")

    while True:
        prompt = input("\nYour question: ").strip()

        if prompt.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not prompt:
            print("Please enter a question.")
            continue

        try:
            answer = generate_answer(prompt)
            print("\nAnswer:\n")
            print(answer)

        except ollama.ResponseError as error:
            print(f"Ollama returned an error: {error}")

        except Exception as error:
            print(
                f"Could not generate an answer: {error}\n"
                "Check that Ollama is running and the model "
                "has been downloaded."
            )


if __name__ == "__main__":
    main()
