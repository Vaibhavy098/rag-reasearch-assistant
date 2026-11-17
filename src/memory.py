
from collections import deque
from typing import TypedDict


class Message(TypedDict):
    role: str
    content: str


class ConversationMemory:
    """Store recent question-answer turns and their feedback."""

    def __init__(self, max_turns: int = 4) -> None:
        if max_turns < 1:
            raise ValueError("max_turns must be at least 1.")

        self._turns: deque[dict[str, str | None]] = deque(
            maxlen=max_turns
        )

    def add_turn(self, question: str, answer: str) -> None:
        question = question.strip()
        answer = answer.strip()

        if not question:
            raise ValueError("Question cannot be empty.")
        if not answer:
            raise ValueError("Answer cannot be empty.")

        self._turns.append(
            {
                "question": question,
                "answer": answer,
                "feedback": None,
            }
        )

    def record_feedback(self, question: str, feedback: str) -> bool:
        """Attach feedback to the most recent matching question."""

        question = question.strip()
        feedback = feedback.strip().lower()

        if feedback not in {"it_helped", "not_helpful"}:
            raise ValueError(
                "Feedback must be 'It_helped' or 'Not_Helpful'."
            )

        # Search newest to oldest in case a question was repeated.
        for turn in reversed(self._turns):
            if turn["question"] == question:
                turn["feedback"] = feedback
                return True

        # The turn may have expired from the four-turn memory.
        return False

    def get_history(self) -> list[Message]:
        """Return ordinary chat messages for the LLM."""

        messages: list[Message] = []

        for turn in self._turns:
            messages.append(
                {"role": "user", "content": turn["question"] or ""}
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": turn["answer"] or "",
                }
            )

        return messages

    def format_history(self) -> str:
        """Format recent turns, including feedback, for prompts."""

        if not self._turns:
            return "No previous conversation."

        history_parts: list[str] = []

        for turn in self._turns:
            part = (
                f"User: {turn['question']}\n"
                f"Assistant: {turn['answer']}"
            )

            feedback = turn["feedback"]

            if feedback == "it_helped":
                part += (
                    "\nUser feedback: Helpful. This response style "
                    "worked for the user."
                )
            elif feedback == "not_helpful":
                part += (
                    "\nUser feedback: Not helpful. In a future "
                    "response, try to improve clarity, relevance, "
                    "or explanation structure."
                )

            history_parts.append(part)

        return "\n\n".join(history_parts)

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)
