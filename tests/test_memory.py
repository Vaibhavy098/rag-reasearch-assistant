
from src.memory import ConversationMemory


def test_memory_starts_empty():
    memory = ConversationMemory(max_turns=4)

    assert memory.turn_count == 0
    assert memory.get_history() == []


def test_memory_stores_question_and_answer():
    memory = ConversationMemory(max_turns=4)

    memory.add_turn("What is BERT?", "BERT is a language model.")

    history = memory.get_history()

    assert memory.turn_count == 1
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is BERT?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "BERT is a language model."


def test_memory_keeps_only_four_turns():
    memory = ConversationMemory(max_turns=4)

    for number in range(1, 6):
        memory.add_turn(
            f"Question {number}",
            f"Answer {number}",
        )

    history = memory.get_history()

    assert memory.turn_count == 4
    assert len(history) == 8

    # The oldest turn should have been discarded.
    assert history[0]["content"] == "Question 2"
    assert history[-1]["content"] == "Answer 5"


def test_memory_clear():
    memory = ConversationMemory(max_turns=4)

    memory.add_turn("What is BERT?", "A language model.")
    memory.clear()

    assert memory.turn_count == 0
    assert memory.get_history() == []


def test_memory_rejects_empty_question():
    memory = ConversationMemory()

    try:
        memory.add_turn("  ", "An answer.")
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for empty question.")

def test_record_helpful_feedback():
    memory = ConversationMemory(max_turns=4)
    memory.add_turn("What is BERT?", "BERT is a language model.")

    saved = memory.record_feedback("What is BERT?", "It_helped")

    assert saved is True
    assert "Helpful" in memory.format_history()


def test_record_not_helpful_feedback():
    memory = ConversationMemory(max_turns=4)
    memory.add_turn("Explain attention.", "A short explanation.")

    saved = memory.record_feedback(
        "Explain attention.",
        "Not_Helpful",
    )

    assert saved is True
    assert "Not helpful" in memory.format_history()


def test_feedback_does_not_add_a_conversation_turn():
    memory = ConversationMemory(max_turns=4)
    memory.add_turn("What is T5?", "T5 uses a text-to-text framework.")

    memory.record_feedback("What is T5?", "It_helped")

    assert memory.turn_count == 1


def test_feedback_for_missing_turn_returns_false():
    memory = ConversationMemory(max_turns=4)

    saved = memory.record_feedback("Unknown question", "It_helped")

    assert saved is False


def test_invalid_feedback_is_rejected():
    memory = ConversationMemory(max_turns=4)
    memory.add_turn("What is GPT-3?", "It is a language model.")

    import pytest

    with pytest.raises(ValueError):
        memory.record_feedback("What is GPT-3?", "maybe")
