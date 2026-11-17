
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "evaluation" / "questions.json"

EXPECTED_PAPERS = {
    "attention_is_all_you_need.pdf",
    "bert.pdf",
    "gpt3_few_shot_learners.pdf",
    "roberta.pdf",
    "t5.pdf",
}


def main() -> None:
    """Validate the structure of the evaluation dataset."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError("The dataset must be a JSON list.")

    if len(questions) != 10:
        raise ValueError(
            f"Expected exactly 10 questions; found {len(questions)}."
        )

    required_fields = {
        "id",
        "question",
        "reference_answer",
        "source_papers",
        "key_points",
    }

    seen_ids = set()

    for item in questions:
        if not isinstance(item, dict):
            raise ValueError("Each dataset entry must be an object.")

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(
                f"Question entry is missing fields: {missing_fields}"
            )

        question_id = item["id"]

        if question_id in seen_ids:
            raise ValueError(f"Duplicate question ID: {question_id}")

        seen_ids.add(question_id)

        if not isinstance(item["question"], str) or not item["question"].strip():
            raise ValueError(f"Question {question_id} has no question text.")

        if (
            not isinstance(item["reference_answer"], str)
            or not item["reference_answer"].strip()
        ):
            raise ValueError(
                f"Question {question_id} has no reference answer."
            )

        if (
            not isinstance(item["source_papers"], list)
            or not item["source_papers"]
        ):
            raise ValueError(
                f"Question {question_id} has no source papers."
            )

        if not isinstance(item["key_points"], list) or not item["key_points"]:
            raise ValueError(
                f"Question {question_id} has no evaluation key points."
            )

        unknown_papers = (
            set(item["source_papers"]) - EXPECTED_PAPERS
        )

        if unknown_papers:
            raise ValueError(
                f"Question {question_id} has unknown papers: "
                f"{unknown_papers}"
            )

    if seen_ids != set(range(1, 11)):
        raise ValueError(
            "Question IDs must be the integers 1 through 10."
        )

    print("Evaluation dataset validation passed.")
    print(f"Total questions: {len(questions)}")
    print(f"Unique IDs: {len(seen_ids)}")
    print(f"Reference answers: {len(questions)}")
    print("All questions have source papers and key points.")


if __name__ == "__main__":
    main()
