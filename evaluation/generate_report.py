
import json
import sys
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results.json"
OUTPUT_PATH = (
    PROJECT_ROOT / "reports" / "rag_research_assistant_report.pdf"
)


def load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_PATH}\n"
            "Run python evaluation/evaluate_rag.py first."
        )

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data.get("results"), list):
        raise ValueError(
            "The evaluation JSON does not contain a results list."
        )

    if not isinstance(data.get("summary"), dict):
        raise ValueError(
            "The evaluation JSON does not contain a summary."
        )

    return data


def safe_text(value) -> str:
    """Escape text before inserting it into ReportLab paragraphs."""
    text = str(value if value is not None else "")
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def percentage(value) -> str:
    return f"{float(value) * 100:.2f}%"


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.drawString(
        2 * cm,
        1.1 * cm,
        "RAG Research Assistant | Evaluation Report",
    )
    canvas.drawRightString(
        A4[0] - 2 * cm,
        1.1 * cm,
        f"Page {doc.page}",
    )
    canvas.restoreState()


def build_report(data: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title="RAG Research Assistant - Final Report",
        author="RAG Research Assistant Project",
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=23,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#183153"),
            spaceAfter=14,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontSize=15,
            leading=19,
            textColor=colors.HexColor("#183153"),
            spaceBefore=14,
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontSize=8.5,
            leading=12,
            spaceAfter=5,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableCell",
            parent=styles["BodyText"],
            fontSize=7.5,
            leading=10,
        )
    )

    story = []
    summary = data["summary"]
    results = data["results"]

    successful = summary.get("successful_evaluations", 0)
    failed = summary.get("failed_evaluations", 0)
    total = summary.get("total_questions", len(results))

    story.append(Spacer(1, 2 * cm))
    story.append(
        Paragraph(
            "RAG Research Assistant",
            styles["ReportTitle"],
        )
    )
    story.append(
        Paragraph(
            "Project Implementation and Evaluation Report",
            styles["Heading2"],
        )
    )
    story.append(Spacer(1, 0.5 * cm))
    story.append(
        Paragraph(
            f"Report generated: {datetime.now().strftime('%d %B %Y')}",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "Evaluation timestamp (UTC): "
            + safe_text(data.get("evaluated_at_utc", "Not recorded")),
            styles["BodyText"],
        )
    )
    story.append(Spacer(1, 1 * cm))

    story.append(
        Paragraph("1. Executive Summary", styles["SectionHeading"])
    )
    story.append(
        Paragraph(
            "This project implements a Retrieval-Augmented Generation "
            "(RAG) research assistant over five research papers covering "
            "the Transformer architecture, BERT, GPT-3, RoBERTa, and T5. "
            "The system retrieves relevant passages from the indexed "
            "documents and supplies them as context to a locally hosted "
            "language model for answer generation.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            f"The recorded baseline evaluation processed {total} "
            f"questions successfully for {successful} questions, with "
            f"{failed} failed evaluations. The measured source-recall "
            f"average was "
            f"{percentage(summary.get('source_recall_average', 0))}, "
            f"and key-point coverage was "
            f"{percentage(summary.get('key_point_coverage_average', 0))}. "
            "These metrics describe retrieval coverage and lexical "
            "coverage of expected answer points; they do not establish "
            "factual correctness on their own.",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph("2. Project Objectives", styles["SectionHeading"])
    )
    objectives = [
        "Ingest and extract text from five research-paper PDFs.",
        "Split extracted content into overlapping chunks.",
        "Create semantic embeddings using Sentence Transformers.",
        "Index embeddings with FAISS for similarity-based retrieval.",
        "Generate evidence-informed answers using a local Ollama model.",
        "Maintain recent conversational context for follow-up questions.",
        "Provide a Streamlit chat interface with source information and feedback.",
        "Evaluate retrieval and answer coverage using a documented dataset.",
    ]

    for objective in objectives:
        story.append(
            Paragraph(
                "&#8226; " + safe_text(objective),
                styles["BodyText"],
            )
        )

    story.append(
        Paragraph("3. Technology Stack", styles["SectionHeading"])
    )
    stack = [
        ["Component", "Technology", "Purpose"],
        ["Language", "Python", "Application logic and pipeline"],
        ["PDF extraction", "PyMuPDF", "Extract text from PDF pages"],
        ["Embeddings", "all-MiniLM-L6-v2", "Convert text to vectors"],
        ["Vector index", "FAISS", "Retrieve similar chunks"],
        ["Local LLM", "Ollama / qwen2.5:3b", "Generate answers"],
        ["Interface", "Streamlit", "Interactive chat application"],
        ["Testing", "pytest", "Automated test execution"],
        ["Reporting", "ReportLab", "Generate this PDF report"],
    ]

    stack_table = Table(
        stack,
        colWidths=[3.0 * cm, 4.2 * cm, 9.0 * cm],
        repeatRows=1,
    )
    stack_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(stack_table)

    story.append(
        Paragraph("4. System Architecture", styles["SectionHeading"])
    )
    story.append(
        Paragraph(
            "Document ingestion: PDFs are downloaded and their page text "
            "is extracted and cleaned. Chunking: extracted text is split "
            "into overlapping text segments with source and page metadata. "
            "Embedding: each chunk is converted into a vector. Indexing: "
            "FAISS stores the vectors for similarity search. Retrieval: "
            "the user's question is embedded and the top-ranked chunks "
            "are selected. Generation: retrieved evidence and recent "
            "conversation history are passed to the local LLM. Interface: "
            "Streamlit presents the answer, source information, and "
            "feedback controls.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "The implementation uses normalized embeddings with a FAISS "
            "inner-product index. Similarity scores are ranking signals, "
            "not probabilities of correctness.",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph("5. Evaluation Methodology", styles["SectionHeading"])
    )
    story.append(
        Paragraph(
            "The evaluation dataset contains ten predefined questions, "
            "reference answers, expected source-paper filenames, and "
            "key points. Each question is evaluated independently by "
            "clearing conversational memory before generating its answer. "
            f"The recorded retrieval setting is top_k={data.get('top_k', 'Not recorded')}.",
            styles["BodyText"],
        )
    )
    story.append(
        Paragraph(
            "<b>Source recall:</b> the fraction of expected source-paper "
            "filenames present among the retrieved chunks. "
            "<b>Key-point coverage:</b> the fraction of predefined key "
            "points for which the generated answer reaches the configured "
            "lexical-overlap threshold. The key-point metric is heuristic "
            "and can penalize valid paraphrases or overlook factual errors.",
            styles["BodyText"],
        )
    )

    story.append(
        Paragraph("6. Evaluation Results", styles["SectionHeading"])
    )
    metric_rows = [
        ["Metric", "Recorded result"],
        ["Questions", str(total)],
        ["Successful evaluations", str(successful)],
        ["Failed evaluations", str(failed)],
        [
            "Average source recall",
            percentage(summary.get("source_recall_average", 0)),
        ],
        [
            "Average key-point coverage",
            percentage(summary.get("key_point_coverage_average", 0)),
        ],
    ]
    metric_table = Table(
        metric_rows,
        colWidths=[8 * cm, 5 * cm],
        repeatRows=1,
    )
    metric_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(metric_table)
    story.append(Spacer(1, 0.3 * cm))

    question_rows = [
        [
            Paragraph("<b>ID</b>", styles["TableCell"]),
            Paragraph("<b>Question</b>", styles["TableCell"]),
            Paragraph("<b>Source recall</b>", styles["TableCell"]),
            Paragraph("<b>Key-point coverage</b>", styles["TableCell"]),
            Paragraph("<b>Status</b>", styles["TableCell"]),
        ]
    ]

    for result in results:
        coverage = result.get("key_point_coverage", {}).get("score", 0)
        question_rows.append(
            [
                str(result.get("id", "")),
                Paragraph(
                    safe_text(result.get("question", "")),
                    styles["TableCell"],
                ),
                percentage(result.get("source_recall", 0)),
                percentage(coverage),
                "Failed" if result.get("error") else "Success",
            ]
        )

    question_table = Table(
        question_rows,
        colWidths=[0.8 * cm, 8.1 * cm, 2.0 * cm, 2.5 * cm, 1.5 * cm],
        repeatRows=1,
    )
    question_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#183153")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(question_table)

    story.append(
        Paragraph("7. Limitations", styles["SectionHeading"])
    )
    limitations = [
        "Evaluation uses only ten predefined questions; it may not represent all user queries.",
        "Source recall operates at paper-file level and does not guarantee the retrieved passage supports the answer.",
        "Lexical key-point coverage is not semantic evaluation or factual verification.",
        "The local LLM can omit important details or generate unsupported claims.",
        "Conversation memory is in-process and is not a persistent user database.",
        "Performance depends on local hardware, model availability, and retrieved-context quality.",
    ]

    for limitation in limitations:
        story.append(
            Paragraph(
                "&#8226; " + safe_text(limitation),
                styles["BodyText"],
            )
        )

    story.append(
        Paragraph("8. Future Improvements", styles["SectionHeading"])
    )
    improvements = [
        "Inspect and improve retrieval for questions requiring multiple papers.",
        "Compare different top_k settings using the same evaluation dataset.",
        "Add retrieval precision and passage-level relevance judgments.",
        "Use human review or a validated semantic evaluation method for answer quality.",
        "Expand the evaluation set with paraphrased, ambiguous, and out-of-scope questions.",
        "Add stronger automated checks for citations and unsupported claims.",
        "Improve error reporting and add integration tests for the full RAG workflow.",
    ]

    for improvement in improvements:
        story.append(
            Paragraph(
                "&#8226; " + safe_text(improvement),
                styles["BodyText"],
            )
        )

    story.append(
        Paragraph("9. Conclusion", styles["SectionHeading"])
    )
    story.append(
        Paragraph(
            "The project demonstrates an end-to-end local RAG workflow "
            "covering document ingestion, semantic retrieval, local "
            "language-model generation, conversational memory, a chat "
            "interface, and quantitative evaluation. The baseline run "
            "completed all ten evaluations without execution failures. "
            "The recorded metrics also identify opportunities to improve "
            "multi-paper retrieval and answer completeness. Further "
            "validation of evidence grounding and factual correctness "
            "is required before treating the system as a dependable "
            "research tool.",
            styles["BodyText"],
        )
    )

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )


def main() -> None:
    try:
        data = load_results()
        build_report(data)
        print("PDF report generated successfully.")
        print(f"Output: {OUTPUT_PATH}")
    except Exception as error:
        print(f"Could not generate the report: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
