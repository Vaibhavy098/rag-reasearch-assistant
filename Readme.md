# RAG Research Assistant

A local Retrieval-Augmented Generation (RAG) application that answers questions using five research papers covering the Transformer architecture, BERT, GPT-3, RoBERTa, and T5.

## Features

- PDF ingestion and text extraction using PyMuPDF
- Overlapping text chunking with document metadata
- Semantic embeddings using Sentence Transformers
- Vector similarity search using FAISS
- Locally hosted language-model responses through Ollama
- Conversational memory retaining the most recent four turns
- Streamlit chat interface
- Source references and retrieved-passage inspection
- User feedback controls
- Evaluation using a predefined ten-question dataset
- Automated tests and PDF evaluation reporting

## Technology Stack

- Python
- PyMuPDF
- Sentence Transformers
- FAISS CPU
- Ollama
- Streamlit
- pytest
- ReportLab

## Research Papers

1. Attention Is All You Need — https://arxiv.org/pdf/1706.03762.pdf
2. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — https://arxiv.org/pdf/1810.04805.pdf
3. Language Models are Few-Shot Learners — https://arxiv.org/pdf/2005.14165.pdf
4. RoBERTa: A Robustly Optimized BERT Pretraining Approach — https://arxiv.org/pdf/1907.11692.pdf
5. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5) — https://arxiv.org/pdf/1910.10683.pdf

## Setup

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Install Ollama and download the model

Install Ollama from https://ollama.com/download/windows.

Download the configured model:

```powershell
ollama pull qwen2.5:3b
ollama list
```

The model name in `src/llm.py` must match the installed Ollama model.

### 4. Prepare the research corpus

Run the following commands in order if the required generated files and index are not already available:

```powershell
python download_pdfs.py
python -m src.ingestion
python -m src.chunking
python -m src.embeddings
python -m src.build_index
```

### 5. Run the application

```powershell
python -m streamlit run app.py
```

Streamlit normally opens the application at http://localhost:8501.

## Evaluation

The evaluation dataset contains ten predefined questions with reference answers, expected source papers, and key points.

Run the evaluation:

```powershell
python evaluation/validate_questions.py
python evaluation/evaluate_rag.py
```

The results are saved to `evaluation/results.json`.

Generate the PDF report:

```powershell
python evaluation/generate_report.py
```

The report is saved to `reports/rag_research_assistant_report.pdf`.

## Baseline Evaluation

The recorded baseline evaluation completed all 10 questions without execution failures.

- Average source recall: 80%
- Average key-point coverage: 68.33%
- Failed evaluations: 0

These are baseline measurements, not guarantees of factual correctness. Source recall checks expected paper-file coverage, while key-point coverage uses lexical overlap and may not recognize correct paraphrases.

Update these values after rerunning evaluation if the results change.

## Testing

Run the automated test suite:

```powershell
python -m pytest -v
```

For targeted retrieval diagnostics:

```powershell
python tests/diagnose_retrieval.py
```

## Architecture

1. Research PDFs are downloaded and extracted into page-level text.
2. Extracted text is split into overlapping chunks with source metadata.
3. Chunks are embedded into vectors.
4. FAISS indexes the vectors.
5. A user query is embedded and matched against indexed chunks.
6. Retrieved passages and conversation history are supplied to the local LLM.
7. The generated answer and source information are displayed in Streamlit.

## Limitations

- The evaluation set is small and may not represent every query.
- Similarity scores do not guarantee factual correctness.
- Retrieved paper filenames do not guarantee that the exact supporting passage was found.
- The lexical key-point metric may penalize valid paraphrases.
- The local model may omit important information or produce unsupported claims.
- Conversation memory is in-process and is not necessarily persisted across application restarts.

## Future Improvements

- Improve retrieval for questions requiring evidence from multiple papers.
- Add passage-level relevance evaluation.
- Add stronger factual-grounding and citation checks.
- Expand the evaluation dataset.
- Compare retrieval settings through controlled experiments.
- Improve automated integration and error-handling tests.

## Running the Tests

```powershell
python -m pytest -v
```
