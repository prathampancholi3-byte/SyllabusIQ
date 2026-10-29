# SyllabusIQ

**Ask your syllabus anything.** A retrieval-augmented generation (RAG) assistant that answers questions about a university syllabus PDF, with page-level citations.

Built with Cohere, PostgreSQL + pgvector and Streamlit.

## What it does

- Answers questions such as attendance rules, CGPA requirements, course credits, units and recommended books
- Cites the course and page numbers for every answer
- Says "not found" for things the syllabus does not cover, instead of guessing
- Keeps conversation context, so follow-up questions work
- Has a student mode and a teacher mode (teachers can ask for question papers and study plans)
- Falls back to other models automatically if the primary one is unavailable

## How it works

```
Question → Planner → Router → Hybrid retrieval → (Rerank) → Answer → Citation check
```

1. **Planner**: one LLM call rewrites the question and extracts the intent, course codes and semesters
2. **Hybrid retrieval**: vector search (Cohere `embed-v4.0`, stored in pgvector) and keyword search are merged with Reciprocal Rank Fusion
3. **Optional reranking**: reorders the retrieved chunks with a Cohere reranker
4. **Answer generation**: `command-a-03-2025`, with fallbacks to `command-r-plus`, `command-r` and `command-r7b`
5. **Citation check**: verifies that every cited page was actually in the retrieved sources

## Evaluation

The project measures each improvement step on a set of test questions with known answer pages (`02_retrieval.py`):

| Version | Hit@5 | MRR | Answers correct | LLM calls / question |
|---|---|---|---|---|
| v0 naive (vector only) | 0.94 | 0.92 | 100% | 1 |
| v1 hybrid | 1.00 | 0.93 | 100% | 1 |
| v2 + planner | 0.94 | 0.84 | 100% | 2 |
| v3 + rerank | 0.94 | 0.82 | 89% | 5 |

On this small test set (18 retrieval questions, 9 answer checks), hybrid retrieval matched the full pipeline at a fifth of the cost, so reranking is off by default. Results on a larger set may differ.

## Project structure

| File | Purpose |
|---|---|
| `01_ingestion.py` | Reads the PDF, chunks it, creates embeddings, builds the index and test questions |
| `02_retrieval.py` | Evaluates each retrieval version and saves the best settings |
| `rag_engine.py` | The RAG engine (planning, retrieval, answering, caching, conversations) |
| `common.py` | Setup, scoring helpers and cost calculation |
| `app.py` | Streamlit chat interface |
| `chat.py` | Simple terminal chat |

## Setup

**Requirements:** Python 3.10+, PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension, and a [Cohere API key](https://dashboard.cohere.com/) (a free trial key works).

```bash
git clone https://github.com/prathampancholi3-byte/SyllabusIQ
cd SyllabusIQ
pip install -r requirements.txt
```

Create the database:

```sql
CREATE DATABASE charusat_syllabus_rag;
\c charusat_syllabus_rag
CREATE EXTENSION IF NOT EXISTS vector;
```

Create a `.env` file next to the scripts:

```
COHERE_API_KEY=your_key_here
COHERE_TIER=trial
DATABASE_URL=postgresql://USER:PASSWORD@localhost:5432/charusat_syllabus_rag
```

Place your syllabus PDF in the folder as `syllabus.pdf`.

## Run

```bash
python 01_ingestion.py     # wait for "Ingestion script completed successfully"
python 02_retrieval.py     # evaluates and saves the best settings
streamlit run app.py       # opens the chat window at http://localhost:8501
```

For a terminal chat instead, run `python chat.py`.

## Notes

- A Cohere trial key allows about 1,000 calls per month, which is enough for this project but not for heavy use
- Answers only cover the content of the PDF you ingest
- Never commit your `.env` file. Add it to `.gitignore`

## Tech stack

Python · Cohere (embeddings, chat, rerank) · PostgreSQL · pgvector · psycopg · pandas · Streamlit
