import os
import atexit
import numpy as np
from dotenv import load_dotenv
import cohere
from psycopg_pool import ConnectionPool
from rag_engine import RAGEngine
from pgvector.psycopg import register_vector

load_dotenv()

USD_TO_INR = 88
PDF_PATH = "syllabus.pdf"
EMBED_MODEL = "embed-v4.0"
EMBED_DIM = 1536

# Cohere list prices per 1M tokens (input, output). A trial key is free.
PRICES = {
    "command-a-03-2025": (2.50, 10.00),
    "command-r-plus-08-2024": (2.50, 10.00),
    "command-r-08-2024": (0.15, 0.60),
    "command-r7b-12-2024": (0.0375, 0.15),
}


def usd(model, in_tok, out_tok=0):
    in_price, out_price = PRICES.get(model, (0, 0))
    return (in_tok / 1e6) * in_price + (out_tok / 1e6) * out_price


def is_hit(result_row, gold_pages):
    """result_row = (id, page_start, page_end); hit if any gold page falls in range."""
    page_start, page_end = result_row[1], result_row[2]
    return any(page_start <= g <= page_end for g in gold_pages)


def score(results, gold):
    """Hit@1, Hit@5 and MRR for a batch of retrieval results."""
    assert results, ("no retrieval results: eval_questions is empty. "
                     "Run 01_ingestion.py until it prints 'Ingestion script completed successfully'.")
    hits_at_1, hits_at_5, mrr_sum = 0, 0, 0.0
    for res_list, gold_pages in zip(results, gold):
        rank = next((i for i, r in enumerate(res_list[:5]) if is_hit(r, gold_pages or [])), None)
        if rank is not None:
            if rank == 0:
                hits_at_1 += 1
            hits_at_5 += 1
            mrr_sum += 1.0 / (rank + 1)
    n = len(results)
    return {"Hit@1": hits_at_1 / n, "Hit@5": hits_at_5 / n, "MRR": mrr_sum / n}


def save_eval(rag, name, results):
    with rag.pool.connection() as conn:
        conn.execute(
            "INSERT INTO eval_results (name, results) VALUES (%s, %s) "
            "ON CONFLICT (name) DO UPDATE SET results = EXCLUDED.results",
            (name, str(results)),
        )


class Setup:
    def __init__(self, client, conn, rag, workers, tier):
        self.client = client
        self.conn = conn
        self.rag = rag
        self.workers = workers
        self.tier = tier


def setup():
    tier = os.environ.get("COHERE_TIER", "trial")
    workers = int(os.environ.get("WORKERS", 4 if tier == "trial" else 32))
    key = os.environ.get("COHERE_API_KEY")
    assert key, "COHERE_API_KEY is missing: put COHERE_API_KEY=... in the .env file next to these scripts"
    client = cohere.ClientV2(api_key=key, timeout=120)

    pool = ConnectionPool(
        os.environ.get("DATABASE_URL",
                       "postgresql://postgres:pratham@localhost:5432/charusat_syllabus_rag"),
        configure=register_vector,
        open=True,
    )

    conn = pool.getconn()
    conn.autocommit = True   # without this, INSERTs made through `conn` are rolled back at exit

    def _shutdown():
        # return the checked-out connection, then close the pool so no worker threads linger
        try:
            pool.putconn(conn)
        except Exception:
            pass
        try:
            pool.close()
        except Exception:
            pass

    atexit.register(_shutdown)

    rag = RAGEngine(pool, client, tier)
    return Setup(client, conn, rag, workers, tier)