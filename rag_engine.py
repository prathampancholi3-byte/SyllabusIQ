import hashlib
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

RETRYABLE = ["429", "500", "502", "503", "504", "TooManyRequests", "ServiceUnavailable", "GatewayTimeout",
             "InternalServer"]
NETWORK_ERRORS = {"RemoteProtocolError", "ConnectError", "ReadTimeout", "ConnectTimeout", "ReadError",
                  "WriteError", "PoolTimeout", "WriteTimeout", "ConnectionError", "TimeoutException"}
EMBED_MODEL = "embed-v4.0"          # Cohere: supports 256 / 512 / 1024 / 1536 dimensions
EMBED_DIM = 1536
RERANK_MODEL = "rerank-v3.5"

# Preferred answer models, best-first. Only the ones your key can really call are used
# (checked once at start-up with client.models.list(), which is what stops the 404s).
CANDIDATE_LLMS = ["command-a-03-2025", "command-r-plus-08-2024", "command-r-08-2024", "command-r7b-12-2024"]
INTENTS = ["fact_lookup", "rule", "multi_step", "list", "compare", "paper", "mcq", "study_plan",
           "overview", "out_of_scope", "chit_chat"]
NOT_FOUND = "Not found in the syllabus."


class DailyQuotaError(Exception):
    """Raised when the trial key's monthly call quota is used up (waiting a few minutes will not help)."""
    pass


def rrf(ranked_lists, k=60):
    """Reciprocal Rank Fusion: merge several ranked id lists into one."""
    scores = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


class RateLimiter:
    """Thread-safe sliding-window limiter: `requests` is the number of API calls about to be made."""
    def __init__(self, per_minute=80, tokens_per_minute=10 ** 9):
        # Cohere trial limits (per minute): chat 20 calls, embed 100 calls, rerank 10 calls; we stay a little under
        self.per_minute = per_minute
        self.tpm = tokens_per_minute
        self.lock = threading.Lock()
        self.events = []          # (timestamp, n_texts, n_tokens)

    def wait(self, model, tokens, requests):
        requests = min(requests, self.per_minute)
        tokens = min(int(tokens * 1.3), self.tpm)         # caller's chars//4 under-counts tables and codes
        while True:
            with self.lock:
                now = time.time()
                self.events = [e for e in self.events if now - e[0] < 60]
                if (sum(e[1] for e in self.events) + requests <= self.per_minute
                        and sum(e[2] for e in self.events) + tokens <= self.tpm):
                    self.events.append((now, requests, tokens))
                    return
                sleep_for = 60 - (now - self.events[0][0]) + 0.5
            time.sleep(max(sleep_for, 1))


class RAGEngine:
    def __init__(self, pool, client, tier):
        self.pool = pool
        self.client = client                           # cohere.ClientV2
        self.tier = tier                               # "trial" or "production"
        self.llm_model = "command-a-03-2025"
        self.fallbacks = ["command-r-plus-08-2024", "command-r-08-2024"]
        self.alt_model = "command-r7b-12-2024"         # the cheaper model compared in notebook 2
        trial = tier == "trial"
        self.chat_limiter = RateLimiter(per_minute=18 if trial else 400)
        self.embed_limiter = RateLimiter(per_minute=90 if trial else 1800)
        self.rerank_limiter = RateLimiter(per_minute=9 if trial else 900)
        self.calls = {"chat": 0, "embed": 0, "rerank": 0}   # trial keys allow only ~1,000 calls per month in total
        self.embed_batch = 96                          # Cohere's maximum texts per embed call (calls are what is limited)
        self._llm_sem = threading.Semaphore(3 if trial else 16)
        self._emb_cache = {}
        self._mem_hist = {}
        self._mem_next = 1000
        self._lock = threading.Lock()
        self._warned = set()

        # Defaults, overridden by load_config
        self.use_rerank = False
        self.strategy = "structure"
        self.variant = "prefixed"

        self.ensure_schema()
        self.resolve_models()

    # ------------------------------------------------------------------ setup
    def _warn(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            print(f"[rag_engine] {msg}")

    def ensure_schema(self):
        """Create the tables this engine writes to, if they are missing."""
        ddl = [
            "CREATE TABLE IF NOT EXISTS eval_results (name text PRIMARY KEY, results text)",
            """CREATE TABLE IF NOT EXISTS llm_cache (key text PRIMARY KEY, model text, task text, response text,
               input_tokens int, thinking_tokens int, output_tokens int, seconds real,
               created_at timestamptz DEFAULT now())""",
            """CREATE TABLE IF NOT EXISTS conversations (id serial PRIMARY KEY, title text, role text,
               created_at timestamptz DEFAULT now(), turns int DEFAULT 0)""",
            """CREATE TABLE IF NOT EXISTS messages (id serial PRIMARY KEY, conversation_id int, role text,
               content text, created_at timestamptz DEFAULT now())""",
            "CREATE TABLE IF NOT EXISTS rag_traces (id serial PRIMARY KEY, trace_json text, created_at timestamptz DEFAULT now())",
        ]
        for stmt in ddl:
            try:
                with self.pool.connection() as c:
                    c.execute(stmt)
            except Exception as e:                      # noqa: BLE001
                self._warn("ddl", f"could not create table ({str(e)[:80]}); persistence may be limited")

    def resolve_models(self):
        """Keep only chat models this API key can call, so a wrong name never causes a 404."""
        try:
            names = {m.name for m in (self.client.models.list(endpoint="chat").models or []) if m.name}
        except Exception as e:                          # noqa: BLE001
            self._warn("list", f"could not list models ({str(e)[:80]}); using defaults")
            return
        chain = [m for m in CANDIDATE_LLMS if m in names]
        if not chain:
            return
        self.llm_model, self.fallbacks = chain[0], chain[1:]
        self.alt_model = chain[-1] if len(chain) > 1 else chain[0]
        print(f"[rag_engine] models → primary {self.llm_model} · fallbacks {self.fallbacks} · alt {self.alt_model}")

    def _retry(self, func, waits=None):
        """Run an API call with backoff on transient errors."""
        waits = waits or [2, 5, 15, 35, 65]            # last wait > 60 s so a per-minute limit resets
        last = None
        for w in waits:
            try:
                return func()
            except Exception as e:                      # noqa: BLE001
                msg = str(e)
                if "API calls / month" in msg or "limited to" in msg and "month" in msg:
                    raise DailyQuotaError(msg) from e   # trial key's monthly quota: waiting won't help
                network = (type(e).__name__ in NETWORK_ERRORS
                           or any(s in msg for s in ("Server disconnected", "timed out", "Connection reset",
                                                     "Connection aborted", "Temporary failure")))
                if network or any(err in msg for err in RETRYABLE):
                    last = e
                    time.sleep(w)
                else:
                    raise
        raise RuntimeError(f"Max retries exceeded; last error: {last}") from last

    # ----------------------------------------------------------------- config
    def load_config(self):
        with self.pool.connection() as c:
            row = c.execute("SELECT config_json FROM pipeline_config ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                return {}
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])

    def save_config(self, **kwargs):
        merged = {**self.load_config(), **kwargs}      # keep earlier keys (chunking, variant, dims...)
        with self.pool.connection() as c:
            c.execute("INSERT INTO pipeline_config (config_json) VALUES (%s)", (json.dumps(merged),))

    # ------------------------------------------------------------- embeddings
    def query_text(self, q, variant=None):
        return q        # Cohere tells queries from documents with input_type, so no text prefix is needed

    @staticmethod
    def _floats(res):
        e = res.embeddings
        return getattr(e, "float_", None) or getattr(e, "float", None) or e["float"]

    def embed_raw(self, texts, input_type):
        """One Cohere embed call (<= 96 texts): paced, retried, L2-normalised numpy vectors."""
        self.embed_limiter.wait(EMBED_MODEL, 0, 1)
        res = self._retry(lambda: self.client.embed(
            model=EMBED_MODEL, texts=list(texts), input_type=input_type, embedding_types=["float"],
            output_dimension=EMBED_DIM, truncate="END"))
        with self._lock:
            self.calls["embed"] += 1
        out = []
        for v in self._floats(res):
            v = np.asarray(v, dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return out

    def embed(self, texts, input_type="search_query"):
        """Cached, batched embeddings. Queries use input_type='search_query'; stored chunks 'search_document'."""
        missing = list(dict.fromkeys(t for t in texts if (t, input_type) not in self._emb_cache))
        for i in range(0, len(missing), self.embed_batch):
            batch = missing[i:i + self.embed_batch]
            for t, v in zip(batch, self.embed_raw(batch, input_type)):
                self._emb_cache[(t, input_type)] = v
        return [self._emb_cache[(t, input_type)] for t in texts]

    def usage_report(self):
        return f"Cohere calls this run: {self.calls} (trial keys allow ~1,000 calls per month in total)"

    # ----------------------------------------------------------------- search
    def _vector(self, vec, k=10, where="", params=()):
        """Cosine search over the chosen chunking strategy and input format."""
        sql = f"""SELECT e.chunk_id FROM chunk_embeddings e JOIN syllabus_chunks c ON c.id = e.chunk_id
                  WHERE c.strategy = %s AND e.variant = %s {where}
                  ORDER BY e.embedding <=> %s::vector LIMIT %s"""
        with self.pool.connection() as c:
            return c.execute(sql, (self.strategy, self.variant, *params, vec, k)).fetchall()

    def _keyword(self, q, k=10):
        """Full-text OR-search (the 'simple' config keeps course codes like OCAIML4002 intact)."""
        words = [w for w in dict.fromkeys(re.findall(r"[A-Za-z0-9]+", q.lower())) if len(w) > 2]
        if not words:
            return []
        tsq = " | ".join(words[:25])
        with self.pool.connection() as c:
            return c.execute(
                """SELECT id FROM syllabus_chunks WHERE strategy = %s
                   AND to_tsvector('simple', content) @@ to_tsquery('simple', %s)
                   ORDER BY ts_rank(to_tsvector('simple', content), to_tsquery('simple', %s)) DESC LIMIT %s""",
                (self.strategy, tsq, tsq, k)).fetchall()

    def fetch(self, ids):
        """Chunks for ids, in the SAME order as ids (SQL ANY() does not keep order)."""
        ids = [i[0] if isinstance(i, (tuple, list)) else i for i in ids]
        if not ids:
            return []
        with self.pool.connection() as c:
            rows = c.execute("""SELECT id, booklet, course_code, course_title, page_start, page_end, content,
                                semester, section_type FROM syllabus_chunks WHERE id = ANY(%s)""", (ids,)).fetchall()
        by_id = {r[0]: {"id": r[0], "booklet": r[1], "code": r[2], "title": r[3], "p0": r[4], "p1": r[5],
                        "content": r[6], "semester": r[7], "section_type": r[8]} for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def dedupe(self, chunks):
        """Drop copies of the same text (e.g. regulations repeated in every booklet)."""
        seen, out = set(), []
        for c in chunks:
            body = re.sub(r"^\[[^\]\n]*\]\n", "", c["content"])           # ignore the [booklet | course | pages] header
            key = re.sub(r"\W+", " ", body.lower()).strip()[:300]
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    # ---------------------------------------------------------------- the LLM
    def _cache_key(self, model, task, system, prompt):
        return hashlib.sha256(f"{model}|{task}|{system}|{prompt}".encode()).hexdigest()

    def _cache_get(self, key):
        try:
            with self.pool.connection() as c:
                row = c.execute("SELECT response FROM llm_cache WHERE key = %s", (key,)).fetchone()
            return row[0] if row else None
        except Exception:                               # noqa: BLE001
            return None

    def _cache_put(self, key, model, task, text, usage, secs):
        try:
            with self.pool.connection() as c:
                c.execute("""INSERT INTO llm_cache (key, model, task, response, input_tokens, thinking_tokens,
                             output_tokens, seconds) VALUES (%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (key) DO NOTHING""",
                          (key, model, task, text, usage["input"], usage["thinking"], usage["output"], secs))
        except Exception as e:                          # noqa: BLE001
            self._warn("cache", f"llm_cache write skipped ({str(e)[:80]})")

    def _generate(self, model, prompt, system, json_mode):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        with self._llm_sem:
            self.chat_limiter.wait(model, 0, 1)
            t = time.time()
            r = self.client.chat(model=model, messages=messages, temperature=0.2, **kwargs)
            secs = time.time() - t
        with self._lock:
            self.calls["chat"] += 1
        tk = getattr(getattr(r, "usage", None), "tokens", None)
        usage = {"input": int(getattr(tk, "input_tokens", 0) or 0), "thinking": 0,
                 "output": int(getattr(tk, "output_tokens", 0) or 0)}
        text = "".join(getattr(p, "text", "") or "" for p in (r.message.content or []) if getattr(p, "type", "text") == "text").strip()
        if not text:
            raise ValueError("EMPTY_RESPONSE")
        return text, usage, secs

    def llm(self, prompt, system_instruction="", cache=True, model=None, task="other", json_mode=False):
        """One LLM call through the fallback chain. `model=` pins a single model (no fallback)."""
        chain = [model] if model else [self.llm_model, *self.fallbacks]
        last = None
        for m in chain:
            key = self._cache_key(m, task, system_instruction, prompt)
            if cache:
                hit = self._cache_get(key)
                if hit is not None:
                    return hit
            try:
                text, usage, secs = self._retry(lambda: self._generate(m, prompt, system_instruction, json_mode),
                                                waits=[2, 6, 20])
            except Exception as e:                      # noqa: BLE001
                if (type(e).__name__ == "BadRequestError" or "status_code: 400" in str(e)) and not isinstance(e, DailyQuotaError):
                    raise                               # our bug: another model won't help
                last = e                                # 404/403, empty reply, quota, retries used up → next model
                continue
            if cache:
                self._cache_put(key, m, task, text, usage, secs)
            return text
        raise RuntimeError(f"All models failed; last error: {last}")

    @staticmethod
    def _parse_json(text):
        """Pull a JSON object or array out of an LLM reply (handles ```json fences and chatter)."""
        if not text:
            return None
        text = re.sub(r"```(?:json)?", "", text).strip()
        for pat in (r"\{.*\}", r"\[.*\]"):
            m = re.search(pat, text, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:                       # noqa: BLE001
                    continue
        return None

    # ------------------------------------------------------- plan and routing
    def _default_plan(self, q):
        return {"standalone": q, "intent": "fact_lookup", "course_codes": [], "semesters": [],
                "rephrasings": [], "sub_questions": [], "keywords": [], "hyde": ""}

    def plan(self, q, history=None):
        """One LLM call: rewrite the question and decide intent, courses, semesters and search variants."""
        hist = ""
        if history:
            hist = "Conversation so far:\n" + "\n".join(f"{r}: {c[:300]}" for r, c in history[-6:]) + "\n\n"
        prompt = f"""You are the query planner of a syllabus assistant for the CHARUSAT B.Tech (IT) syllabus
(academic regulations, teaching schemes, course syllabi for 4 years / 8 semesters).
{hist}New user message: {q}

Reply with ONE JSON object only (no prose) with these keys:
"standalone": the message rewritten as a complete English question (resolve "it", "its" from the conversation; fix typos; translate Gujarati/Hindi written in Latin letters),
"intent": one of {INTENTS},
  (out_of_scope = not answerable from a syllabus, e.g. fees, hostel, canteen, placements, staff; chit_chat = greetings/thanks;
   list = "list all ..."; compare = compare courses; paper = question paper; mcq = MCQs; study_plan = study plan;
   overview = overview of one course; rule = regulations/rules; multi_step = needs several facts),
"course_codes": course codes mentioned or clearly meant, e.g. ["IT343"] (else []),
"semesters": semester numbers 1-8 involved (else []),
"rephrasings": 2 alternative phrasings using the booklet's likely wording,
"sub_questions": for multi-part questions, one search question per part (else []),
"keywords": up to 6 exact terms or codes worth a keyword search,
"hyde": 2 sentences that look like the syllabus passage which would answer the question."""
        try:
            data = self._parse_json(self.llm(prompt, task="plan"))
        except Exception:                               # noqa: BLE001
            data = None
        plan = self._default_plan(q)
        if isinstance(data, dict):
            plan.update({k: data[k] for k in plan if k in data and data[k] is not None})
        return self._normalize_plan(plan, q)

    def _normalize_plan(self, plan, q):
        as_list = lambda v: [str(x).strip() for x in v if str(x).strip()] if isinstance(v, list) else []
        plan["standalone"] = str(plan.get("standalone") or q).strip() or q
        intent = str(plan.get("intent", "fact_lookup")).strip().lower().replace(" ", "_")
        intent = {"mcqs": "mcq", "question_paper": "paper"}.get(intent, intent)
        plan["intent"] = intent if intent in INTENTS else "fact_lookup"
        codes = [c.upper().replace(" ", "") for c in as_list(plan.get("course_codes"))]
        codes += re.findall(r"\b[A-Z]{2,8}\d{3,4}(?:\.\d{1,2})?\b", q)        # codes typed in the question itself
        plan["course_codes"] = list(dict.fromkeys(codes))
        sems = []
        for s in plan.get("semesters") or []:
            try:
                if 1 <= int(s) <= 8:
                    sems.append(int(s))
            except (TypeError, ValueError):
                pass
        plan["semesters"] = sorted(set(sems))
        for k in ("rephrasings", "sub_questions", "keywords"):
            plan[k] = as_list(plan.get(k))[:6]
        plan["hyde"] = str(plan.get("hyde") or "")
        return plan

    def route(self, plan):
        intent = plan.get("intent", "fact_lookup")
        if intent in ("out_of_scope", "chit_chat"):
            return "refuse"
        if intent in ("paper", "mcq", "study_plan", "overview") and plan.get("course_codes"):
            return "whole_course"
        if intent in ("list", "compare"):
            return "catalog_search"
        return "search"

    # -------------------------------------------------------------- retrieval
    def retrieve(self, plan):
        """Multi-query hybrid retrieval: one ranked list per query variant, fused with RRF."""
        queries = {"standalone": plan["standalone"]}
        for i, r in enumerate(plan.get("rephrasings", []), 1):
            queries[f"rephrase {i}"] = r
        if plan.get("hyde"):
            queries["hyde"] = plan["hyde"]
        for i, s in enumerate(plan.get("sub_questions", []), 1):
            queries[f"sub {i}"] = s
        vecs = self.embed([self.query_text(t) for t in queries.values()])
        per_query = {name: [r[0] for r in self._vector(v, k=10)] for name, v in zip(queries, vecs)}
        per_query["keyword"] = [r[0] for r in self._keyword(
            " ".join([plan["standalone"], *plan.get("keywords", [])]), k=10)]
        if plan.get("course_codes"):                    # same question, but only inside the named courses
            per_query["course filter"] = [r[0] for r in self._vector(
                vecs[0], k=10, where="AND c.course_code = ANY(%s)", params=(plan["course_codes"],))]
        if plan.get("semesters"):
            per_query["semester filter"] = [r[0] for r in self._vector(
                vecs[0], k=10, where="AND c.semester = ANY(%s)", params=(plan["semesters"],))]
        return rrf([v for v in per_query.values() if v])[:40], per_query

    def whole_course(self, plan, limit=14):
        """Every chunk of the named courses, in order (for papers, MCQs, study plans, overviews)."""
        with self.pool.connection() as c:
            ids = [r[0] for r in c.execute(
                """SELECT id FROM syllabus_chunks WHERE strategy = %s AND course_code = ANY(%s)
                   ORDER BY course_code, chunk_index LIMIT %s""",
                (self.strategy, plan["course_codes"], limit)).fetchall()]
        return self.fetch(ids)

    def catalog_chunk(self, plan):
        """One synthetic chunk listing courses from the catalog table: complete by construction."""
        where, params = [], []
        if plan.get("semesters"):
            where.append("semester = ANY(%s)"); params.append(plan["semesters"])
        if plan.get("course_codes"):
            where.append("code = ANY(%s)"); params.append(plan["course_codes"])
        if "elective" in plan["standalone"].lower():
            where.append("elective_group IS NOT NULL")
        if not where:
            return None
        with self.pool.connection() as c:
            rows = c.execute(f"""SELECT code, title, booklet, semester, elective_group, page_start, page_end
                                 FROM syllabus_courses WHERE {' AND '.join(where)} ORDER BY semester, code""",
                             tuple(params)).fetchall()
        if not rows:
            return None
        text = "Course catalog (complete list for the filters asked):\n" + "\n".join(
            f"- {r[0]}: {r[1]} | {r[2]} | semester {r[3]}" + (f" | {r[4]}" if r[4] else "") + f" | pages {r[5]}-{r[6]}"
            for r in rows)
        return {"id": -1, "booklet": "Catalog", "code": None, "title": "Course catalog",
                "p0": min(r[5] for r in rows), "p1": max(r[6] for r in rows), "content": text}

    def rerank(self, plan, chunks, keep=10):
        """Cohere's rerank model scores the top 40 chunks against the question (1 call), then reorders them."""
        pool = chunks[:40]
        if len(pool) <= 1:
            return pool[:keep]
        try:
            self.rerank_limiter.wait(RERANK_MODEL, 0, 1)
            res = self._retry(lambda: self.client.rerank(
                model=RERANK_MODEL, query=plan["standalone"], documents=[c["content"][:2000] for c in pool],
                top_n=min(keep, len(pool))), waits=[3, 10, 30, 65])
            with self._lock:
                self.calls["rerank"] += 1
            out = []
            for r in res.results:
                c = pool[r.index]
                c["rerank"] = float(r.relevance_score)
                out.append(c)
            return out
        except DailyQuotaError:
            raise
        except Exception:                               # noqa: BLE001
            return pool[:keep]                          # keep the fused order

    # -------------------------------------------------------------- answering
    MODE_RULES = {
        "paper": "Write the exam question paper requested (default 70 marks for an end-semester paper). Spread the marks "
                 "over the units in proportion to the unit hours/weightage in the sources. Finish with a marks-allocation "
                 "table (unit, weightage, marks) and a check line showing the marks add up to the total.",
        "mcq": "Write the number of MCQs requested, each with 4 options, the correct option marked, and the unit it comes from.",
        "study_plan": "Write a day-by-day study plan for the number of days requested, covering every unit in proportion to its hours.",
        "overview": "Give an overview: credits, teaching hours, units with hours, course outcomes and the recommended books.",
        "list": "Answer with a complete list using the catalog and sources; do not leave any listed course out.",
        "compare": "Compare the courses side by side (credits, units, outcomes, books) in a short table.",
    }

    def answer(self, q, plan, chunks, model=None):
        """Write the final answer from the chunks only, with page citations. Returns (mode, text)."""
        mode = plan.get("intent", "fact_lookup")
        sources = "\n\n".join(f"[{c['booklet']} | {(c['code'] + ' ') if c['code'] else ''}{c['title'] or ''} | "
                              f"pages {c['p0']}-{c['p1']}]\n{c['content']}" for c in chunks)
        system = ("You answer questions about the CHARUSAT B.Tech (IT) syllabus for students and teachers.\n"
                  "Rules:\n- Use ONLY the sources below. Never use outside knowledge and never invent numbers, codes or pages.\n"
                  f"- If the sources do not contain the answer, reply exactly: \"{NOT_FOUND}\"\n"
                  "- Answer every part of a multi-part question.\n"
                  "- After each fact add its page in the form (p.12) or (pp.12-14), using only pages shown in the sources.\n"
                  "- If the same rule appears in several booklets, state it once.\n"
                  "- If a course name matches more than one course, mention each match.\n"
                  "- Be concise. Reply in English.\n" + self.MODE_RULES.get(mode, ""))
        prompt = f"SOURCES:\n{sources}\n\nQUESTION: {q}\n\nANSWER:"
        return mode, self.llm(prompt, system, model=model, task="answer")

    def check_citations(self, text, chunks):
        """Warn about cited pages that are not inside any source chunk. No LLM call."""
        ranges = [(c["p0"], c["p1"]) for c in chunks]
        warnings = []
        for m in re.finditer(r"(?:\bpp?\.|\bpages?\b|\bpg\.)\s*(\d+)(?:\s*[-–]\s*(\d+))?", text, re.I):
            a = int(m.group(1)); b = int(m.group(2) or a)
            if not all(any(lo <= p <= hi for lo, hi in ranges) for p in (a, b)):
                warnings.append(f"cites p.{a}" + (f"-{b}" if b != a else "") + " which is not in the sources")
        return list(dict.fromkeys(warnings))

    # ------------------------------------------------------------- the pipeline
    def ask(self, q, conversation_id=None, user_role="student"):
        """Plan → route → retrieve → (rerank) → answer → citation check, saved as a conversation turn."""
        t0, timings = time.time(), {}
        hist = self.history(conversation_id) if conversation_id else []
        t = time.time(); plan = self.plan(q, hist); timings["plan"] = round(time.time() - t, 2)
        route = self.route(plan)
        top, mode = [], plan["intent"]

        if route == "refuse":
            if plan["intent"] == "chit_chat":
                ans = ("Hi! I can answer questions about the CHARUSAT B.Tech (IT) syllabus: regulations, exam schemes, "
                       "courses, units, credits and books. I can also draft question papers, MCQs and study plans.")
            else:
                ans = (f"{NOT_FOUND} I can only answer from the B.Tech syllabus booklets "
                       "(regulations, teaching schemes and course syllabi).")
        else:
            t = time.time()
            if route == "whole_course":
                top = self.whole_course(plan)
            if not top:
                fused, _ = self.retrieve(plan)
                top = self.dedupe(self.fetch(fused))
                if route == "catalog_search":
                    cat = self.catalog_chunk(plan)
                    top = ([cat] if cat else []) + top[:7]
                elif self.use_rerank:
                    t2 = time.time(); top = self.rerank(plan, top); timings["rerank"] = round(time.time() - t2, 2)
                else:
                    top = top[:8]
            timings["retrieve"] = round(time.time() - t - timings.get("rerank", 0), 2)
            t = time.time(); mode, ans = self.answer(q, plan, top); timings["answer"] = round(time.time() - t, 2)

        warnings = self.check_citations(ans, top)
        timings["total"] = round(time.time() - t0, 2)
        cid = self._save_turn(conversation_id, user_role, q, ans)
        tid = self._save_trace({"question": q, "mode": mode, "route": route, "plan": plan, "warnings": warnings,
                                "timings": timings, "chunks": [(c["id"], c["p0"], c["p1"]) for c in top]})
        return {"conversation_id": cid, "trace_id": tid, "role": user_role, "question": q, "plan": plan,
                "route": route, "mode": mode, "answer": ans,
                "sources": [(c["booklet"], c["code"], c["p0"], c["p1"]) for c in top],
                "warnings": warnings, "seconds": time.time() - t0}

    # ------------------------------------------------------------ persistence
    def _save_turn(self, cid, role, q, ans):
        try:
            with self.pool.connection() as c:
                if not cid:
                    cid = c.execute("INSERT INTO conversations (title, role, turns) VALUES (%s, %s, 0) RETURNING id",
                                    (q[:60], role)).fetchone()[0]
                c.execute("INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s), (%s, 'assistant', %s)",
                          (cid, q, cid, ans))
                c.execute("UPDATE conversations SET turns = turns + 1 WHERE id = %s", (cid,))
        except Exception as e:                          # noqa: BLE001
            self._warn("turn", f"conversation not saved to Postgres ({str(e)[:80]}); kept in memory only")
            if not cid:
                with self._lock:
                    self._mem_next += 1
                    cid = self._mem_next
        self._mem_hist.setdefault(cid, []).extend([("user", q), ("assistant", ans)])
        return cid

    def _save_trace(self, trace):
        try:
            with self.pool.connection() as c:
                return c.execute("INSERT INTO rag_traces (trace_json) VALUES (%s) RETURNING id",
                                 (json.dumps(trace, default=str),)).fetchone()[0]
        except Exception:                               # noqa: BLE001
            return None

    def conversations(self, limit=8):
        with self.pool.connection() as c:
            return c.execute("SELECT id, title, role, created_at, turns FROM conversations "
                             "ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()

    def history(self, cid):
        try:
            with self.pool.connection() as c:
                rows = c.execute("SELECT role, content FROM messages WHERE conversation_id = %s "
                                 "ORDER BY created_at ASC, id ASC", (cid,)).fetchall()
            if rows:
                return rows
        except Exception:                               # noqa: BLE001
            pass
        return list(self._mem_hist.get(cid, []))

    def trace(self, tid):
        with self.pool.connection() as c:
            row = c.execute("SELECT trace_json FROM rag_traces WHERE id = %s", (tid,)).fetchone()
            if not row:
                return {}
            return row[0] if isinstance(row[0], dict) else json.loads(row[0])
