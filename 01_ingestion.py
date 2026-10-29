# 0. Setup
import os, sys, re, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader
from common import setup, score, is_hit, save_eval, usd, PDF_PATH, EMBED_MODEL, EMBED_DIM, USD_TO_INR
from rag_engine import DailyQuotaError

# Colab only (does nothing on a laptop)
if "google.colab" in sys.modules:
    from glob import glob
    from google.colab import drive, userdata
    drive.mount("/content/drive")
    FOLDER = ""   # e.g. "/content/drive/MyDrive/charusat-syllabus-rag"; empty = search My Drive for rag_engine.py
    if not FOLDER:
        hits = glob("/content/drive/MyDrive/**/rag_engine.py", recursive=True)
        assert hits, "rag_engine.py not found in My Drive: put common.py, rag_engine.py, requirements.txt and the PDF next to this notebook"
        FOLDER = os.path.dirname(hits[0])
        if len(hits) > 1:
            print(f"{len(hits)} copies found, using the first; set FOLDER to pick another")
    os.chdir(FOLDER); sys.path.insert(0, FOLDER)
    for name in ("COHERE_API_KEY", "COHERE_TIER", "WORKERS"):          # secrets; default tier: trial
        try:
            os.environ.setdefault(name, userdata.get(name))
        except Exception:  
            pass
    print(f"folder  {FOLDER}")

pd.set_option("display.max_colwidth", 90)
S = setup()
client, conn, rag, WORKERS = S.client, S.conn, S.rag, S.workers

# The database health check
tables = ["syllabus_chunks", "chunk_embeddings", "syllabus_courses", "eval_questions", "pipeline_config",
          "query_embeddings", "llm_cache", "conversations", "messages", "rag_traces"]
print(pd.Series({t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in tables}, name="rows").to_frame())

# 1. Load and clean the PDF
BOOKLETS = [(1, 156, "First Year (FY)"), (157, 277, "Second Year (SY)"),
            (278, 359, "Third Year (TY)"), (360, 405, "Final Year")]
booklet_of = lambda pg: next(name for a, b, name in BOOKLETS if a <= pg <= b)

NOISE = re.compile(r"@\s*CHARUSAT\s+ACADEMIC YEAR.*|P\s?a\s?g\s?e\s*\|\s*\d+|Page\s+\d+\s+of\s+\d+|"
                   r"©\s*CHARUSAT\s*\d{4}.*?Academic Year\s*\d{4}\s*-\s*\d{2}", re.I)
SEM_DIVIDER = re.compile(r"SYLLABI\s*\(\s*Semester\s*[–-]\s*(\d)", re.I)

def clean(text):
    text = NOISE.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

raw = [p.extract_text() or "" for p in PdfReader(PDF_PATH).pages]
pages, sem, prev = [], None, None
for i, text in enumerate(raw):
    booklet = booklet_of(i + 1)
    if booklet != prev:
        sem, prev = None, booklet                    
    text = clean(text)
    if (m := SEM_DIVIDER.search(text)) and i > 2:    
        sem = int(m.group(1))
    pages.append({"page": i + 1, "booklet": booklet, "semester": sem, "text": text})

n_raw, n_clean = sum(map(len, raw)), sum(len(p["text"]) for p in pages)
print(f"{len(pages)} pages · {n_raw:,} raw characters → {n_clean:,} after cleaning ({1 - n_clean / n_raw:.0%} was noise)")

# 2. Chunking, three ways
PAGE = {p["page"]: p for p in pages}

def concat(pgs):
    full, starts = "", []
    for p in pgs:
        starts.append((len(full), p["page"]))
        full += p["text"] + "\n\n"
    return full, starts

page_at = lambda starts, offset: max((pg for s, pg in starts if s <= offset), default=starts[0][1])

def chunk(strategy, i, text, p0, p1, code=None, title=None):
    return dict(strategy=strategy, chunk_index=i, booklet=booklet_of(p0), course_code=code, course_title=title,
                semester=PAGE[p0]["semester"], page_start=p0, page_end=p1, content=text.strip())

# ── A · fixed size 
def chunk_fixed(pgs, size=1000):
    full, starts = concat(pgs)
    return [chunk("fixed", n, full[s:s + size], page_at(starts, s), page_at(starts, min(len(full), s + size) - 1))
            for n, s in enumerate(range(0, len(full), size))]

# ── B · recursive with overlap
SEPARATORS = ["\n\n", "\n", ". ", " "]

def split_recursive(text, size, seps=SEPARATORS):
    if len(text) <= size or not seps:
        return [text]
    out, cur = [], ""
    for part in text.split(seps[0]):
        cand = f"{cur}{seps[0]}{part}" if cur else part
        if len(cand) <= size:
            cur = cand
            continue
        if cur:
            out.append(cur)
        if len(part) > size:                           
            *done, cur = split_recursive(part, size, seps[1:])
            out.extend(done)
        else:
            cur = part
    return out + ([cur] if cur else [])

def chunk_recursive(pgs, size=1200, overlap=200, strategy="recursive"):
    full, starts = concat(pgs)
    out, pos, tail = [], 0, ""
    for piece in split_recursive(full, size - overlap):
        start = full.find(piece, pos)
        start = pos if start < 0 else start
        pos = start + len(piece)
        text = f"{tail} {piece}" if tail else piece
        p0, p1 = page_at(starts, max(0, start - len(tail))), page_at(starts, pos - 1)
        tail = piece[-overlap:]
        if text.strip():
            out.append(chunk(strategy, len(out), text, p0, p1))
    return out

# ── C · structure-aware sections + context header
COURSE_HEADINGS = [
    re.compile(r"^Subject:\s*(?P<title>.+?)\s*\((?P<code>[A-Z]{2,8}\d{3,4})[^)]*\)"),                       
    re.compile(r"^(?P<code>[A-Z]{2,8}\s?(?:\d{3,4}|X{3,4})(?:\.\d{1,2})?)\s*:\s*(?P<title>[A-Za-z].{2,})$"),  
    re.compile(r"^(?P<code>[A-Z]{2,8}\d{3,4}\.\d{1,2})\s+(?P<title>[A-Z][A-Z &,\-–()0-9.]{4,})$"),          
]
SECTION_HEADINGS = [(re.compile(r"TEACHING\s*&\s*EXAMINATION\s*SCHEME", re.I), "Teaching & Examination Scheme"),
                    (re.compile(r"ACADEMIC\s+REGULATIONS", re.I), "Academic Regulations")]

def detect_heading(text):
    head = [l for l in text.splitlines() if l.strip()][:8]              
    for line in head:
        if "...." in line or line.upper().startswith("ISBN"):            
            continue
        for rx in COURSE_HEADINGS:
            if m := rx.search(line):
                return m["code"].replace(" ", ""), m["title"].strip(" -–")
    for rx, name in SECTION_HEADINGS:
        if rx.search("\n".join(head)):
            return None, name
    return None

def split_sections(pgs):
    secs, cur = [], None
    for p in pgs:
        h = detect_heading(p["text"])
        if h or cur is None or cur["booklet"] != p["booklet"]:
            code, title = h or (None, "Academic Regulations")
            if cur:
                secs.append(cur)
            cur = dict(booklet=p["booklet"], code=code, title=title, pages=[])
        cur["pages"].append(p)
    return secs + [cur]

def chunk_structure(pgs, size=1600, overlap=200):
    out = []
    for s in split_sections(pgs):
        name = f"{s['code']} {s['title']}" if s["code"] else s["title"]
        p0, p1 = s["pages"][0]["page"], s["pages"][-1]["page"]
        header = f"[{s['booklet']} | {name} | pages {p0}-{p1}]"
        for c in chunk_recursive(s["pages"], size, overlap):
            out.append(chunk("structure", len(out), f"{header}\n{c['content']}", c["page_start"], c["page_end"],
                             s["code"], s["title"]) | {"semester": s["pages"][0]["semester"]})
    return out

CHUNKS = {"fixed": chunk_fixed(pages), "recursive": chunk_recursive(pages), "structure": chunk_structure(pages)}
SECTIONS = split_sections(pages)

# 3. Metadata
def section_type(c):
    t = c["content"]
    digits = sum(ch.isdigit() for ch in t) / max(1, len(t))
    if re.search(r"Articulation Matrix|PO\s?1\s+PO\s?2", t) and digits > 0.08: return "co_po_matrix"
    if c["course_code"] is None:
        return "scheme" if "Scheme" in (c["course_title"] or "") else "regulation"
    if re.search(r"Text\s?Books?|Reference\s?Books?|Recommended Study Material|URL Links", t, re.I): return "references"
    if re.search(r"\bCO\s?\d\b|Course Outcomes?|Learning Outcome", t): return "outcomes"
    if re.search(r"\bUnit\b|\bHours?\b|\bWeek\s?\d|Detailed Syllabus|Outline of the Course", t, re.I): return "units"
    if re.search(r"Teaching Scheme|Credits? and Hours|Marks", t, re.I): return "scheme"
    return "other"

for chunks in CHUNKS.values():
    for c in chunks:
        c["section_type"] = section_type(c)

ELECTIVE = re.compile(r"(University\s+)?Elective\s*[–-]?\s*([\dIV.]+)", re.I)
catalog = {}
for s in SECTIONS:
    if s["code"] and s["code"] not in catalog:
        m = ELECTIVE.search(s["title"])
        catalog[s["code"]] = dict(code=s["code"], title=re.sub(r"\s*\(\s*(University\s+)?Elective.*?\)", "", s["title"]).strip(),
                                  booklet=s["booklet"], semester=s["pages"][0]["semester"],
                                  elective_group=m.group(0).replace("–", "-") if m else None,
                                  page_start=s["pages"][0]["page"], page_end=s["pages"][-1]["page"])

COLS = ["strategy", "chunk_index", "booklet", "course_code", "course_title", "semester", "section_type",
        "page_start", "page_end", "content"]

for strategy, chunks in CHUNKS.items():
    have = conn.execute("SELECT count(*) FROM syllabus_chunks WHERE strategy = %s", (strategy,)).fetchone()[0]
    if not have:
        with conn.cursor() as cur, cur.copy(f"COPY syllabus_chunks ({', '.join(COLS)}) FROM STDIN") as cp:
            for c in chunks:
                cp.write_row([c[k] for k in COLS])
    print(f"{strategy:<9} {len(chunks):>4} chunks  {'(already stored)' if have else 'stored ✓'}")

with conn.cursor() as cur:
    cur.executemany("""INSERT INTO syllabus_courses VALUES (%(code)s, %(title)s, %(booklet)s, %(semester)s,
                       %(elective_group)s, %(page_start)s, %(page_end)s)
                       ON CONFLICT (code) DO UPDATE SET title = EXCLUDED.title, elective_group = EXCLUDED.elective_group""",
                    list(catalog.values()))
print("catalog stored ✓")

# 4. Embeddings
def doc_text(content, title, variant):
    # "prefixed" puts the course title in front of the text; Cohere itself separates documents from queries via input_type
    return content if variant == "raw" else f"title: {title or 'none'} | text: {content}"

def query_text(q, variant):
    return q

def embed_and_store(rows):
    texts = [doc_text(r[1], r[2], v) for v, r in rows]
    vecs = rag.embed_raw(texts, "search_document")
    with rag.pool.connection() as c, c.cursor() as cur, cur.copy(
            "COPY chunk_embeddings (chunk_id, variant, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
        cp.set_types(["int8", "text", "vector"])
        for (v, r), e in zip(rows, vecs):
            cp.write_row([r[0], v, e])
    return len(rows)

RUNS = [("structure", "prefixed"), ("structure", "raw"), ("recursive", "raw"), ("fixed", "raw")]   

def missing(strategy, variant):
    return conn.execute("""SELECT c.id, c.content, c.course_title FROM syllabus_chunks c
                           LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id AND e.variant = %s
                           WHERE c.strategy = %s AND e.chunk_id IS NULL ORDER BY c.id""", (variant, strategy)).fetchall()

todo = [(v, r) for s, v in RUNS for r in missing(s, v)]              
t = time.time()
if todo:
    print(f"{len(todo):,} chunks to embed · {S.tier} tier"
          + (" · 96 texts per call, paced under the trial limit (a trial key allows ~1,000 calls per month)" if S.tier == "trial" else ""))
    parts = [todo[i:i + rag.embed_batch] for i in range(0, len(todo), rag.embed_batch)]
    failed = []

    def safe_embed(batch):
        """One bad batch (network drop, timeout) must not kill the run: log it and carry on."""
        try:
            return embed_and_store(batch)
        except DailyQuotaError:
            raise
        except Exception as e:                      # noqa: BLE001
            failed.append(len(batch))
            print(f"  batch of {len(batch)} failed ({type(e).__name__}: {str(e)[:80]}): will be retried on the next run")
            return 0

    try:
        # trial keys are rate-limited per call, so extra threads only add 429s
        with ThreadPoolExecutor(2 if S.tier == "trial" else WORKERS) as pool:
            done = 0
            for n in pool.map(safe_embed, parts):
                done += n
                if n:
                    print(f"  embedded {done:,} / {len(todo):,}", end="\r")
    except DailyQuotaError:
        left = sum(len(missing(s, v)) for s, v in RUNS)
        raise RuntimeError(f"Cohere trial quota used up: {left:,} of {len(todo):,} chunks still to embed. Everything "
                           "embedded so far is saved. Trial keys allow ~1,000 calls per month: run again next month or "
                           "with a production key; it continues where it stopped.") from None
if todo and failed:
    raise RuntimeError(f"{sum(failed)} chunks failed (network/timeouts). Everything else is saved: run this file "
                       "again and it embeds only what is missing.")
print(f"embedded {len(todo)} chunks in {time.time() - t:.0f}s" if todo else "all chunks already embedded ✓")

# 5. Experiments & Evaluation Data Setup
EVAL = [  
    ("What is the minimum attendance required to appear in the examination?", r"minimum attendance of 75"),
    ("What CGPA is needed for First Class with Distinction?", r"First Class with Distinction"),
    ("What minimum CGPA must a student secure to be awarded the degree?", r"minimum CGPA of 4\.5"),
    ("What are the minimum passing marks in the university exam and overall per subject?", r"Minimum marks in University"),
    ("What does a student receive if they exit the programme after the first year?", r"Exit after First Year"),
    ("Which second-year course teaches sliding window protocols?", r"Sliding Window Protocols"),
    ("How many credits and hours per week does the Minor Project carry?", r"IT452: MINOR PROJECT"),
    ("What is the minimum duration of Summer Internship-II?", r"at least 90 hours"),
    ("Which elective introduces Large Language Models and which NPTEL course is it?", r"OCAIML4002: Introduction to Large Language Models"),
    ("What units are covered in the Cloud Computing course?", r"IT367: CLOUD COMPUTING"),
    ("What topics does the AI & Machine Learning course (IT362) cover?", r"IT362: ARTIFICIAL INTELLIGENCE"),
    ("Which textbooks are recommended for Engineering Mathematics?", r"Kreyszig"),
    ("How is SGPA calculated?", r"SGPA = "),
    ("What happens if a student scores less than 45% in the final examination?", r"will get a FF grade"),
    ("How many credits is the Major Project in semester 8?", r"IT461:\s*MAJOR PROJECT"),
    ("What is the course code for Google Cloud Computing Foundations?", r"Google Cloud Computing Foundations \(CSUE202\)"),
    ("Which course teaches Bharatanatyam?", r"HSUS117"),
    ("Can a student move to fourth year without clearing all second-year courses?", r"not be allowed to move to fourth year"),
    ("Which semester is Design and Analysis of Algorithms taught in?", r"Design and Analysis of Algorithms \(ITUE204\)"),
    ("ITUE205", r"\(ITUE205\)"),                                       
    ("What is the course layout of the Reinforcement Learning elective?", r"OCCSE4001: Reinforcement Learning"),
    ("What are the criteria to get a certificate for SWAYAM/NPTEL elective courses?", r"Criteria to get a certificate"),
    ("How are activity points earned in Skill Augmentation?", r"ACTIVITY POINT"),
    ("What is the teaching scheme for Fundamentals of Computer Networks?",
     r"ITUC201 Fundamentals of Computer Networks|Fundamentals of Computer Networks \(ITUC201\)"),
]
# Extra questions that exist in the FIRST-YEAR booklet (used when your PDF is only that booklet)
EVAL += [
    ("Which course covers basic sensors interfacing using Arduino?", r"basic sensors interfacing using Arduino"),
    ("Which textbook by M. Morris Mano is recommended?", r"M\. Morris Mano"),
    ("What is the exit course ITUS102?", r"ITUS102"),
    ("Which course teaches Fusion 360?", r"Fusion 360"),
    ("What is the minimum percentage of marks in the quiz?", r"minimum 45% of marks in the quiz"),
    ("What does the Programming with C++ course (CEUC102) cover?", r"CEUC102: PROGRAMMING WITH C\+\+"),
    ("How many activity points are needed annually?", r"minimum of 25 points annually"),
    ("When can a student move to the next year?", r"cleared all the courses of the"),
    ("Which book 'Let us C' is recommended?", r"Let us C"),
    ("What is Community Engagement (CUUV101)?", r"CUUV101: COMMUNITY ENGAGEMENT"),
    ("What is Environmental Sciences (CLUV101)?", r"CLUV101: ENVIRONMENTAL SCIENCES"),
]
flat = {p["page"]: re.sub(r"\s+", " ", p["text"]) for p in pages}
_all = [(e, sorted(pg for pg, t in flat.items() if re.search(e[1], t, re.I))) for e in EVAL]
_dropped = [e[0] for e, g in _all if not g]
if _dropped:
    print(f"⚠ {len(_dropped)} questions have no matching page in this PDF and were skipped "
          f"(they target syllabus booklets that are not in {PDF_PATH}):")
    for q_ in _dropped: print("   -", q_)
EVAL = [e for e, g in _all if g]
GOLD = [g for e, g in _all if g]
assert len(EVAL) >= 5, "too few questions match this PDF: check PDF_PATH in common.py"

with conn.cursor() as cur:
    cur.execute("DELETE FROM eval_questions")
    cur.executemany("""INSERT INTO eval_questions VALUES (%s, %s, %s, %s) ON CONFLICT (qid) DO UPDATE
                       SET question = EXCLUDED.question, evidence = EXCLUDED.evidence, gold_pages = EXCLUDED.gold_pages""",
                    [(i, q, rx, g) for i, ((q, rx), g) in enumerate(zip(EVAL, GOLD))])

def search_all(strategy, variant, k=10):
    qvecs = rag.embed([query_text(q, variant) for q, _ in EVAL])
    def one(qv):
        with rag.pool.connection() as c:
            return c.execute("""SELECT c.id, c.page_start, c.page_end, c.content FROM chunk_embeddings e
                                JOIN syllabus_chunks c ON c.id = e.chunk_id WHERE c.strategy = %s AND e.variant = %s
                                ORDER BY e.embedding <=> %s LIMIT %s""", (strategy, variant, qv, k)).fetchall()
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(one, qvecs))

R = {s: search_all(s, "raw") for s in ("fixed", "recursive", "structure")}
for name, res in R.items():
    save_eval(rag, f"{name}·raw", res)

R["structure·prefixed"] = search_all("structure", "prefixed")
save_eval(rag, "structure·prefixed", R["structure·prefixed"])

BEST_VARIANT = max(["raw", "prefixed"], key=lambda v: score(R["structure" + ("·prefixed" if v == "prefixed" else "")], GOLD)["MRR"])
rows = conn.execute("""SELECT c.id, c.page_start, c.page_end, c.content, e.embedding
                       FROM chunk_embeddings e JOIN syllabus_chunks c ON c.id = e.chunk_id
                       WHERE c.strategy = 'structure' AND e.variant = %s ORDER BY c.id""", (BEST_VARIANT,)).fetchall()

# 7. Decisions - Save config for Notebook 2
chunk_mrr = {s: score(R[s], GOLD)["MRR"] for s in ("fixed", "recursive", "structure")}
FINAL = {"chunking": max(chunk_mrr, key=chunk_mrr.get),
         "variant": BEST_VARIANT,
         "dims": EMBED_DIM,                          
         "ann_index": "none (exact scan)"}
best = score(R["structure·prefixed" if BEST_VARIANT == "prefixed" else "structure"], GOLD)
rag.save_config(**FINAL, ingestion_scores={k: round(v, 3) for k, v in best.items()})

print(rag.usage_report())
print("Ingestion script completed successfully. You can now run 02_retrieval.py.")