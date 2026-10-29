# 0. Setup
import os, sys
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

# Install dependencies (If running in a notebook, use: %pip install -q --disable-pip-version-check -r requirements.txt)
import json, re, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

# NOTE: The following modules are required but are not in the provided code snippet
from common import setup, score, save_eval, usd, USD_TO_INR
from rag_engine import rrf, RETRYABLE, EMBED_MODEL

pd.set_option("display.max_colwidth", 90)
S = setup()
client, conn, rag, WORKERS = S.client, S.conn, S.rag, S.workers

CFG = rag.load_config()
assert CFG.get("chunking"), "run 01_ingestion.py first"
rag.strategy, rag.variant = CFG["chunking"], CFG["variant"]
EVAL = conn.execute("SELECT question, gold_pages FROM eval_questions ORDER BY qid").fetchall()
assert EVAL, "no test questions in the database: 01_ingestion.py did not finish. Run it until it prints 'Ingestion script completed successfully', then run this file."
QUESTIONS, GOLD = [q for q, _ in EVAL], [g for _, g in EVAL]

# 1. Calling the LLM the robust way
r = client.chat(model=rag.llm_model, messages=[{"role": "user", "content": "In one sentence: what is retrieval-augmented generation (RAG)?"}])
tk = r.usage.tokens
print("".join(p.text for p in r.message.content if getattr(p, "type", "text") == "text").strip(), "\n")
tok = {"input": int(tk.input_tokens or 0), "thinking": 0, "output": int(tk.output_tokens or 0)}
cost = lambda m: usd(m, tok["input"], tok["thinking"] + tok["output"])
print(f"tokens {tok}")
print(f"cost = input × in-price + output × out-price → {rag.llm_model}: ${cost(rag.llm_model):.6f}"
      f" · same call on {rag.alt_model}: ${cost(rag.alt_model):.6f}  (a trial key itself is free)")

def decision(err):
    msg = str(err)
    if any(s in msg for s in RETRYABLE):
        return "transient: retry with backoff, then fall back"
    if type(err).__name__ == "BadRequestError" or "status_code: 400" in msg:
        return "our bug: fix the request (retrying or another model won't help)"
    return "this model/key can't serve it: fix the name or key, or fall back to the next model"

for label, call in [
    ("wrong model name", lambda: client.chat(model="command-9-ultra", messages=[{"role": "user", "content": "hi"}])),
    ("bad config",       lambda: client.chat(model=rag.llm_model, messages=[{"role": "user", "content": "hi"}], temperature=7)),
]:
    try:
        call()
    except Exception as e:
        print(f"{label:<17} {str(e)[:75]}…\n{'':<17} → {decision(e)}\n")
print("backoff waits:", " + ".join(f"{2 ** i}s" for i in range(5)), "≈ 31 s: enough for a per-minute limit to reset")

primary = rag.llm_model
rag.llm_model = "command-99-does-not-exist" 
try:
    print("answer:", rag.llm("Reply with exactly: fallback works", "Be brief.", cache=False))
finally:
    rag.llm_model = primary
print("chain:", [primary, *rag.fallbacks], "· all one provider: add a second provider (OpenAI / Gemini / local Ollama) to survive a full outage")

# 2. Two Test Sets
ANSWER_EVAL = [
    ("What is the minimum attendance needed to sit the end-semester exam?", [r"75\s?%"]),
    ("What CGPA do I need for First Class with Distinction?", [r"7\.0"]),
    ("If I exit after the first year, what do I get?", [r"certificate"]),
    ("Which course teaches Bharatanatyam?", [r"HSUS117"]),
    ("Which textbook is recommended for Engineering Mathematics-I?", [r"Kreyszig"]),
    ("Which course uses Fusion 360?", [r"Fusion 360"]),
    ("What is the exit course code ITUS102?", [r"ITUS102"]),
    ("Who wrote the recommended book 'Let us C'?", [r"Kanetkar|Let us C"]),
    ("What is the hostel fee?", [r"Not found|not (available|mentioned|covered|included)|outside"]),
]
passes = lambda text, checks: all(re.search(rx, text, re.I) for rx in checks)
print(f"{len(ANSWER_EVAL)} answer checks")

# 3. The improvement ladder: v0 → v4
VERSIONS = ["v0 naive", "v1 hybrid", "v2 + planner", "v3 + rerank"]
CALLS = {"v0 naive": 1, "v1 hybrid": 1, "v2 + planner": 2, "v3 + rerank": 5}

def retrieve(q, version):
    if version in ("v0 naive", "v1 hybrid"):
        plan = {"standalone": q, "intent": "fact_lookup", "sub_questions": [], "params": {}}   
        vec = [r[0] for r in rag._vector(rag.embed([rag.query_text(q)])[0], k=10)]
        ids = vec if version == "v0 naive" else rrf([vec, [r[0] for r in rag._keyword(q, k=10)]])
        return plan, rag.fetch(ids)[:10]
    plan = rag.plan(q)
    ids, _ = rag.retrieve(plan)
    chunks = rag.dedupe(rag.fetch(ids))
    return plan, (rag.rerank(plan, chunks, keep=10) if version == "v3 + rerank" else chunks[:10])

as_rows = lambda chunks: [(c["id"], c["p0"], c["p1"]) for c in chunks]

t = time.time()
RET = {}
for v in VERSIONS:
    with ThreadPoolExecutor(WORKERS) as ex:
        RET[v] = [as_rows(chunks) for _, chunks in ex.map(lambda q: retrieve(q, v), QUESTIONS)]
    save_eval(rag, f"ladder·{v}", RET[v])
RS = {v: score(RET[v], GOLD) for v in VERSIONS}
print(f"{len(VERSIONS) * len(QUESTIONS)} retrievals in {time.time() - t:.0f}s")
print(pd.DataFrame(RS).T.round(2).to_string())

def answer(q, version, model=None):
    plan, chunks = retrieve(q, version)
    return rag.answer(q, plan, chunks[:8], model=model)[1]

t = time.time()
jobs = [(i, v) for i in range(len(ANSWER_EVAL)) for v in VERSIONS]
with ThreadPoolExecutor(WORKERS) as ex:
    texts = dict(zip(jobs, ex.map(lambda j: answer(ANSWER_EVAL[j[0]][0], j[1]), jobs)))
print(f"{len(jobs)} answers in {time.time() - t:.0f}s")

GRID = pd.DataFrame({v: ["✅" if passes(texts[(i, v)], ANSWER_EVAL[i][1]) else "❌" for i in range(len(ANSWER_EVAL))]
                     for v in VERSIONS}, index=[q[:70] for q, _ in ANSWER_EVAL])
ACC = {v: (GRID[v] == "✅").mean() for v in VERSIONS}
GRID.loc["accuracy"] = [f"{ACC[v]:.0%}" for v in VERSIONS]
print(GRID.to_string())

best_v = max(VERSIONS, key=lambda v: (ACC[v], RS[v]["MRR"]))
worst_v = min(VERSIONS, key=lambda v: (ACC[v], RS[v]["MRR"]))
i = next((i for i in range(len(ANSWER_EVAL)) if GRID.iloc[i][best_v] == "✅" and GRID.iloc[i][worst_v] == "❌"), None)
if i is not None:
    print("Q:", ANSWER_EVAL[i][0], "\n")
    print(f"❌ {worst_v}:\n{texts[(i, worst_v)][:500]}\n")
    print(f"✅ {best_v}:\n{texts[(i, best_v)][:500]}")

summary = pd.DataFrame({v: {**RS[v], "answers correct": ACC[v], "LLM calls / question": CALLS[v]} for v in VERSIONS}).T  
print(summary.round(2).to_string())
pick = lambda vs: max(vs, key=lambda v: (ACC[v], RS[v]["MRR"], -CALLS[v]))
BEST = pick(VERSIONS)
rag.use_rerank = pick(["v2 + planner", "v3 + rerank"]) == "v3 + rerank"
rag.save_config(search_version=BEST, use_rerank=rag.use_rerank,
                ladder={v: {**{k: round(x, 3) for k, x in RS[v].items()}, "answers": round(ACC[v], 3)} for v in VERSIONS})

fig, ax = plt.subplots(figsize=(8, 2.8))
x, w = np.arange(len(VERSIONS)), 0.27
for j, (col, colour) in enumerate([("Hit@5", "#90caf9"), ("MRR", "#1e88e5"), ("answers correct", "#2e7d32")]):
    ax.bar(x + (j - 1) * w, summary[col], w, label=col, color=colour)
ax.set_xticks(x, VERSIONS); ax.set_ylim(0, 1.1); ax.legend(ncol=3, fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False); ax.set_title("Each version, measured")
fig.tight_layout(); fig.savefig("ladder.png", dpi=150); plt.close(fig)
print("chart saved → ladder.png")

print(f"best overall: {BEST} · rerank in rag.ask(): {'ON' if rag.use_rerank else 'OFF'}")

SEARCH = "v3 + rerank" if rag.use_rerank else "v2 + planner"
MODELS = [rag.llm_model, rag.alt_model]
jobs = [(i, m) for i in range(len(ANSWER_EVAL)) for m in MODELS]
with ThreadPoolExecutor(WORKERS) as ex:
    swap = dict(zip(jobs, ex.map(lambda j: answer(ANSWER_EVAL[j[0]][0], SEARCH, model=j[1]), jobs)))

usage = {m: conn.execute("""SELECT avg(input_tokens), avg(coalesce(thinking_tokens, 0) + output_tokens), avg(seconds)
                            FROM llm_cache WHERE model = %s AND task = 'answer' AND input_tokens IS NOT NULL""", (m,)).fetchone()
         for m in MODELS}

# 4. Routing
EXAMPLES = ["What is the hostel fee?", "hi!", "Prepare a 50-mark paper for Programming with C++ (CEUC102).",
            "Make a 7-day study plan for Programming with C++.", "List all courses in semester 1.",
            "Compare Community Engagement and Environmental Sciences.",
            "Can I move to the next year without clearing all courses?", "ITUS102?"]
with ThreadPoolExecutor(WORKERS) as ex:
    plans = list(ex.map(rag.plan, EXAMPLES))
print(pd.DataFrame([{"question": q, "intent": p["intent"], "courses": ", ".join(p["course_codes"]),
                     "semesters": p["semesters"], "route": rag.route(p)}
                    for q, p in zip(EXAMPLES, plans)]).to_string(index=False))
print()
# 5. One question, step by step
q = "What is the minimum attendance needed, and what happens to a student who does not have it?"
plan = rag.plan(q)
print("① PLAN  (one LLM call, JSON out)")
print(json.dumps({k: plan[k] for k in ("standalone", "intent", "course_codes", "semesters", "rephrasings", "sub_questions", "keywords")},
                 indent=2, ensure_ascii=False))

fused, per_query = rag.retrieve(plan)
pool = rag.fetch(fused)
deduped = rag.dedupe(list(pool))
top = rag.rerank(plan, deduped) if rag.use_rerank else deduped[:8]
mode, text = rag.answer(q, plan, top)
print(f"⑥ ANSWER (mode: {mode})\n\n{text}\n")
print("⑦ CITATION CHECK:", rag.check_citations(text, top) or "every cited page is in the sources ✓")

# 6. Question Bank Execution
BANK = [
    ("student", ["What's the minimum attendance to be allowed in end-sem exams?"]),
    ("student", ["ITUS102?"]),
    ("student", ["Give me an overview of the Programming with C++ course (CEUC102): credits, units and books."]),
    ("teacher", ["Prepare a 70-mark end-semester question paper for Programming with C++ (CEUC102)."]),
]

def run_chain(item):
    role, turns = item
    cid, out = None, []
    for q in turns:
        r = rag.ask(q, conversation_id=cid, user_role=role)
        cid = r["conversation_id"]
        out.append({"role": role, "question": q, **r})
    return out

t = time.time()
with ThreadPoolExecutor(WORKERS) as ex:
    RESULTS = [r for chain in ex.map(run_chain, BANK) for r in chain]

def show(i):
    r = RESULTS[i - 1]
    print(f"#{i} [{r['role']} · {r['route']} · {r['mode']}]  ❓ {r['question']}\n")
    print(r["answer"])
    print("\n📄", ", ".join(f"{b.split(' (')[0]} {c or ''} p.{p0}" + (f"-{p1}" if p1 != p0 else "") for b, c, p0, p1 in r["sources"][:8]))
    print("─" * 100)

for i in (1, 2):          
    show(i)

# 7. Grounding and 8. Saved Conversations
OUT_OF_SCOPE = ["What is the hostel fee?", "Who is the head of the IT department?"]
for q in OUT_OF_SCOPE:
    r = rag.ask(q)
    print(f"\n❓ {q} [route: {r['route']}]\n{r['answer'][:300]}")

print(pd.DataFrame(rag.conversations(limit=8),
                   columns=["conversation_id", "title", "role", "created", "turns"]).to_string(index=False))