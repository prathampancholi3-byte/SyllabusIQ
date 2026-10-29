# app.py: chat window for the CHARUSAT syllabus assistant
# run with:  streamlit run app.py
import streamlit as st
from common import setup

st.set_page_config(page_title="SyllabusIQ", page_icon="🎓", layout="centered")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
  #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}
  .block-container {max-width: 820px; padding-top: 2rem; padding-bottom: 6rem;}
  html, body, [class*="css"] {font-family: "Segoe UI", system-ui, -apple-system, sans-serif;}
  .hero h1 {font-size: 2rem; font-weight: 650; margin: 0 0 .25rem 0; letter-spacing: -0.01em;}
  .hero p {color: #6b7280; margin: 0 0 1.5rem 0; font-size: 1.02rem;}
  div[data-testid="stChatMessage"] {border-radius: 14px; padding: .9rem 1rem;}
  div[data-testid="stChatInput"] textarea {font-size: 1rem;}
  section[data-testid="stSidebar"] .stButton > button {
      width: 100%; text-align: left; border-radius: 10px; justify-content: flex-start;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
  .src {color: #6b7280; font-size: .86rem; line-height: 1.5;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ engine (loaded once)
@st.cache_resource(show_spinner="Connecting to the database…")
def load_engine():
    S = setup()
    rag = S.rag
    cfg = rag.load_config()
    assert cfg.get("chunking"), "No ingested syllabus found. Run 01_ingestion.py first (or fix_config.py)."
    rag.strategy, rag.variant = cfg["chunking"], cfg["variant"]
    rag.use_rerank = cfg.get("use_rerank", False)
    return rag


try:
    rag = load_engine()
except AssertionError as e:
    st.error(str(e))
    st.stop()

# ------------------------------------------------------------------ session state
ss = st.session_state
ss.setdefault("cid", None)          # current conversation id (None = new chat)
ss.setdefault("messages", [])       # [{"role", "content", "sources", "meta"}]
ss.setdefault("role", "student")


def fmt_sources(sources):
    out = []
    for b, c, p0, p1 in sources[:6]:
        pages = f"p.{p0}" + (f"-{p1}" if p1 != p0 else "")
        out.append(f"{b.split(' (')[0]} {c or ''} {pages}".replace("  ", " "))
    return out


def new_chat():
    ss.cid, ss.messages = None, []


def open_chat(cid, role):
    ss.cid, ss.role = cid, role or "student"
    ss.messages = [{"role": r, "content": c, "sources": [], "meta": ""} for r, c in rag.history(cid)]


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎓 Syllabus Assistant")
    st.button("＋  New chat", on_click=new_chat, type="primary", use_container_width=True)
    ss.role = st.radio("I am a", ["student", "teacher"], horizontal=True,
                       index=0 if ss.role == "student" else 1)
    st.caption("Teachers can also ask for question papers and study material.")
    st.divider()
    st.caption("Recent chats")
    try:
        for cid, title, role, created, turns in rag.conversations(limit=12):
            label = (title or "Untitled")[:34]
            st.button(label, key=f"c{cid}", on_click=open_chat, args=(cid, role))
    except Exception:
        st.caption("History is unavailable.")

# ------------------------------------------------------------------ header + empty state
SUGGESTIONS = [
    "What is the minimum attendance to sit the end-semester exam?",
    "Which course teaches Bharatanatyam?",
    "Give me an overview of CEUC102: credits, units and books.",
    "What do I get if I exit after the first year?",
]

pending = None
if not ss.messages:
    st.markdown('<div class="hero"><h1>What would you like to know?</h1>'
                '<p>Ask about courses, credits, books, attendance rules and exams in the syllabus.</p></div>',
                unsafe_allow_html=True)
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"s{i}", use_container_width=True):
            pending = s

# ------------------------------------------------------------------ chat history
for m in ss.messages:
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for line in m["sources"]:
                    st.markdown(f'<div class="src">📄 {line}</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ input
prompt = st.chat_input("Ask about the syllabus…") or pending
if prompt:
    ss.messages.append({"role": "user", "content": prompt, "sources": [], "meta": ""})
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Searching the syllabus…"):
            try:
                r = rag.ask(prompt, conversation_id=ss.cid, user_role=ss.role)
                ss.cid = r["conversation_id"]
                answer, sources = r["answer"], fmt_sources(r.get("sources") or [])
            except Exception as e:
                answer, sources = f"Something went wrong: {str(e)[:200]}. Try again in a moment.", []
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for line in sources:
                    st.markdown(f'<div class="src">📄 {line}</div>', unsafe_allow_html=True)

    ss.messages.append({"role": "assistant", "content": answer, "sources": sources, "meta": ""})
    st.rerun()      # redraw: hides the intro text and adds this chat to the sidebar list
