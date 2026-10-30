# app.py: chat window for the syllabus assistant
# run with:  streamlit run app.py
import datetime as dt
import streamlit as st
from common import setup

st.set_page_config(page_title="Syllabus Assistant", page_icon="🎓", layout="centered",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
  #MainMenu, footer, [data-testid="stDeployButton"], header[data-testid="stHeader"] {display: none;}
  /* we draw our own toggle, so hide Streamlit's built-in sidebar arrows */
  [data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"],
  [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] {display: none !important;}
  .block-container {max-width: 820px; padding-top: 1rem; padding-bottom: 6rem;}
  html, body, [class*="css"] {font-family: "Segoe UI", system-ui, -apple-system, sans-serif;}
  .st-key-topbar {border-bottom: 1px solid rgba(128,128,128,.25); padding-bottom: .4rem; margin-bottom: 1.2rem;}
  .st-key-topbar .stButton > button {border: none; background: transparent; box-shadow: none; font-size: 1.25rem;}
  .st-key-topbar .stButton > button:hover {background: rgba(128,128,128,.15);}
  .topbar-title {font-size: 1.02rem; font-weight: 600; padding-top: .45rem; white-space: nowrap;
                 overflow: hidden; text-overflow: ellipsis;}
  .hero h1 {font-size: 2rem; font-weight: 650; margin: 0 0 .25rem 0; letter-spacing: -0.01em;}
  .hero p {color: #6b7280; margin: 0 0 1.5rem 0; font-size: 1.02rem;}
  div[data-testid="stChatMessage"] {border-radius: 14px; padding: .9rem 1rem;}
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
ss.setdefault("cid", None)
ss.setdefault("messages", [])
ss.setdefault("role", "student")
ss.setdefault("sb_open", True)


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
    ss.messages = [{"role": r, "content": c, "sources": []} for r, c in rag.history(cid)]


def toggle_sidebar():
    ss.sb_open = not ss.sb_open


def group_of(created):
    try:
        days = (dt.date.today() - created.date()).days
    except Exception:
        return "Earlier"
    return "Today" if days <= 0 else "Previous 7 days" if days <= 7 else "Older"


# when closed, the sidebar is simply hidden; the toggle button below brings it back
if not ss.sb_open:
    st.markdown("<style>section[data-testid='stSidebar']{display:none !important;}</style>",
                unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎓 Syllabus Assistant")
    st.button("＋  New chat", on_click=new_chat, type="primary", use_container_width=True, key="new_side")
    st.text_input("Search chats", key="q_search", placeholder="Search chats…", label_visibility="collapsed")
    needle = (ss.get("q_search") or "").lower().strip()
    try:
        last = None
        for cid, title, role, created, turns in rag.conversations(limit=40):
            if needle and needle not in (title or "").lower():
                continue
            g = group_of(created)
            if g != last:
                st.caption(g)
                last = g
            st.button((title or "Untitled")[:34], key=f"c{cid}", on_click=open_chat, args=(cid, role))
    except Exception:
        st.caption("History is unavailable.")
    st.divider()
    ss.role = st.radio("I am a", ["student", "teacher"], horizontal=True,
                       index=0 if ss.role == "student" else 1)
    st.caption("Teachers can also ask for question papers and study plans.")

# ------------------------------------------------------------------ top bar (like Claude / ChatGPT)
with st.container(key="topbar"):
    c1, c2, c3 = st.columns([0.09, 0.69, 0.22], vertical_alignment="center")
    c1.button("☰", on_click=toggle_sidebar, key="toggle", help="Show / hide the menu")
    first_q = next((m["content"] for m in ss.messages if m["role"] == "user"), None)
    c2.markdown(f'<div class="topbar-title">{(first_q or "New chat")[:60]}</div>', unsafe_allow_html=True)
    c3.button("＋ New chat", on_click=new_chat, key="new_top", use_container_width=True)

# ------------------------------------------------------------------ empty state
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
    ss.messages.append({"role": "user", "content": prompt, "sources": []})
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

    ss.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()      # redraw: hides the intro text and adds this chat to the sidebar list
