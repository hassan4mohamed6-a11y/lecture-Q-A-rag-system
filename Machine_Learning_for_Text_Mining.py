# ============================================================================
# LECTURE Q&A SYSTEM — FINAL VERSION
# ─────────────────────────────────────────────────────────────────────────────
# Stack:
#   Embedding  : BAAI/bge-large-en-v1.5  (1024-dim)
#   Vector DB  : FAISS IndexFlatIP       (exact cosine, no RAM waste)
#   Reranker   : cross-encoder/ms-marco-MiniLM-L-6-v2
#   LLM        : Groq  llama-3.1-8b-instant  (fast + free tier)
#   Chunking   : Sentence-aware with overlap
#   Multi-turn : Last N Q&A pairs injected into the LLM prompt
#   Extras     : Important Notes  |  Quiz Generator
#
# Run: streamlit run lecture_qa_final.py
# ============================================================================

# ── Bootstrap: re-launch under Streamlit when executed with plain python ─────
import sys, os, subprocess

if __name__ == "__main__" and os.environ.get("STREAMLIT_LAUNCHED") != "true":
    env = {**os.environ, "STREAMLIT_LAUNCHED": "true"}
    subprocess.run([sys.executable, "-m", "streamlit", "run", os.path.abspath(__file__)], env=env)
    sys.exit(0)

# ── Silence noisy runtime warnings ───────────────────────────────────────────
os.environ.setdefault("KMP_DUPLICATE_LIB_OK",     "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM",    "false")

# ============================================================================
# IMPORTS
# ============================================================================
import json
import re
import tempfile
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import faiss
import fitz                                     # PyMuPDF
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import CrossEncoder, SentenceTransformer

# ============================================================================
# CONSTANTS
# ============================================================================
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL        = "llama-3.1-8b-instant"

QUERY_PREFIX     = "Represent this sentence for searching relevant passages: "
EMBED_DIM        = 1024

CHUNK_SIZE       = 400      # approximate tokens
CHUNK_OVERLAP    = 50
RETRIEVAL_TOP_K  = 12       # FAISS candidates before reranking
RERANK_TOP_K     = 4        # chunks sent to LLM
CONTEXT_CHAR_LIM = 800      # chars per chunk inside prompt
HISTORY_WINDOW   = 6        # Q&A turns kept for multi-turn context
MAX_CHAT_SESSIONS = 20      # max saved chat sessions

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================
st.set_page_config(
    page_title="Lecture Q&A System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #f0f6fc !important; font-size: 0.72rem; letter-spacing: .07em;
    text-transform: uppercase; font-weight: 600;
}

/* ── Main container ── */
.main .block-container { padding-top: 2rem; max-width: 900px; }

/* ── Page title ── */
h1 {
    font-family: 'DM Mono', monospace !important;
    font-size: 1.55rem !important; font-weight: 500 !important;
    color: #0d1117 !important; letter-spacing: -.02em;
    border-bottom: 2.5px solid #0d1117; padding-bottom: .45rem;
    margin-bottom: 1.4rem !important;
}

/* ── Chat bubbles ── */
.chat-q {
    background: #0d1117; color: #e6edf3;
    border-radius: 4px 4px 4px 0; padding: .75rem 1.1rem;
    margin: 1.2rem 0 .3rem; font-size: .92rem; font-weight: 500;
    border-left: 3px solid #388bfd;
}
.chat-a {
    background: #f6f8fa; color: #0d1117;
    border-radius: 0 4px 4px 4px; padding: .8rem 1.1rem;
    font-size: .90rem; line-height: 1.7; border-left: 3px solid #d0d7de;
}

/* ── Source badge ── */
.badge {
    display: inline-block; background: #eef1f8; color: #3a4060;
    border: 1px solid #d0d5e8; border-radius: 3px;
    padding: 2px 8px; font-size: .72rem;
    font-family: 'DM Mono', monospace; margin: 2px 3px 2px 0;
}

/* ── Status chips ── */
.chip-ok  { display:inline-block; background:#e6f4ea; color:#1e6e34; border:1px solid #b7dfbf; border-radius:3px; padding:3px 10px; font-size:.75rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase; }
.chip-off { display:inline-block; background:#fff8e6; color:#7a5800; border:1px solid #f0d890; border-radius:3px; padding:3px 10px; font-size:.75rem; font-weight:600; letter-spacing:.05em; text-transform:uppercase; }

/* ── Metric row ── */
.mrow { display:flex; gap:.9rem; margin:.4rem 0; }
.mbox { flex:1; background:#f6f8fa; border:1px solid #d0d7de; border-radius:4px; padding:.45rem .8rem; text-align:center; }
.mval  { font-size:1.25rem; font-weight:600; color:#0d1117; }
.mlbl  { font-size:.65rem; color:#888; text-transform:uppercase; letter-spacing:.06em; }

/* ── Inputs ── */
.stTextInput>div>div>input {
    font-family:'DM Sans',sans-serif; border-radius:4px;
    border:1.5px solid #d0d7de; font-size:.91rem;
}
.stTextInput>div>div>input:focus { border-color:#388bfd; box-shadow:0 0 0 2px rgba(56,139,253,.15); }

/* ── Buttons ── */
.stButton>button {
    border-radius:4px; font-family:'DM Mono',monospace; font-size:.80rem;
    letter-spacing:.04em; border:1.5px solid #0d1117;
    background:#0d1117; color:#e6edf3; transition:background .15s;
}
.stButton>button:hover { background:#21262d; }

/* ── Note card ── */
.note-card {
    background:#fffdf5; border:1px solid #e8e0c0; border-radius:6px;
    padding:.8rem 1rem; margin:.5rem 0;
}
.note-title { font-weight:600; font-size:.93rem; margin-bottom:.25rem; color:#2d2a1e; }
.note-meta  { font-size:.72rem; color:#888; font-family:'DM Mono',monospace; }

#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALISATION
# ============================================================================
_DEFAULTS = {
    "chunks":            [],
    "faiss_index":       None,
    "chat_history":      [],
    "system_ready":      False,
    "num_pdfs":          0,
    "notes":             [],
    "quiz_history":      [],      # list of past quiz sessions
    "generated_notes":   "",
    "active_quiz":       None,    # parsed questions list for current interactive quiz
    "quiz_answers":      {},      # {q_index: chosen_option_index}
    "quiz_submitted":    False,   # True after student clicks Submit
    # ── Chat sessions ──────────────────────────────────────────────────────
    "chat_sessions":     [],      # [{id, title, created_at, messages}]
    "active_session_id": None,    # id of currently open saved session
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# ============================================================================
# CHUNKING
# ============================================================================

def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    sentences  = re.split(r'(?<=[.?!])\s+|\n{2,}', text)
    sentences  = [s.strip() for s in sentences if s.strip()]
    chunks, cur, cur_tok = [], [], 0

    for sent in sentences:
        s_tok = _approx_tokens(sent)

        if s_tok > size:                        # hard-split long sentence
            for word in sent.split():
                cur.append(word); cur_tok += 1
                if cur_tok >= size:
                    chunks.append(" ".join(cur))
                    cur = cur[-overlap:] if overlap else []
                    cur_tok = len(cur)
            continue

        if cur and cur_tok + s_tok > size:
            chunks.append(" ".join(cur))
            rollback, rb_tok = [], 0
            for s in reversed(cur):
                t = _approx_tokens(s)
                if rb_tok + t <= overlap:
                    rollback.insert(0, s); rb_tok += t
                else:
                    break
            cur, cur_tok = rollback, rb_tok

        cur.append(sent); cur_tok += s_tok

    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ============================================================================
# PDF EXTRACTION
# ============================================================================

def extract_chunks_from_pdf(pdf_file) -> List[Dict]:
    chunks = []
    suffix = os.path.splitext(pdf_file.name)[-1] or ".pdf"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    try:
        doc = fitz.open(tmp_path)
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text().strip()
            if not page_text:
                continue
            for idx, chunk in enumerate(chunk_text(page_text)):
                if chunk.strip():
                    chunks.append({
                        "text":        chunk,
                        "lecture":     pdf_file.name.replace(".pdf", ""),
                        "file":        pdf_file.name,
                        "page":        page_num,
                        "chunk_index": idx,
                        "id":          f"{pdf_file.name}_p{page_num}_c{idx}",
                    })
        doc.close()
    finally:
        try: os.remove(tmp_path)
        except OSError: pass

    return chunks

# ============================================================================
# MODEL LOADING  (cached across Streamlit sessions)
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner=False)
def load_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)

# ============================================================================
# EMBEDDING
# ============================================================================

def embed_docs(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    return model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, batch_size=64).astype(np.float32)


def embed_query(model: SentenceTransformer, query: str) -> np.ndarray:
    return model.encode(QUERY_PREFIX + query, normalize_embeddings=True,
                        show_progress_bar=False).astype(np.float32).reshape(1, -1)

# ============================================================================
# FAISS
# ============================================================================

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    idx = faiss.IndexFlatIP(EMBED_DIM)
    idx.add(embeddings)
    return idx

# ============================================================================
# RETRIEVAL + RERANKING
# ============================================================================

def retrieve(
    query: str,
    chunks: List[Dict],
    index: faiss.IndexFlatIP,
    embed_model: SentenceTransformer,
    reranker: CrossEncoder,
    retrieval_k: int = RETRIEVAL_TOP_K,
    rerank_k: int    = RERANK_TOP_K,
) -> List[Dict]:
    q_emb = embed_query(embed_model, query)
    scores, indices = index.search(q_emb, retrieval_k)

    candidates = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0: continue
        c = chunks[idx]
        candidates.append({
            "text":         c["text"],
            "metadata":     {k: c[k] for k in ("file","page","chunk_index","lecture")},
            "dense_score":  float(score),
        })

    if not candidates:
        return []

    rerank_scores = reranker.predict([(query, c["text"]) for c in candidates])
    for c, rs in zip(candidates, rerank_scores):
        c["rerank_score"] = float(rs)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:rerank_k]

# ============================================================================
# PROMPT BUILDER
# ============================================================================

def build_rag_prompt(query: str, docs: List[Dict], history: List[Dict]) -> str:
    # History block — last HISTORY_WINDOW turns, trimmed to avoid context bloat
    hist_block = ""
    if history:
        lines = []
        for turn in history[-HISTORY_WINDOW:]:
            q = turn['question']
            # strip the "Sources: …" trailer from stored answers
            a = turn['answer'].split("\n\nSources:")[0].strip()
            lines += [f"Student: {q}", f"Assistant: {a}"]
        hist_block = (
            "CONVERSATION HISTORY (use this to understand follow-up questions "
            "and resolve pronouns / references like 'it', 'that', 'the above'):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    # Context block
    ctx = "\n\n".join(
        f"[{d['metadata']['file']} | Page {d['metadata']['page']} | Chunk {d['metadata']['chunk_index']}]\n"
        f"{d['text'][:CONTEXT_CHAR_LIM]}"
        for i, d in enumerate(docs, 1)
    )

    return (
        "You are a helpful teaching assistant. "
        "Answer the student's question using ONLY the lecture content below. "
        "If a question refers to a previous answer or uses vague pronouns, "
        "use the conversation history to resolve what is being asked. "
        "If the answer is not in the context, say: 'This topic is not covered in the uploaded lectures.' "
        "Do not use outside knowledge.\n\n"
        f"{hist_block}"
        f"LECTURE CONTEXT:\n{ctx}\n\n"
        f"STUDENT QUESTION: {query}\n\nANSWER:"
    )


def build_notes_prompt(chunks: List[Dict], scope: str) -> str:
    ctx = "\n\n".join(
        f"[{c['file']} | Page {c['page']}]\n{c['text'][:800]}"
        for c in chunks
    )
    return (
        f"Create important study notes for: {scope}.\n"
        "Use ONLY the lecture content below. Focus on: definitions, key concepts, formulas, comparisons, exam-worthy points.\n\n"
        f"LECTURE CONTENT:\n{ctx}\n\n"
        "Return the notes as:\n- Short title\n- Important bullet points\n- Mini summary"
    )


def build_quiz_prompt(chunks: List[Dict], scope: str, n: int, difficulty: str) -> str:
    # Number each chunk so the LLM can reference it as a source
    numbered = []
    for idx, c in enumerate(chunks):
        numbered.append(f"[CHUNK {idx} | {c['file']} | Page {c['page']}]\n{c['text'][:800]}")
    ctx = "\n\n".join(numbered)

    return (
        f"Create a {difficulty.lower()} multiple-choice quiz about: {scope}.\n"
        f"Use ONLY the lecture content below. Generate exactly {n} questions.\n\n"
        "Return ONLY a valid JSON array — no markdown fences, no extra text.\n"
        "Each element must have exactly these keys:\n"
        '  "question": string\n'
        '  "options": array of exactly 4 strings (the answer choices, no A/B/C/D prefix)\n'
        '  "answer_index": integer 0-3 (index of the correct option in the options array)\n'
        '  "explanation": string — 2-3 sentences explaining WHY the correct answer is right '
        'and why the other options are wrong or less accurate\n'
        '  "source_chunk": integer — the CHUNK index (from the headers above) '
        'where this question\'s answer comes from\n\n'
        f"LECTURE CONTENT:\n{ctx}\n\n"
        "JSON ARRAY:"
    )

# ============================================================================
# LLM CALL  (Groq)
# ============================================================================

def call_llm(api_key: str, prompt: str, max_tokens: int = 500, temperature: float = 0.3) -> str:
    if not api_key:
        return "⚠️ No Groq API key provided."
    try:
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"LLM error: {exc}"

# ============================================================================
# FULL RAG PIPELINE
# ============================================================================

def run_rag(question: str, api_key: str) -> Dict:
    embed_model = load_embedding_model()
    reranker    = load_reranker()
    top_docs    = retrieve(
        question,
        st.session_state.chunks,
        st.session_state.faiss_index,
        embed_model, reranker,
    )

    if not top_docs:
        return {"answer": "No relevant content found in the uploaded lectures.", "sources": [], "docs": []}

    prompt  = build_rag_prompt(question, top_docs, st.session_state.chat_history)
    answer  = call_llm(api_key, prompt)
    sources = [
        f"{d['metadata']['file']} (Page {d['metadata']['page']}, Chunk {d['metadata']['chunk_index']})"
        for d in top_docs
    ]
    return {
        "answer":  f"{answer}\n\nSources: {' | '.join(sources)}",
        "sources": [d["metadata"] for d in top_docs],
        "docs":    top_docs,
    }

# ============================================================================
# HELPER — pick chunks for notes / quiz
# ============================================================================

def sample_chunks(scope: str, n: int) -> List[Dict]:
    pool = (
        [c for c in st.session_state.chunks if c["lecture"] == scope]
        if scope != "All lectures"
        else list(st.session_state.chunks)
    )
    pool.sort(key=lambda c: (c["lecture"], c["page"], c["chunk_index"]))
    if n <= 0 or len(pool) <= n:
        return pool
    step = max(1, len(pool) // n)
    return pool[::step][:n]


def lecture_options() -> List[str]:
    return ["All lectures"] + sorted({c["lecture"] for c in st.session_state.chunks})


def parse_quiz_json(raw: str, chunks_used: List[Dict] = None) -> List[Dict]:
    """Extract and parse the JSON quiz array from the LLM response."""
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    questions = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q        = str(item.get("question", "")).strip()
        opts     = item.get("options", [])
        ans_idx  = item.get("answer_index", 0)
        expl     = str(item.get("explanation", "")).strip()
        src_idx  = item.get("source_chunk")

        if q and isinstance(opts, list) and len(opts) == 4 and isinstance(ans_idx, int) and 0 <= ans_idx <= 3:
            # Resolve source metadata from the chunk list
            source_meta = None
            if chunks_used and isinstance(src_idx, int) and 0 <= src_idx < len(chunks_used):
                c = chunks_used[src_idx]
                source_meta = {
                    "file":    c["file"],
                    "page":    c["page"],
                    "lecture": c["lecture"],
                    "text":    c["text"][:300],
                }

            questions.append({
                "question":    q,
                "options":     [str(o).strip() for o in opts],
                "answer_index": ans_idx,
                "explanation": expl,
                "source_meta": source_meta,
            })
    return questions


def add_note(title: str, body: str, source: str = "Manual") -> None:
    st.session_state.notes.insert(0, {
        "id":         datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "title":      title.strip() or "Untitled note",
        "body":       body.strip(),
        "source":     source,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    })


# ── Chat session helpers ──────────────────────────────────────────────────────

def save_current_session(title: str = "") -> str:
    """Save chat_history as a new named session. Returns session id."""
    if not st.session_state.chat_history:
        return ""
    sid = datetime.now().strftime("%Y%m%d%H%M%S%f")
    auto_title = title.strip() or (
        st.session_state.chat_history[0]["question"][:50] + "…"
        if len(st.session_state.chat_history[0]["question"]) > 50
        else st.session_state.chat_history[0]["question"]
    )
    st.session_state.chat_sessions.insert(0, {
        "id":         sid,
        "title":      auto_title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages":   list(st.session_state.chat_history),
    })
    # keep only MAX_CHAT_SESSIONS
    st.session_state.chat_sessions = st.session_state.chat_sessions[:MAX_CHAT_SESSIONS]
    st.session_state.active_session_id = sid
    return sid


def load_session(sid: str) -> None:
    """Load a saved session into chat_history."""
    for sess in st.session_state.chat_sessions:
        if sess["id"] == sid:
            st.session_state.chat_history    = list(sess["messages"])
            st.session_state.active_session_id = sid
            return


def delete_session(sid: str) -> None:
    st.session_state.chat_sessions = [
        s for s in st.session_state.chat_sessions if s["id"] != sid
    ]
    if st.session_state.active_session_id == sid:
        st.session_state.active_session_id = None


def start_new_session() -> None:
    """Clear current chat and start fresh (does NOT auto-save)."""
    st.session_state.chat_history      = []
    st.session_state.active_session_id = None

# ============================================================================
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ============================================================================

with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    st.markdown("#### Groq API Key")
    api_key = st.text_input("Groq key", type="password",
                            help="Get a free key at console.groq.com",
                            label_visibility="collapsed")
    st.markdown(
        '<span class="chip-ok">Key Set</span>'  if api_key else
        '<span class="chip-off">No Key</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Upload Lectures")
    uploaded_files = st.file_uploader("PDF files", type=["pdf"],
                                      accept_multiple_files=True,
                                      label_visibility="collapsed")

    if st.button("▶ Initialize System", disabled=not api_key):
        if not uploaded_files:
            st.error("Upload at least one PDF first.")
        else:
            with st.spinner("Building FAISS index…"):
                try:
                    embed_model = load_embedding_model()
                    all_chunks  = []
                    bar         = st.progress(0)
                    for i, f in enumerate(uploaded_files):
                        all_chunks.extend(extract_chunks_from_pdf(f))
                        bar.progress((i + 1) / len(uploaded_files))

                    embeddings = embed_docs(embed_model, [c["text"] for c in all_chunks])
                    st.session_state.update({
                        "chunks":       all_chunks,
                        "faiss_index":  build_index(embeddings),
                        "system_ready": True,
                        "num_pdfs":     len(uploaded_files),
                        "chat_history": [],
                    })
                    st.success(f"✓ {len(all_chunks)} chunks indexed from {len(uploaded_files)} file(s).")
                except Exception as exc:
                    st.error(f"Init failed: {exc}")

    st.markdown("---")
    st.markdown("#### System Status")
    if st.session_state.system_ready:
        st.markdown('<span class="chip-ok">Ready</span>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="mrow">
            <div class="mbox"><div class="mval">{len(st.session_state.chunks)}</div><div class="mlbl">Chunks</div></div>
            <div class="mbox"><div class="mval">{st.session_state.num_pdfs}</div><div class="mlbl">Files</div></div>
            <div class="mbox"><div class="mval">{len(st.session_state.chat_history)}</div><div class="mlbl">Q&amp;A</div></div>
            <div class="mbox"><div class="mval">{len(st.session_state.notes)}</div><div class="mlbl">Notes</div></div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<span class="chip-off">Not Initialized</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""#### Model Stack
<small>
Embedding : <code>bge-large-en-v1.5</code><br>
Reranker  : <code>ms-marco-MiniLM-L-6-v2</code><br>
LLM       : <code>llama-3.1-8b-instant (Groq)</code><br>
Vector DB : <code>FAISS IndexFlatIP</code>
</small>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🗑 Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    # ── Chat Sessions ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💾 Chat Sessions")

    col_save, col_new = st.columns(2)
    with col_save:
        if st.button("Save Chat", disabled=not st.session_state.chat_history,
                     use_container_width=True):
            sid = save_current_session()
            if sid:
                st.success("Saved!")
                st.rerun()
            else:
                st.warning("Nothing to save.")
    with col_new:
        if st.button("New Chat", use_container_width=True):
            start_new_session()
            st.rerun()

    if st.session_state.chat_sessions:
        active_sid = st.session_state.active_session_id
        for sess in st.session_state.chat_sessions:
            is_active = sess["id"] == active_sid
            label     = ("▶ " if is_active else "") + sess["title"]
            sc1, sc2 = st.columns([4, 1])
            with sc1:
                if st.button(label, key=f"sess_{sess['id']}", use_container_width=True,
                             help=f"{sess['created_at']} · {len(sess['messages'])} Q&As"):
                    load_session(sess["id"])
                    st.rerun()
            with sc2:
                if st.button("✕", key=f"del_sess_{sess['id']}"):
                    delete_session(sess["id"])
                    st.rerun()
    else:
        st.caption("No saved sessions yet.")

# ============================================================================
# ── MAIN AREA ─────────────────────────────────────────────────────────────────
# ============================================================================

st.markdown("# 📚 Lecture Q&A System")

if not st.session_state.system_ready:
    st.markdown("""
**Getting started**

1. Paste your **Groq API key** in the sidebar — free at [console.groq.com](https://console.groq.com).
2. Upload one or more **lecture PDF** files.
3. Click **Initialize System** — the FAISS index is built once and cached.
4. Use the tabs below to **Ask questions**, write **Important Notes**, or generate a **Quiz**.

---

| Component | Details |
|---|---|
| Embedding | bge-large-en-v1.5 (1024-dim) |
| Chunking | Sentence-aware, 400-token + 50 overlap |
| Vector DB | FAISS IndexFlatIP (exact cosine) |
| Reranker | Cross-encoder ms-marco-MiniLM-L-6-v2 |
| LLM | Llama-3.1-8b-instant via Groq |
| Multi-turn | Last 3 turns injected into prompt |
""")

else:
    qa_tab, notes_tab, quiz_tab = st.tabs(["💬 Q&A", "📝 Important Notes", "🎯 Quiz"])

    # ─────────────────────────────────────────────────────────── Q&A Tab ──────
    with qa_tab:
        # Session indicator
        if st.session_state.active_session_id:
            for s in st.session_state.chat_sessions:
                if s["id"] == st.session_state.active_session_id:
                    st.markdown(
                        f"<div style='background:#eef5ff;border:1px solid #c8daff;border-radius:4px;"
                        f"padding:.3rem .8rem;font-size:.78rem;color:#1e3a6e;margin-bottom:.6rem;'>"
                        f"📂 Session: <b>{s['title']}</b> · {s['created_at']}</div>",
                        unsafe_allow_html=True,
                    )
                    break

        # Display chat history
        for i, turn in enumerate(st.session_state.chat_history):
            st.markdown(f'<div class="chat-q">Q{i+1}: {turn["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-a">{turn["answer"]}</div>', unsafe_allow_html=True)

            if turn.get("docs"):
                with st.expander(f"📎 Retrieved context ({len(turn['docs'])} chunks)"):
                    for j, doc in enumerate(turn["docs"], 1):
                        m = doc["metadata"]
                        st.markdown(
                            f'<span class="badge">{m["file"]}</span>'
                            f'<span class="badge">Page {m["page"]}</span>'
                            f'<span class="badge">Chunk {m["chunk_index"]}</span>'
                            f'<span class="badge">dense {doc["dense_score"]:.3f}</span>'
                            f'<span class="badge">rerank {doc["rerank_score"]:.2f}</span>',
                            unsafe_allow_html=True)
                        st.caption(doc["text"][:400] + ("…" if len(doc["text"]) > 400 else ""))
                        if j < len(turn["docs"]): st.markdown("---")

        # Input row
        st.markdown("---")
        col_q, col_btn = st.columns([5, 1])
        with col_q:
            question = st.text_input("Question", placeholder="Ask anything about your lectures…",
                                     label_visibility="collapsed", key="q_input")
        with col_btn:
            ask = st.button("Ask", use_container_width=True)

        if ask:
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Retrieving and generating…"):
                    try:
                        result = run_rag(question.strip(), api_key)
                        st.session_state.chat_history.append({
                            "question": question.strip(),
                            "answer":   result["answer"],
                            "sources":  result["sources"],
                            "docs":     result["docs"],
                        })
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pipeline error: {exc}")

        with st.expander("💡 Example questions"):
            st.markdown("""
- What does page 5 explain?
- How does backpropagation work?
- What is the difference between overfitting and underfitting?
- Summarise the key concepts from lecture 2.
""")

    # ──────────────────────────────────────────────────── Important Notes Tab ──
    with notes_tab:
        st.subheader("📝 Important Notes")

        # Manual note
        with st.form("manual_note"):
            t = st.text_input("Note title")
            b = st.text_area("Note body", height=110)
            if st.form_submit_button("Add note"):
                if b.strip():
                    add_note(t, b, source="Manual")
                    st.success("Note saved.")
                else:
                    st.warning("Write the note body first.")

        st.markdown("---")
        st.markdown("**Generate notes from lectures**")
        col1, col2 = st.columns(2)
        with col1:
            note_scope  = st.selectbox("Lecture scope", lecture_options(), key="ns")
        with col2:
            note_chunks = st.slider("Chunks to use", 4, 30, 12, 2, key="nc")

        if st.button("⚡ Generate Notes"):
            selected = sample_chunks(note_scope, note_chunks)
            if not api_key:
                st.warning("Add your Groq API key first.")
            elif not selected:
                st.warning("No chunks available for this scope.")
            else:
                with st.spinner("Generating…"):
                    text = call_llm(api_key, build_notes_prompt(selected, note_scope),
                                    max_tokens=900, temperature=0.3)
                    st.session_state.generated_notes = text
                    add_note(f"Generated — {note_scope}", text, source=f"Auto from {note_scope}")
                    st.success("Notes generated and saved.")

        if st.session_state.generated_notes:
            with st.expander("Latest generated notes", expanded=True):
                st.markdown(st.session_state.generated_notes)

        st.markdown("---")
        # Download button
        md_content = "# Important Notes\n\n" + "\n\n---\n\n".join(
            f"## {n['title']}\n_{n['source']} · {n['created_at']}_\n\n{n['body']}"
            for n in st.session_state.notes
        ) if st.session_state.notes else "# Important Notes\n\nNo notes yet."

        st.download_button("⬇️ Download as Markdown", data=md_content,
                           file_name="important_notes.md", mime="text/markdown",
                           disabled=not st.session_state.notes)

        st.markdown("---")
        if not st.session_state.notes:
            st.info("No notes saved yet.")
        else:
            for note in list(st.session_state.notes):
                with st.expander(f"{note['title']}  ·  {note['created_at']}"):
                    st.caption(f"Source: {note['source']}")
                    st.markdown(note["body"])
                    if st.button("🗑 Delete", key=f"del_{note['id']}"):
                        st.session_state.notes = [n for n in st.session_state.notes if n["id"] != note["id"]]
                        st.rerun()

    # ──────────────────────────────────────────────────────────── Quiz Tab ──
    with quiz_tab:
        st.subheader("🎯 Interactive Quiz")

        # ── Generation controls ───────────────────────────────────────────
        with st.container():
            colA, colB, colC, colD = st.columns(4)
            with colA: quiz_scope  = st.selectbox("Scope",      lecture_options(), key="qs")
            with colB: quiz_n      = st.slider("Questions", 3, 12, 5, key="qn")
            with colC: quiz_diff   = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="qd")
            with colD: quiz_chunks = st.slider("Chunks",    4, 30, 12, 2, key="qc")

        gen_col, reset_col = st.columns([3, 1])
        with gen_col:
            gen_btn = st.button("⚡ Generate New Quiz", use_container_width=True)
        with reset_col:
            if st.button("🔄 Reset", use_container_width=True, disabled=not st.session_state.active_quiz):
                st.session_state.quiz_answers   = {}
                st.session_state.quiz_submitted = False
                st.rerun()

        if gen_btn:
            selected = sample_chunks(quiz_scope, quiz_chunks)
            if not api_key:
                st.warning("Add your Groq API key first.")
            elif not selected:
                st.warning("No chunks available for this scope.")
            else:
                with st.spinner("Generating quiz…"):
                    raw = call_llm(
                        api_key,
                        build_quiz_prompt(selected, quiz_scope, quiz_n, quiz_diff),
                        max_tokens=2000, temperature=0.5,
                    )
                    questions = parse_quiz_json(raw, chunks_used=selected)
                    if not questions:
                        st.error("Could not parse quiz. Try again — the LLM occasionally returns malformed JSON.")
                    else:
                        st.session_state.active_quiz    = questions
                        st.session_state.quiz_answers   = {}
                        st.session_state.quiz_submitted = False
                        # Archive in history
                        st.session_state.quiz_history.insert(0, {
                            "id":         datetime.now().strftime("%Y%m%d%H%M%S%f"),
                            "scope":      quiz_scope,
                            "difficulty": quiz_diff,
                            "n":          len(questions),
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                            "questions":  questions,
                        })
                        st.rerun()

        # ── Active quiz ───────────────────────────────────────────────────
        if st.session_state.active_quiz:
            questions  = st.session_state.active_quiz
            submitted  = st.session_state.quiz_submitted
            answers    = st.session_state.quiz_answers
            n_q        = len(questions)

            st.markdown("---")

            # Score banner (shown after submit)
            if submitted:
                score = sum(
                    1 for i, q in enumerate(questions)
                    if answers.get(i) == q["answer_index"]
                )
                pct = int(score / n_q * 100)
                if pct >= 80:
                    banner_color, emoji = "#e6f4ea", "🏆"
                elif pct >= 50:
                    banner_color, emoji = "#fff8e6", "📚"
                else:
                    banner_color, emoji = "#fdecea", "💪"

                st.markdown(
                    f"<div style='background:{banner_color};border-radius:8px;"
                    f"padding:1rem 1.4rem;margin-bottom:1.2rem;text-align:center;'>"
                    f"<span style='font-size:1.9rem;'>{emoji}</span><br>"
                    f"<span style='font-size:1.4rem;font-weight:700;color:#0d1117;'>"
                    f"{score} / {n_q} &nbsp;({pct}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Render each question
            for i, q in enumerate(questions):
                chosen    = answers.get(i)          # None = not answered yet
                correct_i = q["answer_index"]
                labels    = ["A", "B", "C", "D"]

                # Question header
                st.markdown(
                    f"<div style='font-weight:600;font-size:.96rem;color:#f0f6fc;"
                    f"background:#1c2333;border-radius:6px;padding:.55rem .9rem;"
                    f"margin-top:1rem;margin-bottom:.35rem;border-left:3px solid #388bfd;'>"
                    f"Q{i+1}. {q['question']}</div>",
                    unsafe_allow_html=True,
                )

                if not submitted:
                    # ── Before submit: radio buttons ──────────────────────
                    radio_options = [f"{labels[j]}. {opt}" for j, opt in enumerate(q["options"])]
                    default_idx   = chosen if chosen is not None else 0
                    picked = st.radio(
                        label=f"q_{i}",
                        options=list(range(4)),
                        format_func=lambda j, opts=radio_options: opts[j],
                        index=default_idx,
                        key=f"radio_{i}",
                        label_visibility="collapsed",
                    )
                    st.session_state.quiz_answers[i] = picked

                else:
                    # ── After submit: coloured result cards ───────────────
                    for j, opt in enumerate(q["options"]):
                        if j == correct_i and j == chosen:
                            bg, icon, txt_color = "#e6f4ea", "✅", "#1e6e34"
                        elif j == correct_i:
                            bg, icon, txt_color = "#e6f4ea", "✅", "#1e6e34"
                        elif j == chosen:
                            bg, icon, txt_color = "#fdecea", "❌", "#9b2226"
                        else:
                            bg, icon, txt_color = "#f6f8fa", "  ", "#4a5568"

                        st.markdown(
                            f"<div style='background:{bg};border-radius:5px;padding:.45rem .9rem;"
                            f"margin:.2rem 0;font-size:.89rem;color:{txt_color};border:1px solid #e0e0e0;'>"
                            f"{icon} <b>{labels[j]}.</b> {opt}</div>",
                            unsafe_allow_html=True,
                        )

                    # Explanation + source
                    if q.get("explanation"):
                        st.markdown(
                            f"<div style='background:#f0f4ff;border-left:3px solid #388bfd;"
                            f"border-radius:0 4px 4px 0;padding:.5rem .9rem;"
                            f"font-size:.86rem;color:#1e3a6e;margin:.4rem 0 .4rem;'>"
                            f"<b>💡 Why this answer?</b><br>{q['explanation']}</div>",
                            unsafe_allow_html=True,
                        )

                    # Source card
                    sm = q.get("source_meta")
                    if sm:
                        st.markdown(
                            f"<div style='background:#f6f8fa;border:1px solid #d0d7de;"
                            f"border-radius:5px;padding:.45rem .9rem;margin:.2rem 0 .8rem;"
                            f"font-size:.80rem;color:#555;'>"
                            f"<b>📄 Source:</b> "
                            f"<span style='font-family:DM Mono,monospace;'>{sm['file']}</span> "
                            f"— Page {sm['page']}<br>"
                            f"<span style='color:#888;font-style:italic;'>\"{sm['text']}…\"</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            st.markdown("---")

            if not submitted:
                # Check if all answered
                answered = sum(1 for i in range(n_q) if i in answers)
                remaining = n_q - answered
                if remaining > 0:
                    st.caption(f"{remaining} question(s) not answered yet — you can still submit.")
                if st.button("📩 Submit Quiz", use_container_width=True, type="primary"):
                    # Fill any unanswered with current radio value (already stored above)
                    st.session_state.quiz_submitted = True
                    st.rerun()
            else:
                if st.button("🔄 Try Again", use_container_width=True):
                    st.session_state.quiz_answers   = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()

        else:
            st.info("Generate a quiz to get started.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#aaa;font-size:.75rem;"
    "font-family:DM Mono,monospace;'>"
    "bge-large-en-v1.5 · FAISS · CrossEncoder · llama-3.1-8b-instant · Streamlit"
    "</div>", unsafe_allow_html=True)