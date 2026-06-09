import base64
import gc
from pathlib import Path

import streamlit as st

from chains.rag_chain import get_rag_chain
from config import DATA_DIR, GROQ_API_KEY
from embeddings.embedding_model import clear_embedding_cache
from ingestion.ingest_data import ingest, ingest_uploads

LOGO_PATH = Path(__file__).parent / "assets" / "logo.svg"
APP_NAME = "RAG Bot"
APP_EMOJI = "🤖"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --blue-primary: #1e40af;
        --blue-light: #dbeafe;
        --blue-bg: #f0f6ff;
        --text-black: #111827;
        --text-muted: #374151;
        --white: #ffffff;
    }

    /* Base: white background, black text */
    .stApp, [data-testid="stAppViewContainer"], .main, .block-container {
        background-color: var(--white) !important;
        color: var(--text-black) !important;
    }

    p, span, label, li, div, h1, h2, h3, h4, h5, h6 {
        color: var(--text-black);
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: var(--text-black) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--blue-bg) !important;
        border-right: 2px solid var(--blue-light);
    }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption {
        color: var(--text-black) !important;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: var(--blue-primary) !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: var(--white) !important;
        border: 1px solid var(--blue-light);
        border-radius: 10px;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li {
        color: var(--text-black) !important;
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text-black) !important;
        background-color: var(--white) !important;
        border: 1px solid var(--blue-light) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #6b7280 !important;
    }

    /* Metrics, captions, expanders */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: var(--text-black) !important;
    }
    .stCaption, small {
        color: var(--text-muted) !important;
    }
    [data-testid="stExpander"] summary p {
        color: var(--text-black) !important;
    }

    /* Buttons */
    .stButton > button {
        color: var(--text-black) !important;
        background-color: var(--white) !important;
        border: 1px solid var(--blue-light) !important;
    }
    .stButton > button[kind="primary"] {
        background-color: var(--blue-primary) !important;
        border: 1px solid var(--blue-primary) !important;
        color: var(--white) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #1e3a8a !important;
        color: var(--white) !important;
    }

    /* File uploader */
    div[data-testid="stFileUploader"] {
        border: 2px dashed var(--blue-primary);
        border-radius: 10px;
        background: var(--white);
    }
    div[data-testid="stFileUploader"] label,
    div[data-testid="stFileUploader"] span,
    div[data-testid="stFileUploader"] small {
        color: var(--text-black) !important;
    }

    /* Brand header */
    .brand-wrap {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.5rem;
        padding: 1rem;
        background: var(--white);
        border: 1px solid var(--blue-light);
        border-radius: 12px;
    }
    .brand-logo {
        width: 68px;
        height: 68px;
        border-radius: 12px;
        border: 2px solid var(--blue-light);
    }
    .brand-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--blue-primary) !important;
        margin: 0;
    }
    .brand-sub {
        color: var(--text-muted) !important;
        font-size: 1rem;
        margin-top: 0.25rem;
    }

    /* Status banners */
    .status-ready {
        background: var(--blue-bg);
        border: 1px solid var(--blue-primary);
        color: var(--text-black) !important;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-wait {
        background: var(--white);
        border: 1px dashed var(--blue-primary);
        color: var(--text-black) !important;
        padding: 0.85rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    /* Source cards */
    .source-card {
        background: var(--white);
        border: 1px solid var(--blue-light);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
        color: var(--text-black) !important;
    }
    .source-preview {
        color: var(--text-muted) !important;
    }

    .quick-label, .section-title {
        color: var(--blue-primary) !important;
        font-weight: 600;
    }

    /* Hide Streamlit header/footer for cleaner look */
    header[data-testid="stHeader"] {
        background-color: var(--white);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_chain():
    return get_rag_chain()


def logo_data_uri() -> str:
    svg = LOGO_PATH.read_text(encoding="utf-8")
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def init_session():
    defaults = {
        "messages": [],
        "docs_ready": False,
        "chunk_count": 0,
        "uploaded_names": [],
        "pending_prompt": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def save_uploads(uploaded_files):
    return [(f.name, f.getvalue()) for f in uploaded_files]


def reset_rag_caches():
    st.cache_resource.clear()
    clear_embedding_cache()
    gc.collect()


def ask_bot(prompt: str):
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in {"user", "assistant"}
    ]
    chain = load_chain()
    return chain.invoke({"input": prompt, "history": history})


def render_sidebar():
    with st.sidebar:
        st.markdown(f"### {APP_EMOJI} {APP_NAME}")
        st.caption("Upload • Analyze • Chat")

        st.markdown("#### 📁 Upload Documents")
        st.caption("PDF, DOCX, TXT, MD")

        uploaded_files = st.file_uploader(
            "Choose files",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if st.button(
            "🔍 Analyze Documents",
            use_container_width=True,
            type="primary",
            disabled=not uploaded_files,
        ):
            uploads = save_uploads(uploaded_files)
            with st.spinner("Reading and indexing your documents..."):
                try:
                    reset_rag_caches()
                    chunks = ingest_uploads(uploads)
                    st.session_state.docs_ready = chunks > 0
                    st.session_state.chunk_count = chunks
                    st.session_state.uploaded_names = [name for name, _ in uploads]
                    st.session_state.messages = []
                    if chunks > 0:
                        st.success(f"Analyzed {len(uploads)} file(s) → {chunks} chunks")
                    else:
                        st.error("No text could be extracted from the uploaded files.")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")

        st.divider()
        st.markdown("#### ⚙️ Settings")

        if GROQ_API_KEY:
            st.info("Powered by Groq API")
        else:
            st.warning("Using local model — add GROQ_API_KEY for better answers")

        if st.button("Re-analyze saved files", use_container_width=True):
            with st.spinner("Re-indexing..."):
                try:
                    reset_rag_caches()
                    chunks = ingest()
                    st.session_state.docs_ready = chunks > 0
                    st.session_state.chunk_count = chunks
                    st.session_state.messages = []
                    st.success(f"Re-indexed {chunks} chunks")
                except Exception as exc:
                    st.error(f"Re-index failed: {exc}")

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        if st.session_state.docs_ready:
            st.metric("Indexed chunks", st.session_state.chunk_count)
            if st.session_state.uploaded_names:
                st.caption("Active files:")
                for name in st.session_state.uploaded_names:
                    st.caption(f"• {name}")


def render_header():
    logo = logo_data_uri()
    st.markdown(
        f"""
        <div class="brand-wrap">
            <img src="{logo}" class="brand-logo" alt="{APP_NAME} logo"/>
            <div>
                <p class="brand-title">{APP_EMOJI} {APP_NAME}</p>
                <p class="brand-sub">Your smart document assistant — upload, analyze, and chat.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_banner():
    if st.session_state.docs_ready:
        names = ", ".join(st.session_state.uploaded_names) or "saved documents"
        st.markdown(
            f'<div class="status-ready">✅ <b>Ready to chat</b> — {st.session_state.chunk_count} chunks from: {names}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-wait">📤 Upload a document in the sidebar and click <b>Analyze Documents</b> to start.</div>',
            unsafe_allow_html=True,
        )


def render_quick_actions():
    if not st.session_state.docs_ready:
        return

    st.markdown('<p class="quick-label">Quick actions</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    prompts = {
        "summary": "Summarize this document in a clear overview with key points.",
        "skills": "What are the main skills, experience, and qualifications mentioned?",
        "topics": "What are the most important topics covered in this document?",
    }
    if c1.button("📋 Summary", use_container_width=True):
        st.session_state.pending_prompt = prompts["summary"]
        st.rerun()
    if c2.button("💼 Key details", use_container_width=True):
        st.session_state.pending_prompt = prompts["skills"]
        st.rerun()
    if c3.button("🧠 Main topics", use_container_width=True):
        st.session_state.pending_prompt = prompts["topics"]
        st.rerun()


def handle_user_message(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("RAG Bot is thinking..."):
            try:
                result = ask_bot(prompt)
                answer = result.get("answer", "Sorry, I could not generate an answer.")
            except Exception as exc:
                answer = f"Sorry, something went wrong: {exc}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


def render_chat():
    for message in st.session_state.messages:
        avatar = "👤" if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if not st.session_state.docs_ready:
        st.chat_input("Upload and analyze documents first...", disabled=True)
        return

    pending = st.session_state.pending_prompt
    if pending:
        st.session_state.pending_prompt = None
        handle_user_message(pending)
        return

    prompt = st.chat_input("Message RAG Bot...")
    if prompt:
        handle_user_message(prompt)


def main():
    init_session()
    render_sidebar()
    render_header()

    col1, col2 = st.columns([3, 1])
    with col2:
        existing = sum(1 for p in DATA_DIR.glob("*") if p.is_file())
        st.metric("Saved files", existing)

    render_status_banner()
    render_quick_actions()

    st.markdown('<p class="section-title" style="font-size:1.25rem;">💬 Conversation</p>', unsafe_allow_html=True)
    render_chat()


if __name__ == "__main__":
    main()
