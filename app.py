"""Streamlit UI for the LangGraph FAQ RAG Chatbot with real-time token streaming and session memory."""

import asyncio
import json
import queue
import threading
import time
import uuid
from typing import Any, Dict, Generator, List, Union

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from config.settings import settings
from src.feedback import get_feedback_analytics, record_user_feedback
from src.graph import rag_app
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Data classes for the streaming bridge
# ---------------------------------------------------------------------------

class _StatusUpdate:
    """Marker object for pipeline step progress (not a displayable token)."""

    __slots__ = ("label",)

    def __init__(self, label: str) -> None:
        self.label = label


_STREAM_DONE = object()

# Node name → user-facing status label
_NODE_STATUS_LABELS: Dict[str, str] = {
    "guardrail": "🛡️ Memeriksa keselamatan & skop soalan...",
    "retrieve": "🔍 Mencari pangkalan data FAQ Tonton...",
    "grade": "📊 Menilai ketepatan dokumen...",
    "generate": "✍️ Menjana jawapan rasmi...",
    "clarification": "💬 Menyediakan soalan berkaitan...",
    "fallback": "⚠️ Menyediakan panduan alternatif...",
}


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tonton FAQ Assistant",
    page_icon="📺",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS (Modern Dark Theme & Polished Micro-Interactions)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Branded Hero Card */
    .hero-container {
        background: linear-gradient(135deg, #161b26 0%, #0d111a 100%);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 14px;
        padding: 20px 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .hero-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin: 0;
        line-height: 1.5;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.76rem;
        font-weight: 600;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #34d399;
        border-radius: 50%;
        box-shadow: 0 0 8px #34d399;
    }

    /* Welcome Category Cards */
    .welcome-header {
        color: #f1f5f9;
        font-size: 1.02rem;
        font-weight: 600;
        margin-bottom: 12px;
        margin-top: 4px;
    }
    .topic-card {
        background: #161c28;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 12px 14px;
        margin-bottom: 8px;
        min-height: 85px;
    }
    .topic-title {
        font-weight: 600;
        font-size: 0.90rem;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .topic-desc {
        font-size: 0.78rem;
        color: #94a3b8;
        margin: 0;
        line-height: 1.4;
    }

    /* Source Box Styling */
    .source-box {
        background: #141923;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-left: 4px solid #00d4ff;
        border-radius: 0 8px 8px 0;
        padding: 12px 14px;
        margin-top: 8px;
        margin-bottom: 8px;
        font-size: 0.88rem;
        color: #cbd5e1;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
    }
    .source-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    .category-pill {
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .id-pill {
        background: rgba(255, 255, 255, 0.07);
        color: #94a3b8;
        border-radius: 4px;
        padding: 2px 6px;
        font-size: 0.72rem;
    }

    /* Live Pipeline Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.3);
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.84rem;
        color: #38bdf8;
        margin: 6px 0 10px 0;
        animation: pulse-glow 1.5s infinite ease-in-out;
    }
    @keyframes pulse-glow {
        0%, 100% { opacity: 0.85; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.01); }
    }

    /* Feedback & Response Meta */
    .response-meta {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 6px;
        margin-bottom: 4px;
    }
    .feedback-bar {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 6px;
        padding-top: 4px;
    }
    .feedback-label {
        font-size: 0.80rem;
        color: #94a3b8;
    }

    /* Button Polish */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.25);
        transform: translateY(-1px);
    }
    .stChatInput {
        padding-bottom: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Auto-indexing on Startup (Ensures seamless Cloud deployment)
# ---------------------------------------------------------------------------
def _ensure_faq_indexed() -> None:
    """Ensures Qdrant database contains FAQ items on initial application boot."""
    if not settings.effective_api_key:
        return
    try:
        from src.vectorstore.qdrant_client import QdrantManager
        qdrant = QdrantManager(
            storage_path=settings.qdrant_storage_path,
            collection_name=settings.qdrant_collection_name,
        )
        if qdrant.get_collection_count() == 0:
            logger.info("Empty Qdrant collection detected on startup. Auto-indexing FAQ...")
            from src.ingestion.indexer import run_indexing
            run_indexing(recreate_collection=False)
    except Exception as e:
        logger.warning(f"Knowledge base auto-indexing check: {e}")

_ensure_faq_indexed()


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sample_query" not in st.session_state:
    st.session_state.sample_query = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = {}  # {msg_index: "up" | "down"}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/4/4b/Tonton_logo.png",
        width=140,
    )
    st.markdown("### **Tonton Support Bot**")
    st.caption("Pembantu AI Sokongan Pelanggan Pintar berasaskan FAQ Rasmi.")

    # 1. Primary Actions
    if st.button("🔄 Perbualan Baharu", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.sample_query = ""
        st.session_state.feedback = {}
        st.rerun()

    # 2. Export / Download Conversation Transcript
    if st.session_state.messages:
        chat_export_lines = [f"# Transkrip Perbualan Tonton Support ({st.session_state.session_id[:8]})\n"]
        for idx, m in enumerate(st.session_state.messages, 1):
            role_name = "Pengguna" if isinstance(m["content"], HumanMessage) else "Tonton Assistant"
            chat_export_lines.append(f"**[{idx}] {role_name}:**\n{m['text']}\n")
        
        st.download_button(
            label="📥 Muat Turun Transkrip",
            data="\n".join(chat_export_lines),
            file_name=f"tonton_chat_{st.session_state.session_id[:8]}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()

    # 3. Quick Sample Questions (Aligned with FAQ.md)
    st.markdown("#### 💡 **Cadangan Soalan**")
    samples = [
        "Bagaimana cara melanggan atau menaik taraf akaun TontonUp?",
        "Kenapa saya tidak boleh menonton selepas membuat bayaran?",
        "Saya telah melanggan tetapi kenapa masih mempunyai iklan?",
        "Bagaimana saya nak membatalkan langganan TontonUp?",
        "Boleh saya tonton di luar negara?",
        "Apakah program TV Tuisyen yang disediakan di Tonton?",
    ]
    for sample in samples:
        if st.button(f"📌 {sample}", use_container_width=True, key=f"sidebar_sample_{hash(sample)}"):
            st.session_state.sample_query = sample

    st.divider()

    # 4. System & Developer Info (Collapsible)
    with st.expander("⚙️ **Maklumat Sistem & Pembelajaran AI**", expanded=False):
        analytics = get_feedback_analytics()
        st.caption(f"**Model LLM:** `{settings.gemini_model}`")
        st.caption(f"**Embeddings:** `{settings.embedding_model}`")
        st.caption(f"**Vector Store:** `Qdrant Local`")
        st.caption(f"**Kepuasan Pengguna:** `{analytics['satisfaction_rate']}% ({analytics['upvotes']}👍 / {analytics['downvotes']}👎)`")
        st.caption(f"**Frasa Dipelajari AI:** `{analytics['learned_variants']} varian dinamik`")
        st.caption(f"**Sesi Thread ID:** `{st.session_state.session_id}`")


# ---------------------------------------------------------------------------
# Main Chat Area Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">
            <span>📺 Tonton FAQ Assistant</span>
            <span class="status-badge"><span class="status-dot"></span> Sistem Aktif</span>
        </div>
        <p class="hero-subtitle">
            Dapatkan bantuan segera untuk langganan TontonUp, pembayaran, reset kata laluan, atau masalah teknikal streaming secara rasmi.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not settings.effective_api_key:
    st.error(
        "⚠️ **Kunci API Gemini Tidak Dikesan / Gemini API Key Missing**\n\n"
        "Sila pastikan anda telah menambah `GEMINI_API_KEY = \"AIzaSy...\"` di dalam **Streamlit Cloud &rarr; App Settings &rarr; Secrets** (format TOML) dan klik **Save**."
    )


# ---------------------------------------------------------------------------
# Welcome Message & Interactive Starter Cards (when conversation is empty)
# ---------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown('<div class="welcome-header">Pilih topik bantuan pantas di bawah:</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    categories = [
        (
            col1,
            "💳",
            "Langganan & TontonUp",
            "Cara melanggan, menaik taraf atau aktifkan pelan",
            "Bagaimana saya nak membatalkan langganan bulanan TontonUp saya?",
        ),
        (
            col2,
            "👤",
            "Akaun & Kata Laluan",
            "Lupa kata laluan atau pautan reset akaun",
            "Bagaimana saya nak menukar kata laluan?",
        ),
        (
            col3,
            "📺",
            "Isu Siaran & Bayaran",
            "Tidak boleh tonton selepas bayaran atau isu iklan",
            "Kenapa saya tidak boleh menonton selepas membuat bayaran?",
        ),
    ]
    
    for col, icon, title, desc, query_text in categories:
        with col:
            st.markdown(
                f"""
                <div class="topic-card">
                    <div class="topic-title">{icon} {title}</div>
                    <p class="topic-desc">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Tanya: {title}", use_container_width=True, key=f"welcome_btn_{title}"):
                st.session_state.sample_query = query_text
                st.rerun()


# ---------------------------------------------------------------------------
# Helper: Render source cards
# ---------------------------------------------------------------------------
def _render_sources(sources: List[Dict[str, Any]]) -> None:
    """Renders FAQ source attribution cards inside a styled expander."""
    if not sources:
        return
    with st.expander(f"📚 Rujukan FAQ Berkaitan ({len(sources)} Dokumen Dijumpai)", expanded=False):
        for src in sources:
            q_text = src.get("question", "")
            cat = src.get("category", "Umum")
            faq_id = src.get("id", "FAQ")
            q_html = f'<div style="font-weight:600; color:#f8fafc; margin-bottom:4px;">📋 {q_text}</div>' if q_text else ""
            
            st.markdown(
                f"""
                <div class="source-box">
                    <div class="source-header">
                        <span class="category-pill">🏷️ {cat}</span>
                        <span class="id-pill">ID: #{faq_id}</span>
                    </div>
                    {q_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Helper: Render feedback buttons
# ---------------------------------------------------------------------------
def _render_feedback(msg_idx: int) -> None:
    """Renders styled feedback reaction bar for assistant messages with toasts and dynamic self-learning."""
    existing_fb = st.session_state.feedback.get(msg_idx)

    fb_cols = st.columns([3, 1, 1, 6])
    with fb_cols[0]:
        st.caption("Adakah jawapan ini membantu?")
    with fb_cols[1]:
        up_label = "✅ Ya" if existing_fb == "up" else "👍 Ya"
        if st.button(up_label, key=f"fb_up_{msg_idx}", help="Jawapan ini tepat & membantu"):
            st.session_state.feedback[msg_idx] = "up"

            # Retrieve preceding user query and current assistant message
            user_query = ""
            if msg_idx > 0 and isinstance(st.session_state.messages[msg_idx - 1]["content"], HumanMessage):
                user_query = st.session_state.messages[msg_idx - 1]["text"]

            curr_msg = st.session_state.messages[msg_idx] if msg_idx < len(st.session_state.messages) else {}

            record = record_user_feedback(
                session_id=st.session_state.session_id,
                message_index=msg_idx,
                user_query=user_query,
                assistant_response=curr_msg.get("text", ""),
                rating="up",
                sources=curr_msg.get("sources", []),
                elapsed_seconds=curr_msg.get("elapsed_seconds"),
            )

            if record.get("learned_as_variant"):
                st.toast("Terima kasih! Frasa soalan anda telah dipelajari oleh AI untuk carian seterusnya! 🚀", icon="🧠")
            else:
                st.toast("Terima kasih atas maklum balas anda! 👍", icon="✨")
            st.rerun()

    with fb_cols[2]:
        down_label = "❌ Tidak" if existing_fb == "down" else "👎 Tidak"
        if st.button(down_label, key=f"fb_down_{msg_idx}", help="Jawapan kurang tepat"):
            st.session_state.feedback[msg_idx] = "down"

            user_query = ""
            if msg_idx > 0 and isinstance(st.session_state.messages[msg_idx - 1]["content"], HumanMessage):
                user_query = st.session_state.messages[msg_idx - 1]["text"]

            curr_msg = st.session_state.messages[msg_idx] if msg_idx < len(st.session_state.messages) else {}

            record_user_feedback(
                session_id=st.session_state.session_id,
                message_index=msg_idx,
                user_query=user_query,
                assistant_response=curr_msg.get("text", ""),
                rating="down",
                sources=curr_msg.get("sources", []),
                elapsed_seconds=curr_msg.get("elapsed_seconds"),
            )

            st.toast("Maklum balas direkodkan. Kami akan menyemak semula FAQ ini! 💡", icon="📝")
            st.rerun()


# ---------------------------------------------------------------------------
# Display Chat History
# ---------------------------------------------------------------------------
for msg_idx, msg in enumerate(st.session_state.messages):
    role = "user" if isinstance(msg["content"], HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg["text"])

        # Source attribution cards
        if msg.get("sources"):
            _render_sources(msg["sources"])

        # Clickable clarification suggestions (re-rendered from history)
        if msg.get("suggestions"):
            st.markdown('<div style="font-size:0.85rem; color:#94a3b8; margin-top:8px;">💡 Soalan berkaitan yang boleh anda klik:</div>', unsafe_allow_html=True)
            for suggestion in msg["suggestions"]:
                if st.button(
                    f"👉 {suggestion}",
                    key=f"hist_sug_{msg_idx}_{hash(suggestion)}",
                    use_container_width=True,
                ):
                    st.session_state.sample_query = suggestion
                    st.rerun()

        # Response metadata line (elapsed time)
        if msg.get("elapsed_seconds"):
            st.markdown(
                f'<div class="response-meta">⏱️ Dijana dalam {msg["elapsed_seconds"]:.1f}s</div>',
                unsafe_allow_html=True,
            )

        # Feedback buttons (assistant messages only)
        if role == "assistant":
            _render_feedback(msg_idx)


# ---------------------------------------------------------------------------
# Streaming Bridge: Async LangGraph → Sync Streamlit
# ---------------------------------------------------------------------------

def stream_rag_tokens(
    prompt_text: str,
    thread_id: str,
    meta_container: Dict[str, Any],
) -> Generator[Union[str, _StatusUpdate], None, None]:
    """Generates real-time token stream from LangGraph astream_events.

    Uses a background thread with its own event loop (via ``asyncio.run``) and a
    thread-safe queue to bridge async token production to Streamlit's synchronous
    consumer. Strictly filters `on_chat_model_stream` events to ONLY capture tokens
    originating from the final `generate` node to prevent leaking intermediate LLM outputs.

    Yields:
        Either a ``str`` token chunk or a ``_StatusUpdate`` for pipeline step progress.
    """
    initial_state = {"question": prompt_text}
    config = {"configurable": {"thread_id": thread_id}}

    token_queue: queue.Queue = queue.Queue()

    async def _consume_events() -> None:
        """Runs in a background thread's isolated event loop via asyncio.run()."""
        try:
            async for event in rag_app.astream_events(
                initial_state, config=config, version="v2"
            ):
                kind = event["event"]

                # --- Pipeline step progress ---
                if kind == "on_chain_start":
                    node_name = event.get("name", "")
                    if node_name in _NODE_STATUS_LABELS:
                        token_queue.put(_StatusUpdate(_NODE_STATUS_LABELS[node_name]))

                # --- Live LLM token chunks (STRICTLY filter to 'generate' node only) ---
                elif kind == "on_chat_model_stream":
                    current_node = event.get("metadata", {}).get("langgraph_node")
                    # Only stream tokens from the generation node, ignoring internal query expansions
                    if current_node == "generate":
                        chunk = event["data"].get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            text_content = chunk.content
                            if isinstance(text_content, str) and text_content:
                                meta_container["has_llm_stream"] = True
                                token_queue.put(text_content)

                # --- Terminal node output & metadata ---
                elif kind == "on_chain_end":
                    node_name = event.get("name")
                    if node_name in {"generate", "clarification", "fallback"}:
                        meta_container["terminal_node"] = node_name
                        output = event["data"].get("output", {})
                        if isinstance(output, dict):
                            if output.get("sources"):
                                meta_container["sources"] = output["sources"]
                            if output.get("generation") and not meta_container.get(
                                "has_llm_stream"
                            ):
                                token_queue.put(output["generation"])

        except Exception as exc:
            token_queue.put(exc)
        finally:
            token_queue.put(_STREAM_DONE)

    worker = threading.Thread(
        target=lambda: asyncio.run(_consume_events()), daemon=True
    )
    worker.start()

    while True:
        item = token_queue.get()
        if item is _STREAM_DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield item

    worker.join()


# ---------------------------------------------------------------------------
# Handle Input
# ---------------------------------------------------------------------------
prompt = st.chat_input("Tanya soalan tentang Tonton di sini...")
if st.session_state.sample_query:
    prompt = st.session_state.sample_query
    st.session_state.sample_query = ""

if prompt:
    # 1. Display & store user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {"content": HumanMessage(content=prompt), "text": prompt}
    )

    # 2. Stream assistant response with dynamic live status pill
    with st.chat_message("assistant"):
        meta: Dict[str, Any] = {
            "sources": [],
            "has_llm_stream": False,
            "terminal_node": None,
        }
        status_area = st.empty()
        response_area = st.empty()

        full_text = ""
        first_token_received = False
        t_start = time.perf_counter()

        try:
            for item in stream_rag_tokens(
                prompt_text=prompt,
                thread_id=st.session_state.session_id,
                meta_container=meta,
            ):
                if isinstance(item, _StatusUpdate):
                    # Show styled live pipeline step badge ONLY if tokens haven't started
                    if not first_token_received:
                        status_area.markdown(
                            f'<div class="status-pill"><span class="status-dot"></span> {item.label}</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    # First real token → clear status immediately
                    if not first_token_received:
                        status_area.empty()
                        first_token_received = True
                    full_text += item
                    response_area.markdown(full_text + " ▌")

            # Final cleanup: clear status area permanently
            status_area.empty()

            # Clean any accidental unclosed leading backticks
            cleaned_text = full_text.strip()
            if cleaned_text.startswith("```") and not cleaned_text.endswith("```"):
                cleaned_text = cleaned_text.lstrip("`").strip()
            full_text = cleaned_text

            # Final render (remove typing cursor)
            response_area.markdown(full_text)
            elapsed = time.perf_counter() - t_start

            # --- Source attribution cards ---
            sources = meta.get("sources", [])
            _render_sources(sources)

            # --- Clickable clarification suggestions ---
            suggestions: list = []
            if meta.get("terminal_node") == "clarification" and sources:
                suggestions = [s["question"] for s in sources if s.get("question")]
                if suggestions:
                    st.markdown('<div style="font-size:0.85rem; color:#94a3b8; margin-top:8px;">💡 Soalan berkaitan yang boleh anda pilih:</div>', unsafe_allow_html=True)
                    for suggestion in suggestions:
                        if st.button(
                            f"👉 {suggestion}",
                            key=f"sug_{hash(suggestion)}_{len(st.session_state.messages)}",
                            use_container_width=True,
                        ):
                            st.session_state.sample_query = suggestion
                            st.rerun()

            # --- Response time badge ---
            st.markdown(
                f'<div class="response-meta">⏱️ Dijana dalam {elapsed:.1f}s</div>',
                unsafe_allow_html=True,
            )

            # --- Feedback buttons ---
            msg_idx = len(st.session_state.messages)
            _render_feedback(msg_idx)

            # --- Save assistant message ---
            st.session_state.messages.append(
                {
                    "content": AIMessage(content=full_text or ""),
                    "text": full_text or "",
                    "sources": sources,
                    "suggestions": suggestions,
                    "elapsed_seconds": elapsed,
                }
            )

        except Exception as e:
            logger.error(f"Streamlit UI Exception: {e}")
            status_area.empty()
            st.error("⚠️ Sistem sedang mengalami kesukaran untuk memproses permintaan anda. Sila cuba sebentar lagi atau klik 'Perbualan Baharu'.")
