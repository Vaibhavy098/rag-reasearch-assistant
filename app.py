
import streamlit as st

from src.rag_pipeline import RAGResearchAssistant


# ---------------------------------------------------------
# 1. Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Research Lab",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 2. Research collection
# ---------------------------------------------------------

PAPERS = [
    {
        "name": "Attention Is All You Need",
        "short": "Transformer",
        "year": "2017",
        "category": "ARCHITECTURE",
    },
    {
        "name": (
            "BERT: Pre-training of Deep Bidirectional "
            "Transformers for Language Understanding"
        ),
        "short": "BERT",
        "year": "2018",
        "category": "LANGUAGE UNDERSTANDING",
    },
    {
        "name": "Language Models are Few-Shot Learners",
        "short": "GPT-3",
        "year": "2020",
        "category": "FEW-SHOT LEARNING",
    },
    {
        "name": (
            "RoBERTa: A Robustly Optimized "
            "BERT Pretraining Approach"
        ),
        "short": "RoBERTa",
        "year": "2019",
        "category": "PRE-TRAINING",
    },
    {
        "name": (
            "Exploring the Limits of Transfer Learning "
            "with a Unified Text-to-Text Transformer"
        ),
        "short": "T5",
        "year": "2019",
        "category": "TEXT-TO-TEXT TRANSFER",
    },
]

SUGGESTIONS = [
    "How does self-attention work?",
    "Compare BERT and GPT-3.",
    "What did RoBERTa change about BERT?",
    "How does T5 approach NLP tasks?",
]


# ---------------------------------------------------------
# 3. Premium dark visual system
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background: #111116;
        color: #F0EEF7;
    }

    [data-testid="stHeader"] {
        background: rgba(17, 17, 22, 0.95);
    }

    [data-testid="stSidebar"] {
        background: #17171F;
        border-right: 1px solid #2C2B38;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    h1, h2, h3 {
        color: #F5F3FC !important;
        letter-spacing: -0.035em;
    }

    p, li, label {
        color: #D0CDDC;
    }

    [data-testid="stCaptionContainer"] {
        color: #9692AA;
    }

    .brand-mark {
        display: inline-flex;
        width: 38px;
        height: 38px;
        align-items: center;
        justify-content: center;
        border-radius: 11px;
        background: #7665D8;
        color: white;
        font-size: 21px;
        margin-right: 10px;
        vertical-align: middle;
    }

    .brand-name {
        color: #F5F3FC;
        font-size: 1.12rem;
        font-weight: 750;
        letter-spacing: -0.035em;
    }

    .eyebrow {
        color: #A99AF5;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }

    .muted {
        color: #9995AD;
        font-size: 0.86rem;
        line-height: 1.65;
    }

    .paper-label {
        color: #9692AA;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        margin: 1rem 0 0.6rem 0;
    }

    .paper-item {
        padding: 0.65rem 0.7rem;
        margin: 0.2rem 0;
        border: 1px solid transparent;
        border-radius: 8px;
        background: transparent;
    }

    .paper-item-title {
        color: #D9D5E6;
        font-size: 0.82rem;
        font-weight: 600;
        line-height: 1.4;
    }

    .paper-item-meta {
        color: #89859D;
        font-size: 0.66rem;
        letter-spacing: 0.06em;
        margin-top: 0.25rem;
    }

    .section-title {
        color: #F2EFFA;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.025em;
    }

    .section-subtitle {
        color: #9692AA;
        font-size: 0.81rem;
        margin-top: 0.25rem;
        margin-bottom: 1rem;
    }

    .welcome-title {
        font-size: clamp(1.8rem, 3vw, 2.7rem);
        font-weight: 760;
        letter-spacing: -0.055em;
        line-height: 1.15;
        color: #F5F3FC;
    }

    .welcome-description {
        color: #A5A1B7;
        max-width: 700px;
        font-size: 0.94rem;
        line-height: 1.8;
        margin-top: 0.8rem;
    }

    .answer-label {
        color: #B5A7FF;
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        margin-bottom: 0.45rem;
    }

    /* Chat message styling */

    div[data-testid="stChatMessage"] {
        background: #191920;
        border: 1px solid #2B2A37;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    /* Chat input: force readable colors in both app themes */

    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div {
        background-color: #211F2B !important;
        border: 1px solid #464154 !important;
        border-radius: 13px !important;
    }

    div[data-testid="stChatInput"] textarea,
    div[data-testid="stChatInput"] textarea:focus {
        background-color: #211F2B !important;
        color: #F2EFFA !important;
        -webkit-text-fill-color: #F2EFFA !important;
        caret-color: #A99BFF !important;
    }

    div[data-testid="stChatInput"] textarea::placeholder {
        color: #A5A1B7 !important;
        -webkit-text-fill-color: #A5A1B7 !important;
        opacity: 1 !important;
    }

    div[data-testid="stChatInput"] button {
        color: #F2EFFA !important;
    }

    /* Expanders and buttons */

    div[data-testid="stExpander"] {
        background: #17171F;
        border: 1px solid #302E3C;
        border-radius: 10px;
    }

    div[data-testid="stExpander"] summary {
        color: #E6E2F1;
    }

    div[data-testid="stButton"] > button {
        border-radius: 9px;
        border-color: #373544;
        transition: border-color 120ms ease;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: #8271E7;
        color: #D8D0FF;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: #7665D8;
        border-color: #7665D8;
        color: #FFFFFF;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #8A78F0;
    }

    hr {
        border-color: #302E3C;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 4. Session state and backend initialization
# ---------------------------------------------------------

def initialize_session() -> None:
    defaults = {
        "chat_messages": [],
        "rag_assistant": None,
        "initialization_error": None,
        "last_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Remove legacy UI state from the previous version.
    # Paper selection is no longer part of the interface.
    st.session_state.pop("selected_paper", None)


def initialize_assistant() -> bool:
    if st.session_state.rag_assistant is not None:
        return True

    try:
        with st.spinner("Loading the research engine..."):
            st.session_state.rag_assistant = RAGResearchAssistant(
                memory_turns=4
            )

        st.session_state.initialization_error = None
        return True

    except Exception as error:
        st.session_state.initialization_error = str(error)
        return False


# ---------------------------------------------------------
# 5. Chat, source and feedback functions
# ---------------------------------------------------------

def submit_question(question: str) -> None:
    question = question.strip()

    if not question:
        return

    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    assistant = st.session_state.rag_assistant

    try:
        with st.spinner("Searching papers and generating your answer..."):
            result = assistant.answer_question(
                question,
                top_k=4,
            )

        message = {
            "role": "assistant",
            "content": result["answer"],
            "question": question,
            "feedback": None,
            "sources": result["sources"],
            "retrieved_chunks": result["retrieved_chunks"],
            "search_query": result["search_query"],
        }

        st.session_state.last_error = None

    except Exception as error:
        message = {
            "role": "assistant",
            "content": (
                "I couldn't generate an answer. Check the local "
                "model and application setup, then try again."
            ),
            "question": question,
            "feedback": None,
            "sources": [],
            "retrieved_chunks": [],
            "search_query": None,
            "debug_error": str(error),
        }

        st.session_state.last_error = str(error)

    st.session_state.chat_messages.append(message)


def render_sources(message: dict) -> None:
    sources = message.get("sources", [])
    chunks = message.get("retrieved_chunks", [])

    with st.expander(
        f"⌕  Evidence & sources · {len(sources)} references"
    ):
        if not sources:
            st.info("No research sources were retrieved.")
            return

        for source in sources:
            st.markdown(
                f"**{source.get('source', 'Unknown source')}**"
                f" · Page {source.get('page_number', '?')}"
                f" · Similarity "
                f"`{source.get('similarity_score', 0.0):.3f}`"
            )

        if chunks:
            st.divider()
            st.caption("RETRIEVED PASSAGES")

        for index, chunk in enumerate(chunks, start=1):
            title = (
                f"{index}. {chunk.get('source', 'Unknown source')}"
                f" · Page {chunk.get('page_number', '?')}"
            )

            with st.expander(title):
                st.caption(
                    f"Chunk ID: {chunk.get('chunk_id', 'Unknown')} "
                    f" · Similarity: "
                    f"{chunk.get('similarity_score', 0.0):.3f}"
                )
                st.write(chunk.get("text", ""))


def render_feedback(message: dict, message_index: int) -> None:
    if not message.get("question") or message.get("debug_error"):
        return

    feedback = message.get("feedback")

    if feedback == "it_helped":
        st.caption("👍 Feedback recorded: Helpful")
        return

    if feedback == "not_helpful":
        st.caption("👎 Feedback recorded: Not helpful")
        return

    col_yes, col_no = st.columns([1, 1])

    with col_yes:
        if st.button(
            "👍 Helpful",
            key=f"helpful_{message_index}",
        ):
            saved = st.session_state.rag_assistant.record_feedback(
                message["question"],
                "it_helped",
            )

            if saved:
                message["feedback"] = "it_helped"
                st.rerun()
            else:
                st.warning(
                    "This turn has expired from memory. "
                    "Rate a more recent answer."
                )

    with col_no:
        if st.button(
            "👎 Not helpful",
            key=f"not_helpful_{message_index}",
        ):
            saved = st.session_state.rag_assistant.record_feedback(
                message["question"],
                "not_helpful",
            )

            if saved:
                message["feedback"] = "not_helpful"
                st.rerun()
            else:
                st.warning(
                    "This turn has expired from memory. "
                    "Rate a more recent answer."
                )


# ---------------------------------------------------------
# 6. Initialize application
# ---------------------------------------------------------

initialize_session()

if not initialize_assistant():
    st.error("Research Lab could not initialize.")

    st.code(
        st.session_state.initialization_error or "Unknown error"
    )

    st.info(
        "Check the FAISS index, processed chunks, embedding model, "
        "and required dependencies."
    )

    st.stop()


# ---------------------------------------------------------
# 7. Compact sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<span class="brand-mark">✳</span>'
        '<span class="brand-name">Research Lab</span>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="muted">Your private AI research workspace</p>',
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown(
        '<div class="paper-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**✳  Research Chat**")
    st.caption("Questions, answers and citations")

    if st.button(
        "＋  New conversation",
        use_container_width=True,
    ):
        st.session_state.rag_assistant.memory.clear()
        st.session_state.chat_messages = []
        st.session_state.last_error = None
        st.rerun()

    st.divider()

    # Non-selectable research paper list
    st.markdown(
        '<div class="paper-label">RESEARCH COLLECTION · 05 PAPERS</div>',
        unsafe_allow_html=True,
    )

    for index, paper in enumerate(PAPERS, start=1):
        st.markdown(
            f"""
            <div class="paper-item">
                <div class="paper-item-title">
                    {index:02d} &nbsp; {paper["short"]}
                </div>
                <div class="paper-item-meta">
                    {paper["category"]} · {paper["year"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div class="paper-label">LOCAL MODEL</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Qwen 2.5 · 3B**")
    st.caption("Ollama · Local inference")

    st.divider()

    assistant = st.session_state.rag_assistant

    st.markdown(
        '<div class="paper-label">MEMORY</div>',
        unsafe_allow_html=True,
    )

    st.progress(
        min(assistant.memory.turn_count / 4, 1.0),
        text=f"{assistant.memory.turn_count} / 4 recent turns",
    )

    st.caption(
        "Conversation memory is session-local and can reset "
        "when the app restarts."
    )

    st.divider()

    st.caption(
        "Answers are generated from retrieved excerpts. "
        "Always verify important claims against the sources."
    )


# ---------------------------------------------------------
# 8. Main research workspace
# ---------------------------------------------------------

header_left, header_right = st.columns(
    [4, 1],
    vertical_alignment="center",
)

with header_left:
    st.markdown(
        '<div class="eyebrow">DOCUMENT INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown("## Research workspace")

    st.markdown(
        '<p class="muted">'
        'Five foundational papers. One place to explore them.'
        '</p>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        '<div style="text-align:right; padding-top:0.5rem;">'
        '<span style="background:#202A25;color:#91D8B0;'
        'border:1px solid #355443;border-radius:20px;'
        'padding:6px 12px;font-size:12px;">'
        '● LOCAL WORKSPACE</span></div>',
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------------------------------------
# 9. Full-width research conversation
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Research conversation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-subtitle">'
    'ASK · EXPLORE · VERIFY'
    '</div>',
    unsafe_allow_html=True,
)

if not st.session_state.chat_messages:
    st.markdown(
        '<div class="welcome-title">'
        'What are you<br>researching today?'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="welcome-description">'
        'Ask a question across your research collection. '
        'Explore model architectures, compare findings, '
        'and inspect the retrieved passages behind each answer.'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    st.markdown(
        '<div class="paper-label">START WITH A QUESTION</div>',
        unsafe_allow_html=True,
    )

    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            use_container_width=True,
        ):
            submit_question(suggestion)
            st.rerun()


# Display the conversation history

for message_index, message in enumerate(
    st.session_state.chat_messages
):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(
                '<div class="answer-label">'
                'RESEARCH ANSWER'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("sources") is not None:
                render_sources(message)

            if message.get("search_query"):
                with st.expander("Retrieval details"):
                    st.caption("Standalone query sent to retrieval")
                    st.code(message["search_query"])

            if message.get("question"):
                render_feedback(message, message_index)

            if message.get("debug_error"):
                with st.expander("Technical error details"):
                    st.code(message["debug_error"])


# Chat input remains connected to the existing RAG pipeline

question = st.chat_input(
    "Ask a research question...",
)

if question and question.strip():
    submit_question(question)
    st.rerun()
