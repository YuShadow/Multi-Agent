import streamlit as st

from SectionProject import (
    AGENT_DESCRIPTIONS,
    AGENT_OPTIONS,
    classify_agent,
    get_agent_response,
    get_document_agent_response,
)
from rag import extract_pdf_text, chunk_text, PDFIndex

st.set_page_config(page_title="Multi-Agent Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent AI Assistant")
st.caption("Choose an agent, enter a prompt, and send it using your Groq API key.")


# -----------------------------
# Cached resources / session state
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def load_embedding_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


if "messages" not in st.session_state:
    st.session_state.messages = []

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None

if "pdf_signature" not in st.session_state:
    st.session_state.pdf_signature = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=st.session_state.groq_api_key,
        key="groq_api_key_input",
    )
    if api_key != st.session_state.groq_api_key:
        st.session_state.groq_api_key = api_key

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

    st.divider()

    st.subheader("📄 Document Agent (RAG)")
    uploaded_pdf = st.file_uploader(
        "Upload a PDF to activate the Document Agent",
        type=["pdf"],
        help="Uploading a PDF automatically routes your next prompts to the "
             "Document Agent, grounded in this file's content. This takes "
             "priority over auto-detect below.",
    )

    if uploaded_pdf is not None:
        signature = f"{uploaded_pdf.name}-{uploaded_pdf.size}"
        if st.session_state.pdf_signature != signature:
            with st.spinner(f"Reading and indexing {uploaded_pdf.name}..."):
                try:
                    text = extract_pdf_text(uploaded_pdf)
                    chunks = chunk_text(text)
                    if not chunks:
                        st.error(
                            "No extractable text found in this PDF (it may be a "
                            "scanned/image-only document with no text layer)."
                        )
                        st.session_state.pdf_index = None
                        st.session_state.pdf_signature = None
                        st.session_state.pdf_name = None
                    else:
                        embedding_model = load_embedding_model()
                        index = PDFIndex(embedding_model)
                        index.build(chunks)
                        st.session_state.pdf_index = index
                        st.session_state.pdf_signature = signature
                        st.session_state.pdf_name = uploaded_pdf.name
                except Exception as exc:
                    st.error(f"Failed to read PDF: {exc}")
                    st.session_state.pdf_index = None
                    st.session_state.pdf_signature = None
                    st.session_state.pdf_name = None

        if st.session_state.pdf_index is not None:
            st.success(
                f"Indexed {len(st.session_state.pdf_index.chunks)} chunks from "
                f"**{st.session_state.pdf_name}**"
            )
    else:
        st.session_state.pdf_index = None
        st.session_state.pdf_signature = None
        st.session_state.pdf_name = None

    if st.session_state.pdf_index is not None:
        if st.button("Remove PDF / return to normal routing", use_container_width=True):
            st.session_state.pdf_index = None
            st.session_state.pdf_signature = None
            st.session_state.pdf_name = None
            st.rerun()

    st.divider()

    pdf_active = st.session_state.pdf_index is not None and bool(st.session_state.pdf_index.chunks)

    st.subheader("🎯 Routing")
    auto_detect = st.checkbox(
        "Auto-detect agent based on prompt",
        value=False,
        disabled=pdf_active,
        help="When enabled, the app reads your prompt and picks the best "
             "agent for you instead of you choosing manually.",
    )
    if pdf_active:
        st.caption("Disabled while a PDF is active — the Document Agent handles all prompts.")

    st.divider()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

    st.info("Your API key is stored only for the current browser session.")


# -----------------------------
# Agent selection area
# -----------------------------
if pdf_active:
    selected_agent = "Document Agent"
    st.info(
        f"📄 A PDF is uploaded (**{st.session_state.pdf_name}**) — prompts will "
        f"automatically be handled by the **Document Agent**, grounded in this file."
    )
elif auto_detect:
    selected_agent = None  # decided per-prompt, after the user submits
    st.info("🎯 **Auto-detect is on** — the right agent will be picked based on what you type below.")
else:
    selected_agent = st.selectbox("Choose an agent", AGENT_OPTIONS, index=0)
    st.info(AGENT_DESCRIPTIONS[selected_agent])

with st.form("agent_prompt_form", clear_on_submit=True):
    prompt = st.text_area(
        "Prompt",
        placeholder="Ask the selected agent to help with research, code, analysis, writing, or planning...",
        key="agent_prompt_input",
    )
    submitted = st.form_submit_button("Send to agent", use_container_width=True)

if submitted:
    if not st.session_state.groq_api_key:
        st.error("Please enter a Groq API key before sending a prompt.")
    elif not prompt.strip():
        st.warning("Please enter a prompt before sending it.")
    else:
        retrieved_chunks = None
        agent_used = None
        try:
            if pdf_active:
                agent_used = "Document Agent"
                with st.spinner("Running Document Agent..."):
                    retrieved_chunks = st.session_state.pdf_index.search(prompt, top_k=3)
                    response = get_document_agent_response(
                        prompt,
                        retrieved_chunks,
                        st.session_state.groq_api_key,
                        temperature=temperature,
                    )

            elif auto_detect:
                with st.spinner("🎯 Detecting the best agent for this prompt..."):
                    agent_used = classify_agent(prompt, st.session_state.groq_api_key)
                st.success(f"🎯 Auto-selected agent: **{agent_used}**")
                with st.spinner(f"Running {agent_used}..."):
                    response = get_agent_response(
                        agent_used,
                        prompt,
                        st.session_state.groq_api_key,
                        temperature=temperature,
                    )

            else:
                agent_used = selected_agent
                with st.spinner(f"Running {agent_used}..."):
                    response = get_agent_response(
                        agent_used,
                        prompt,
                        st.session_state.groq_api_key,
                        temperature=temperature,
                    )

        except Exception as exc:
            st.error(f"The request failed: {exc}")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "agent_used": agent_used,
                "retrieved_chunks": retrieved_chunks,
            })

st.divider()

if not st.session_state.messages:
    st.info("Start the conversation by selecting an agent (or enabling auto-detect) and sending a prompt.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            agent_used = message.get("agent_used")
            if agent_used:
                st.caption(f"🏷️ Handled by: **{agent_used}**")

            chunks = message.get("retrieved_chunks")
            if chunks:
                with st.expander(f"📄 {len(chunks)} excerpt(s) retrieved from the PDF"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Excerpt {i}:**")
                        st.text(chunk)