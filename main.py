import streamlit as st

st.set_page_config(
    page_title="Multi-Agent Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent AI Assistant")

with st.sidebar:
    st.header("Settings")

    model = st.selectbox(
        "Model",
        ["GPT-5.5", "GPT-4.1", "Llama 3", "Gemini"]
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.7
    )

    st.divider()

    st.checkbox("Research Agent", True)
    st.checkbox("Coder Agent", True)
    st.checkbox("Critic Agent", True)
    st.checkbox("Memory Agent", True)

    st.button("Clear Chat")

    left, right = st.columns([3,1])

with left:
    st.chat_message("user").write("Explain transformers.")

    st.chat_message("assistant").write(
        "Here is the final response..."
    )

with right:
    st.subheader("Agents")

    st.success("Coordinator ✓")
    st.info("Research ✓")
    st.warning("Coder Running")
    st.error("Critic Waiting")

with st.expander("Research Agent"):
    st.write("Searching documentation...")
    st.write("Found 5 relevant papers.")

# with st.expander("Coding Agent"):
#     st.code(code)

# with st.expander("Critic Agent"):
#     st.write("Reviewing answer...")

progress = st.progress(0)

progress.progress(20)
progress.progress(40)
progress.progress(70)
progress.progress(100)


chat, agents, memory, logs = st.tabs(
    [" Chat", "Agents", " Memory", " Logs"]
)

with chat:
    ...

with agents:
    ...

with memory:
    ...

with logs:
    ...