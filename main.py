import streamlit as st

from SectionProject import AGENT_DESCRIPTIONS, AGENT_OPTIONS, get_agent_response

st.set_page_config(page_title="Multi-Agent Assistant", page_icon="🤖", layout="wide")

st.title("🤖 Multi-Agent AI Assistant")
st.caption("Choose an agent, enter a prompt, and send it using your Groq API key.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

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

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []

    st.info("Your API key is stored only for the current browser session.")

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
        with st.spinner(f"Running {selected_agent}..."):
            try:
                response = get_agent_response(
                    selected_agent,
                    prompt,
                    st.session_state.groq_api_key,
                    temperature=temperature,
                )
            except Exception as exc:
                st.error(f"The request failed: {exc}")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response})

st.divider()

if not st.session_state.messages:
    st.info("Start the conversation by selecting an agent and sending a prompt.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
