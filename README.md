# Multi-Agent AI Assistant

A Streamlit-based assistant that routes user prompts to one of six specialized agents
(Research, Coding, Data Analysis, Document, Email, Planning), backed by Groq-hosted LLMs
via LangChain. The repo also contains a separate, more advanced LangGraph orchestration
prototype with a RAG pipeline, which is not yet wired into the deployed app — see
[Project Status](#project-status) below.

## Table of Contents
- [Project Status](#project-status)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Roadmap](#roadmap)

## Project Status

This repository currently contains **two independent implementations** of the multi-agent
concept, built at different points in the project and not yet merged:

| | `main.py` + `SectionProject.py` | `LangGraph.py` |
|---|---|---|
| **Status** | ✅ Deployed / runnable as the Streamlit app | 🧪 Standalone prototype, not connected to the app |
| **Orchestration** | Manual dropdown selection — user picks the agent directly | LangGraph `StateGraph` — a supervisor node reads the prompt and routes automatically |
| **Retrieval** | None | RAG pipeline over local `.txt` files (FAISS + sentence-transformers) feeds the Document agent |
| **Memory** | None | `InMemorySaver` checkpointing (session-scoped) |
| **API key input** | Entered in the Streamlit sidebar at runtime | Entered via a blocking terminal `input()` prompt |

**In short:** if you run `streamlit run main.py` today, you get the simple dropdown
version. `LangGraph.py` demonstrates the more sophisticated routing/RAG approach but has
to be run separately (`python LangGraph.py`) and does not affect the Streamlit app. Merging
these into one system is the main next step — see [Roadmap](#roadmap).

## Features

**Currently working (via `main.py`):**
- Six selectable agents: Research, Coding, Data Analysis, Document, Email, Planning
- Per-agent system prompts tailored to that agent's role
- Adjustable temperature via a sidebar slider
- Chat-style message history within a session
- Groq API key entered directly in the UI (not stored beyond the browser session)

**Prototyped separately (via `LangGraph.py`, not yet in the app):**
- Automatic routing: a supervisor node classifies the user's request and picks the agent,
  instead of the user picking manually
- Retrieval-augmented generation: the Document agent retrieves relevant chunks from local
  `.txt` files (via FAISS similarity search) and grounds its output in them
- Session-scoped conversation memory via LangGraph's checkpointer

## Repository Structure

```
.
├── main.py             # Streamlit app entrypoint (the deployed version)
├── SectionProject.py    # Agent definitions, system prompts, and Groq call logic used by main.py
├── LangGraph.py           # Standalone LangGraph + RAG prototype (not imported by main.py)
├── starter.py               # ngrok tunnel helper for exposing a local Streamlit instance
└── requirements.txt           # Python dependencies
```

## Architecture

### Deployed app (`main.py` / `SectionProject.py`)

```
User selects an agent (dropdown) → enters a prompt
        ↓
SectionProject.get_agent_response(agent_name, prompt, api_key, temperature)
        ↓
ChatGroq (llama-3.3-70b-versatile) called once with:
   [SystemMessage: agent-specific prompt, HumanMessage: user's prompt]
        ↓
Response displayed in the Streamlit chat window
```

Each of the six agents in `SectionProject.py` (`AGENT_OPTIONS`) maps to its own entry in
`SYSTEM_PROMPTS`. There is no automatic routing — the user explicitly picks the agent from
a `st.selectbox`.

### Prototype (`LangGraph.py`)

```
START → supervisor_node
           ↓ (classifies the topic into one of 6 categories)
   ┌───────┼────────┬─────────┬─────────┬──────────┐
research  coding  data_analysis  email  document  planning
                                            │
                                    (document agent also
                                     retrieves context via
                                     FAISS + sentence-transformers
                                     from local .txt files)
           ↓
          END
```

`supervisor_node` calls the LLM to classify the incoming topic, then `route_task` sends
execution to the matching node via `add_conditional_edges`. The graph is compiled with an
`InMemorySaver` checkpointer, keyed by a `thread_id`, for session-scoped memory.

## Setup

### Prerequisites
- Python 3.10+
- A [Groq API key](https://console.groq.com/keys)

### Installation

```bash
git clone <your-repo-url>
cd Multi-Agent
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` currently covers only the deployed app's dependencies (`streamlit`,
`langchain`, `langchain-groq`, `langchain-core`). To run `LangGraph.py` as well, you'll
additionally need:

```bash
pip install langgraph sentence-transformers faiss-cpu scikit-learn numpy langchain-community
```

These will be folded into `requirements.txt` as the two implementations are merged.

## Usage

### Running the deployed app

```bash
streamlit run main.py
```

Then in the browser:
1. Paste your Groq API key into the sidebar.
2. Choose an agent from the dropdown.
3. Enter a prompt and click **Send to agent**.

### Running the LangGraph/RAG prototype standalone

```bash
python LangGraph.py
```

This will prompt for your API key in the terminal, then build a FAISS index over `.txt`
files found under `FOLDER_PATH`, and run an example topic defined near the bottom of the
file.

### Exposing a local instance publicly (optional)

```bash
streamlit run main.py &> log.txt &
python starter.py
```

`starter.py` opens an ngrok tunnel to port 8501 and prints a public URL.

## Roadmap

- [ ] Merge `LangGraph.py`'s supervisor routing into `main.py` so the app auto-selects the
      agent instead of requiring a manual dropdown pick
- [ ] Wire the RAG/document-retrieval pipeline into the deployed app
- [ ] Bring the `LangGraph.py` API key flow in line with the sidebar pattern in `main.py`
- [ ] Complete `requirements.txt` to cover both implementations
