# Multi-Agent AI Assistant

A Streamlit-based assistant that routes user prompts to one of six specialized agents
(Research, Coding, Data Analysis, Document, Email, Planning), backed by Groq-hosted LLMs.
The app supports three complementary ways of picking the right agent — manual selection,
automatic classification, and PDF-triggered routing — all implemented directly in
LangChain, with no external orchestration framework.

## Table of Contents
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Design Rationale](#design-rationale)
- [Roadmap](#roadmap)

## Features

- Six specialist agents: Research, Coding, Data Analysis, Document, Email, Planning
- **Manual mode** — pick an agent from a dropdown
- **Auto-detect mode** — a classifier call (`classify_agent` in `SectionProject.py`) reads
  the prompt and picks the agent for you; the selection is shown in the UI both as an
  immediate banner and as a label on each chat message
- **PDF-triggered mode** — uploading a PDF automatically routes all prompts to the
  Document Agent, grounded in that file's content via a live-built FAISS index
  (`rag.py`); takes priority over auto-detect
- Adjustable temperature via a sidebar slider
- Groq API key entered directly in the UI (not stored beyond the browser session)

## Repository Structure

```
.
├── main.py             # Streamlit app entrypoint
├── SectionProject.py    # Agent definitions, system prompts, classifier, Groq call logic
├── rag.py                 # PDF text extraction, chunking, and FAISS retrieval
├── starter.py                # ngrok tunnel helper for exposing a local Streamlit instance
└── requirements.txt              # Python dependencies
```

## Architecture

```
                        User enters a prompt
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                        │
   PDF uploaded?          Auto-detect ON?            Manual mode
        │                       │                        │
        ▼                       ▼                        ▼
 rag.PDFIndex.search()   classify_agent(prompt)    User-picked agent
 (FAISS top-3 chunks)    (single LangChain call)    (dropdown)
        │                       │                        │
        ▼                       ▼                        ▼
 get_document_agent_response()   get_agent_response(agent_name, ...)
        │                       │                        │
        └───────────────────────┴───────────────────────┘
                                │
                    ChatGroq (llama-3.3-70b-versatile)
                                │
                Response + "Handled by: <agent>" label
                      displayed in the chat window
```

There is no orchestration framework here — `classify_agent()` and `get_agent_response()`
are direct functions in `SectionProject.py`, each issuing a single `model.invoke()` call.
Routing between the three modes (PDF / auto-detect / manual) is plain Python `if/elif`
logic in `main.py`.

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

`requirements.txt` includes `pypdf`, `sentence-transformers`, and `faiss-cpu` for the
PDF/RAG feature. Note that `sentence-transformers` pulls in PyTorch, so this install is
noticeably larger and slower than a minimal LangChain-only setup.

## Usage

```bash
streamlit run main.py
```

Then in the browser:
1. Paste your Groq API key into the sidebar.
2. Pick a routing mode:
   - Upload a PDF to activate the **Document Agent** automatically, or
   - Check **"Auto-detect agent based on prompt"** to let the app choose, or
   - Leave both off and pick an agent manually from the dropdown.
3. Enter a prompt and click **Send to agent**.

### Exposing a local instance publicly (optional)

```bash
streamlit run main.py &> log.txt &
python starter.py
```

`starter.py` opens an ngrok tunnel to port 8501 and prints a public URL.

## Design Rationale

An earlier version of this project explored [LangGraph](https://www.langchain.com/langgraph)
for automatic routing — a supervisor node classifying each request and routing it through
a `StateGraph`. That approach was not carried into the final application, in favor of
implementing the same automatic-routing goal with a single, direct LangChain
classification call (`classify_agent`).

**Why:**
- **Simplicity.** With three routing modes converging on two dispatch functions, the full
  request path from prompt to response is easy to trace without reasoning about graph
  state, node registration, or conditional edge maps.
- **Fewer failure modes.** A flat classifier-then-dispatch design has one place routing
  can go wrong — a single dictionary lookup, guarded with a fallback default — rather than
  a graph configuration where a mismatch between a prompt's expected output and a node's
  registered name can silently break a route.
- **Matches the deployment target.** The app is a single Streamlit script; a flat
  function-call architecture maps directly onto Streamlit's rerun-based execution model.

**Trade-off accepted:** an explicit state graph makes multi-step pipelines easier to
extend (e.g. chaining a research step into a document-drafting step, or a critic/revision
loop). Without that structure, extending this system to multi-step workflows would need
additional application-level state management. This is a deliberate scope decision for the
current version: three single-step routing modes, no multi-agent handoff between
specialists.

## Roadmap

- [ ] Add a lightweight state-passing mechanism for multi-step workflows (e.g. research
      findings feeding into a document draft)
- [ ] Support multiple uploaded documents rather than one at a time
- [ ] Extend the classifier with a confidence threshold or clarification step for
      ambiguous prompts
- [ ] Replace the hardcoded ngrok token in `starter.py` with an environment variable
