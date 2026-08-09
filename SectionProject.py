import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

AGENT_OPTIONS = [
    "Research Agent",
    "Coding Agent",
    "Data Analyst",
    "Document Agent",
    "Email Agent",
    "Planning Agent",
]

AGENT_DESCRIPTIONS = {
    "Research Agent": "A research specialist that summarizes information clearly and points out key facts.",
    "Coding Agent": "A software engineer that explains code, proposes fixes, and improves implementation quality.",
    "Data Analyst": "A data analyst that interprets trends, metrics, and provides concise insights.",
    "Document Agent": "A documentation specialist that turns raw ideas into polished documents.",
    "Email Agent": "A communications specialist that rewrites prompts into polished email messages.",
    "Planning Agent": "A planning specialist that turns a request into a structured action plan.",
}

SYSTEM_PROMPTS = {
    "Research Agent": "You are a research specialist. Answer the user's prompt with clear findings, useful context, and practical next steps.",
    "Coding Agent": "You are a senior software engineer. Help with coding tasks, explain the solution, and provide concise, correct guidance.",
    "Data Analyst": "You are a data analyst. Interpret the request, explain the results in plain language, and provide useful insights.",
    "Document Agent": "You are a documentation specialist. Transform the user's request into a professional document structure.",
    "Email Agent": "You are a communications specialist. Turn the request into a polished email suitable for professional use.",
    "Planning Agent": "You are a planning specialist. Convert the request into a practical action plan with clear steps and priorities.",
}


def build_groq_model(api_key: str, temperature: float = 0.7) -> ChatGroq:
    if not api_key or not api_key.strip():
        raise ValueError("Please enter a Groq API key in the sidebar before sending a prompt.")

    os.environ["GROQ_API_KEY"] = api_key.strip()
    return ChatGroq(
        api_key=os.environ["GROQ_API_KEY"],
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        max_tokens=4096,
    )


def get_agent_response(agent_name: str, prompt: str, api_key: str, temperature: float = 0.7) -> str:
    if agent_name not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported agent: {agent_name}")

    model = build_groq_model(api_key, temperature)
    response = model.invoke([
        SystemMessage(content=SYSTEM_PROMPTS[agent_name]),
        HumanMessage(content=prompt),
    ])
    return str(response.content)


CATEGORY_TO_AGENT = {
    "research": "Research Agent",
    "coding": "Coding Agent",
    "code": "Coding Agent",
    "data_analysis": "Data Analyst",
    "data analysis": "Data Analyst",
    "email": "Email Agent",
    "document": "Document Agent",
    "planning": "Planning Agent",
    "plan": "Planning Agent",
}

CLASSIFIER_SYSTEM_PROMPT = (
    "You are a routing classifier for a multi-agent assistant. Read the user's "
    "request and choose exactly one category that best matches it from this "
    "list: research, coding, data_analysis, email, document, planning. "
    "Reply with ONLY the category word, lowercase, nothing else."
)


def classify_agent(prompt: str, api_key: str) -> str:
    """Classify a prompt into one of AGENT_OPTIONS. Uses temperature=0 for a
    consistent, deterministic-as-possible classification. Falls back to
    'Research Agent' if the model's reply doesn't match a known category
    (rather than raising), so a routing hiccup never crashes the app -
    the same failure mode that broke LangGraph.py's routing."""
    model = build_groq_model(api_key, temperature=0.0)
    response = model.invoke([
        SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
        HumanMessage(content=f"Request:\n\n{prompt}"),
    ])
    category = str(response.content).strip().lower().strip(".\"' ")
    return CATEGORY_TO_AGENT.get(category, "Research Agent")


DOCUMENT_AGENT_RAG_SYSTEM_PROMPT = (
    "You are a documentation specialist. Transform the user's request into a "
    "well-structured, professional document, grounded ONLY in the provided "
    "context excerpts from the uploaded PDF. If the context does not contain "
    "enough information to answer confidently, say so explicitly rather than "
    "guessing or inventing details."
)


def get_document_agent_response(
    prompt: str,
    context_chunks: list[str],
    api_key: str,
    temperature: float = 0.7,
) -> str:
    """Document Agent variant that grounds its answer in retrieved PDF
    excerpts instead of relying solely on the model's own knowledge."""
    model = build_groq_model(api_key, temperature)

    if context_chunks:
        context_text = "\n\n".join(
            f"[Excerpt {i + 1}]: {chunk}" for i, chunk in enumerate(context_chunks)
        )
    else:
        context_text = "(No relevant excerpts were found in the uploaded PDF.)"

    response = model.invoke([
        SystemMessage(content=DOCUMENT_AGENT_RAG_SYSTEM_PROMPT),
        HumanMessage(content=f"Context from uploaded PDF:\n{context_text}\n\nRequest:\n{prompt}"),
    ])
    return str(response.content)
