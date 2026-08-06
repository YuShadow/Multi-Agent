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

