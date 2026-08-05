import os
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

os.environ["GROQ_API_KEY"] = "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"
model = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=4096
)

# Initialize HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

@tool
def search_web(query: str) -> str:
    """Search the web for information about a topic."""
    return f"""
    SEARCH RESULTS for "{query}":
    - The topic is rapidly growing with applications in AI.
    - Key researchers include Smith, Jones, and Patel.
    - Latest developments include new algorithms and frameworks.
    """

@tool
def check_grammar(text: str) -> str:
    """Check text for grammar and clarity issues."""
    return "Grammar check: The text is well-written with no major issues."

@tool
def check_syntax(text: str) -> str:
    """Check code for syntax errors"""
    return "Syntax check: the text has no syntax errors."

@tool
def data_analysis(text: str) -> str:
    """Check for patterns in the data and provide a consice summary."""
    return "The data has been analyzed." 

@tool
def text_to_doc(text: str) -> str:
    """Turn the provided text into a document format."""
    return "This text has been documentized."

@tool
def text_to_email(text: str) -> str:
    """Turn the provided text into an email format."""
    return "The text has been converted to email format."

researcher = create_agent(
    model=model,
    tools=[search_web],
    system_prompt="You are a Research Specialist. Gather comprehensive, accurate information."
)

coding = create_agent(
    model=model,
    tools=[check_syntax],
    system_prompt="You are an Expert Programmer. Check the provided code for syntax errors."
)

data_analyst = create_agent(
    model=model,
    tools=[data_analysis],
    system_prompt="You are a Data Analyst. Analyze the provided data and summarize key insights."
)

documenter = create_agent(
    model=model,
    tools=[text_to_doc],
    system_prompt="You are a Document Specialist. Convert the provided text into a well-structured document format."
)

