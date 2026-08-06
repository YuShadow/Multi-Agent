# ============================================
# LangGraph Multi-Agent Research Writer
# Groq Version (No ReAct Tool Calling Issue)
# ============================================
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
import json
import time
import glob
import numpy as np
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss

#----------------------------

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

FOLDER_PATH = "/workspaces/Multi-Agent/"

txt_files = sorted(glob.glob(os.path.join(FOLDER_PATH, "**", "*.txt"), recursive=True))
print(f"Found {len(txt_files)} .txt files")
for f in txt_files[:10]:
    print(" -", f)

def load_txt_folder(folder_path: str) -> List[Dict]:

    file_paths = sorted(glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True))
    documents = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            # fallback for files not saved as utf-8
            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

        if text.strip():
            documents.append({"filename": os.path.basename(path), "text": text})

    return documents


documents = load_txt_folder(FOLDER_PATH)
print(f"✅ Loaded {len(documents)} documents")

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:

    text = " ".join(text.split()) 
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

all_chunks = []
chunk_sources = []  

for doc in documents:
    doc_chunks = chunk_text(doc["text"], chunk_size=800, overlap=150)
    all_chunks.extend(doc_chunks)
    chunk_sources.extend([doc["filename"]] * len(doc_chunks))

print(f"Total chunks created: {len(all_chunks)}")

chunk_embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

embedding_dim = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(chunk_embeddings)

print(f"✅ Index built successfully")
print(f"Vectors stored: {index.ntotal}")
print(f"Vector dimension: {embedding_dim}")

def retrieve_chunks(query: str, top_k: int = 3) -> List[Dict]:

    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append({
            "chunk": all_chunks[idx],
            "source_file": chunk_sources[idx],
            "distance": float(distances[0][rank])
        })
    return results



#----------------------------
apikey = input("Enter the api key: ")
# -----------------------------
# Model
# -----------------------------.
os.environ["GROQ_API_KEY"] = apikey
model = ChatGroq(
    api_key=os.environ["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0.5
)



class MultiAgentState(TypedDict):
    topic: str
    research_findings: str
    draft: str
    review_feedback: str
    final_output: str
    iteration: int
    max_iterations: int
    status: str



# -----------------------------
# Agents
# -----------------------------

def research_node(state: MultiAgentState):

    print(" RESEARCHER")

    response = model.invoke([
        SystemMessage(
            content="""
            You are a research assistant.
            Provide detailed and accurate research.
            Include important fact.
            """
        ),
        HumanMessage(
            content=f"""
            Research topic:

            {state['topic']}
            """
        )
    ])

    return {
        "research_findings": response.content,
        "status": "researched"
    }

def coding_node(state: MultiAgentState):

    print("coding agent")

    response = model.invoke([
        SystemMessage(
            content="""
            you are a coding assistant,
            generate well organized code with comments and provide syntax correction 
            and also provide a brief explanation of the code.
            """
        ),
        HumanMessage(
            content=f"""
            Draft/Prompt:
            {state['draft']}
            
            Feedback:
            {state.get('review_feedback', '')}
            """
        )
    ])

    return {
        "final_output": response.content,
        "status": "improved"
    }
def Data_Analysis_node(state: MultiAgentState):

    print("🔄 data Analysis")

    response = model.invoke([
        SystemMessage(
            content="""
            you are a data analysis assistant,provide data analysis 
            and generate well organized data analysis report if needed
            """
        ),
        HumanMessage(
            content=f"""
            Draft:

            {state['draft']}


            Feedback:

            {state['review_feedback']}
            """
        )
    ])

    return {
        "final_output": response.content,
        "status": "improved"
    }

def email_node(state: MultiAgentState):

    print("Email")

    response = model.invoke([
        SystemMessage(
            content="""
            Trnsform the text into an email format.
            Return only the improved email format.
            """
        ),
        HumanMessage(
            content=f"""
            Draft:

            {state['draft']}


            Feedback:

            {state['review_feedback']}
            """
        )
    ])

    return {
        "final_output": response.content,
        "status": "transformed"
    }

def document_node(state: MultiAgentState):
    print("document agent with RAG")

    query_text = state.get("draft") if state.get("draft") else state.get("topic", "")

    retrieved_chunks = retrieve_chunks(query_text, top_k=3)
    
    context_text = "\n\n".join(
        [f"[Source: {c['source_file']}]: {c['chunk']}" for c in retrieved_chunks]
    )

    response = model.invoke([
        SystemMessage(
            content="""
            You are a document creation assistant.
            Transform the provided text into a well-structured document using ONLY 
            the provided context and draft text. 
            If feedback is provided, incorporate it into the document structure.
            Return ONLY the transformed document.
            """
        ),
        HumanMessage(
            content=f"""
            Context from Documents:
            {context_text}

            Draft:
            {state.get('draft', '')}

            Feedback:
            {state.get('review_feedback', '')}
            """
        )
    ])

    return {
        "draft": response.content,
        "final_output": response.content,
        "status": "transformed"
    }

def planning_node(state: MultiAgentState):
    print("Planner")
    
    response = model.invoke([
        SystemMessage(
            content=f"""
                      You are an expert at planning,
                      plan for the provided topic 
                      with the shortest amount of steps.
                    """
        ),
        HumanMessage(
            content=f"""
            Draft:

            {state['draft']}


            Feedback:

            {state['review_feedback']}
            """
        )
    ])
    return {
            "final_output": response.content,
            "status": "planned"
        }

def supervisor_node(state: MultiAgentState):
    response = model.invoke([
        SystemMessage(content="""
             review the user's request and choose a category from the following options according to the user's request:
            research, coding, data_analysis, email, document, planning.
            Reply with only the category word.
        """),
        HumanMessage(
            content=f"""
            topic:

            {state['topic']}
            """
        )
    ])
    category = response.content.strip().lower()
    
    # Preserve research findings if present; otherwise fall back to the initial topic
    draft_content = state["research_findings"] if state.get("research_findings") else state["topic"]
    
    return {
        "draft": draft_content, 
        "status": category
    }


def route_task(state: MultiAgentState):
    # must return a key that matches the conditional_edges mapping below
    return state["status"]






builder = StateGraph(MultiAgentState)

builder.add_node("supervisor", supervisor_node) 
builder.add_node("research", research_node)
builder.add_node("coding", coding_node)
builder.add_node("data analysis", Data_Analysis_node)
builder.add_node("email", email_node)
builder.add_node("document", document_node)
builder.add_node("planning", planning_node)

builder.add_edge(START,"supervisor")


builder.add_conditional_edges(
    "supervisor",
    route_task,
    {
        "coding": "coding",
        "data analysis": "data analysis",
        "email": "email",
        "document": "document",
        "planning": "planning",
        "research": "research",
    }
)

for node in ["coding", "data analysis", "email", "document", "planning"]:
    builder.add_edge(node, END)

graph = builder.compile(
    checkpointer=InMemorySaver()
)


topic = "In the document, was the discovery of a single brass key inside an abandoned typewriter expected?"


result = graph.invoke(
    {
        "topic": topic,
        "research_findings": "",
        "draft": "",
        "review_feedback": "",
        "final_output": "",
        "iteration": 0,
        "max_iterations": 1,
        "status": "started"
    },
    config={
        "configurable":{
            "thread_id":"ai-healthcare"
        }
    }
)

print("\n" + "="*60)
print("FINAL OUTPUT")
print("="*60)

print(
    result["final_output"]
    if result["final_output"]
    else result["draft"]
)

