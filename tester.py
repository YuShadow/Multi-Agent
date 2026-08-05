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


# -----------------------------
# Model
# -----------------------------.
os.environ["GROQ_API_KEY"] = "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"
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

    print("document")

    response = model.invoke([
        SystemMessage(
            content="""
            Trnsform the text into a document.
            Return only the document.
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
    return {"draft": state["research_findings"], "status": response.content.strip().lower()}


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


topic = "give me a python code to implement a simple hello world prompt"


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

