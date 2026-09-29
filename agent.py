from typing_extensions import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
import langgraph.checkpoint.memory
from mem0 import Memory
from memory_config import config
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
memory = Memory.from_config(config)

class SupportState(TypedDict):
    customer_id: str
    question: str
    category: str
    response: str
    ticket_id: str
    messages: Annotated[list, add_messages]

def classify_node(state):
    """Classify the question"""
    question = state["question"]
    
    if "how" in question.lower() or "what" in question.lower():
        category = "faq"
    elif "code" in question.lower() or "program" in question.lower():
        category = "technical"
    elif "price" in question.lower() or "cost" in question.lower():
        category = "billing"
    else:
        category = "escalate"
    
    print(f"[CLASSIFY] {category}")
    return {"category": category}

def faq_node(state):
    """Answer FAQ questions"""
    question = state["question"]
    customer_id = state["customer_id"]
    
    history = memory.search(
        query=f"customer {customer_id}",
        user_id=customer_id,
        limit=1
    )
    
    history_context = ""
    if history:
        history_context = f"\nCustomer history: {history[0]}\n"
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"You are a FAQ assistant.{history_context}Answer concisely."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    answer = response.choices[0].message.content
    print(f"[FAQ] Answer provided")
    return {"response": answer}

def technical_node(state):
    """Answer technical questions"""
    question = state["question"]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a technical support specialist. Answer technical questions."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    answer = response.choices[0].message.content
    print(f"[TECHNICAL] Answer provided")
    return {"response": answer}

def billing_node(state):
    """Answer billing questions"""
    question = state["question"]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a billing specialist. Answer billing questions."
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )
    
    answer = response.choices[0].message.content
    print(f"[BILLING] Answer provided")
    return {"response": answer}

def escalate_node(state):
    """Escalate to human"""
    customer_id = state["customer_id"]
    question = state["question"]
    
    ticket_id = f"TKT-{hash(customer_id + question) % 10000:04d}"
    response_text = f"""Thank you for contacting support. Your issue requires specialized attention.

Ticket ID: {ticket_id}
Our team will respond within 24 hours."""
    
    print(f"[ESCALATE] Ticket {ticket_id} created")
    return {"response": response_text, "ticket_id": ticket_id}

def save_node(state):
    """Save to memory"""
    customer_id = state["customer_id"]
    question = state["question"]
    response = state["response"]
    
    memory.add(
        messages=[f"Q: {question} | A: {response[:50]}..."],
        user_id=customer_id
    )
    
    print(f"[SAVE] Saved")
    return {"messages": [{"role": "assistant", "content": "Saved"}]}

def route_category(state) -> Literal["faq", "technical", "billing", "escalate"]:
    """Route based on category"""
    return state["category"]

graph = StateGraph(SupportState)

graph.add_node("classify", classify_node)
graph.add_node("faq", faq_node)
graph.add_node("technical", technical_node)
graph.add_node("billing", billing_node)
graph.add_node("escalate", escalate_node)
graph.add_node("save", save_node)

graph.add_edge(START, "classify")

graph.add_conditional_edges(
    "classify",
    route_category,
    {"faq": "faq", "technical": "technical", "billing": "billing", "escalate": "escalate"}
)

graph.add_edge("faq", "save")
graph.add_edge("technical", "save")
graph.add_edge("billing", "save")
graph.add_edge("escalate", "save")
graph.add_edge("save", END)

app = graph.compile(checkpointer=langgraph.checkpoint.memory.MemorySaver())

def support_chat(customer_id, question):
    """Main function"""
    print(f"\n[SUPPORT] Customer: {customer_id}")
    print(f"Question: {question}\n")
    
    result = app.invoke(
        {
            "customer_id": customer_id,
            "question": question,
            "category": "",
            "response": "",
            "ticket_id": "",
            "messages": []
        },
        config={"configurable": {"thread_id": f"customer_{customer_id}"}}
    )
    
    print(f"Response: {result['response']}\n")
    return result

if __name__ == "__main__":
    print("=== Test 1: FAQ ===")
    support_chat("john", "How do I reset my password?")
    
    print("=== Test 2: Technical ===")
    support_chat("sarah", "My code keeps crashing")
    
    print("=== Test 3: Billing ===")
    support_chat("mike", "What's the cost?")
    
    print("=== Test 4: Escalate ===")
    support_chat("alice", "I need urgent help")