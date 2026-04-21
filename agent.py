import json
import os
import uuid
from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict, Annotated, Optional, Literal
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile")

# ==========================================
# 1. TOOL EXECUTION
# ==========================================

def mock_lead_capture(name: str, email: str, platform: str):
    """Executes the lead capture via mock API."""
    print(f"\n[TOOL EXECUTION] Lead captured successfully: {name}, {email}, {platform}\n")

# ==========================================
# 2. INTENT DETECTION
# ==========================================

class IntentClassification(BaseModel):
    intent: Literal["Greeting", "RAG_Query", "Lead_Capture"] = Field(
        description="Classify the user intent into one of these three categories based on the conversation history."
    )

def classify_intent(state: "AgentState"):
    """Classifies the intent based on the conversation history."""
    classifier_llm = llm.with_structured_output(IntentClassification)
    system_msg = SystemMessage(content="You are an intent classifier for StreamGuide, the AI agent of AutoStream, a video editor company. Classify the intent of the user's latest interaction. Use 'Greeting' for casual hellos. Use 'RAG_Query' for product or pricing inquiries. Use 'Lead_Capture' for high-intent users ready to sign up (e.g., wanting the Pro plan for their channel).")
    history = state["messages"][-6:] # Keep the last few messages for context
    response = classifier_llm.invoke([system_msg] + history)
    intent = response.intent if response else "Greeting"
    return {"intent": intent}
    
def route_intent(state: "AgentState") -> str:
    """Routes based on the classified intent."""
    if state.get("intent") == "Lead_Capture":
        return "lead_collector"
    return "responder"

# ==========================================
# 3. RAG PIPELINE
# ==========================================

# Load knowledge.json
script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "knowledge.json"), "r") as f:
    knowledge_base = json.load(f)
knowledge_str = json.dumps(knowledge_base, indent=2)

def respond(state: "AgentState"):
    """Answers general questions or RAG queries using the loaded knowledge."""
    system_msg = SystemMessage(content=f"""You are an AI assistant named StreamGuide for 'AutoStream', an AI video editor.
You assist users in a friendly way. Here is the company knowledge base on pricing and policies:
{knowledge_str}

Answer the user's questions based on this knowledge. If they just say hi, greet them back enthusiastically.
Keep your answers concise and accurate.""")
    history = state["messages"][-12:] 
    response = llm.invoke([system_msg] + history)
    return {"messages": [response]}

# ==========================================
# 4. AGENT LOGIC
# ==========================================

def _is_valid(val: Optional[str]) -> bool:
    """Helper to catch LLM hallucinating 'Not Provided' instead of null"""
    if not val:
        return False
    if str(val).strip().lower() in ["none", "null", "not provided", "n/a", "unknown", "missing"]:
        return False
    return True

class AgentState(TypedDict):
    """The central state of the LangGraph agent."""
    messages: Annotated[list[AnyMessage], add_messages]
    intent: Optional[str]
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    lead_captured: bool

class LeadExtraction(BaseModel):
    name: Optional[str] = Field(None, description="User's name if mentioned. Return literally null if not present.")
    email: Optional[str] = Field(None, description="User's email if mentioned. Return literally null if not present.")
    platform: Optional[str] = Field(None, description="User's creator platform (e.g., YouTube, Instagram) if mentioned. Return literally null if not present.")

def welcome_prompt(state: AgentState):
    """Initiates the conversation asking the user for their intent."""
    msg = AIMessage(content="Hello! I am StreamGuide, your AutoStream assistant. How can I help you today? Are you just dropping by to say hello, do you have questions about our pricing, or are you ready to sign up?")
    return {"messages": [msg]}

def collect_lead(state: AgentState):
    """Collects name, email, and platform details sequentially until all are present."""
    history = state["messages"][-12:]
    
    current_name = state.get("name")
    current_email = state.get("email")
    current_platform = state.get("platform")
    
    extractor_llm = llm.with_structured_output(LeadExtraction)
    sys_instruction = SystemMessage(content="Extract the user's name, email, and creator platform from the current conversation. Only output values if you are reasonably confident they are provided.")
    extraction = extractor_llm.invoke([sys_instruction] + history)
    
    name = current_name or (extraction.name if extraction and _is_valid(extraction.name) else None)
    email = current_email or (extraction.email if extraction and _is_valid(extraction.email) else None)
    platform = current_platform or (extraction.platform if extraction and _is_valid(extraction.platform) else None)
    
    # Clean up empty strings back to None
    name = name if _is_valid(name) else None
    email = email if _is_valid(email) else None
    platform = platform if _is_valid(platform) else None
    
    updates = {"name": name, "email": email, "platform": platform}
    
    # Check if we have all slots filled
    if name and email and platform:
        if not state.get("lead_captured"):
            mock_lead_capture(name, email, platform)
            updates["lead_captured"] = True
            msg = AIMessage(content=f"Thanks, {name}! We've captured your details ({email}) for your {platform} channel. Someone from our team will reach out shortly to get you started on the Pro plan.")
            updates["messages"] = [msg]
        else:
             msg = AIMessage(content="We've already captured your lead info! Let me know if you need anything else.")
             updates["messages"] = [msg]
    else:
        missing = []
        if not name: missing.append("name")
        if not email: missing.append("email")
        if not platform: missing.append("creator platform (e.g., YouTube, Instagram)")
        
        missing_str = ", ".join(missing)
        prompt_instruction = f"The user wants to sign up, but we are missing some info. Nicely ask the user to provide their {missing_str} so we can get them signed up for the Pro plan. Limit your response strictly to making that request."
        
        collector_agent = llm.invoke([SystemMessage(content=prompt_instruction), history[-1]])
        updates["messages"] = [collector_agent]
        
    return updates

def route_start(state: AgentState) -> str:
    """Routes to welcome node if history is missing, otherwise to classifier."""
    if not state.get("messages"):
        return "welcome"
    return "classifier"

# Graph Construction
builder = StateGraph(AgentState)
builder.add_node("welcome_prompt", welcome_prompt)
builder.add_node("classifier", classify_intent)
builder.add_node("responder", respond)
builder.add_node("lead_collector", collect_lead)

builder.add_conditional_edges(START, route_start, {"welcome": "welcome_prompt", "classifier": "classifier"})
builder.add_edge("welcome_prompt", END)
builder.add_conditional_edges("classifier", route_intent, {"responder": "responder", "lead_collector": "lead_collector"})
builder.add_edge("responder", END)
builder.add_edge("lead_collector", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Main Testing Loop
if __name__ == "__main__":
    if "GROQ_API_KEY" not in os.environ:
        print("Please set the GROQ_API_KEY environment variable to run the test script.")
        exit(1)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    print("\n" + "="*50)
    print("--- Starting StreamGuide Agent Workflow ---")
    print("="*50 + "\n")
    
    # 1. Trigger the agent's welcome prompt implicitly by sending an empty state
    result = graph.invoke({"messages": []}, config)
    print(f"StreamGuide: {result['messages'][-1].content}\n")

    print("-" * 50)
    print("Simulation complete! You can now test the agent manually.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            user_input = input("User: ")
        except (EOFError, KeyboardInterrupt):
            break
            
        if user_input.lower() in ["quit", "exit"]:
            break
            
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]}, config)
        print(f"StreamGuide: {result['messages'][-1].content}\n")
