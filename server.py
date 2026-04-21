from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Ensure dotEnv is loaded before anything else
from dotenv import load_dotenv
load_dotenv()

if "GROQ_API_KEY" not in os.environ:
    print("GROQ_API_KEY environment variable is required to run the server.")
    sys.exit(1)

# Import the pre-configured langgraph agent from agent.py
from agent import graph
from langchain_core.messages import HumanMessage

app = FastAPI(title="StreamGuide Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Host static files properly
script_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(script_dir, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

class ChatRequest(BaseModel):
    message: str
    thread_id: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Enforce session memory
    config = {"configurable": {"thread_id": req.thread_id}}
    
    # Empty triggers invoke the welcome loop gracefully on AgentState
    body = {"messages": []}
    if req.message.strip():
        body = {"messages": [HumanMessage(content=req.message)]}
    
    # Run the graph
    try:
        result = graph.invoke(body, config)
        last_msg = result["messages"][-1].content
        return {"response": last_msg}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
