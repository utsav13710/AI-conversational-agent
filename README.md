# StreamGuide Agent - AutoStream

StreamGuide is a conversational, agentic workflow built for "AutoStream" (an AI video editor company). It handles dynamic intent classification, knowledge retrieval (RAG) for product pricing, and strict stateful lead capture.

---

## 1. How to run the project locally

Follow these steps to launch both the LangGraph Agent and its high-end glassmorphic web interface:

1. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

2. **Install Dependencies**
   Install all required libraries using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Local Web Server**
   Boot up the FastAPI server, which dynamically links the Python graph to the frontend:
   ```bash
   python -m uvicorn server:app --reload
   ```

4. **Access the Chat Client**
   Open your browser and navigate to exactly: **[http://localhost:8000](http://localhost:8000)**

---

## 2. Architecture Explanation

For this Social-to-Lead Agentic Workflow, **LangGraph** was chosen over frameworks like *AutoGen* for multiple distinct reasons. AutoGen excels at allowing multiple LLMs autonomously converse with each other in an open-ended swarm, but it lacks strict determinism. Since AutoStream requires rigid guardrails (e.g., *never* triggering the API tool until exact criteria are met), LangGraph's explicit Node and Edge architecture empowers us to physically draw boundaries around the conversation logic.

**State Management:**
At the core of the implementation is the `AgentState` object. It acts as a single, immutable container passed down the pipeline carrying the conversational history (`messages`), the inferred `intent`, and our tracking properties (`name`, `email`, `platform`, `lead_captured`). 
State continuity is securely handled through LangGraph's `MemorySaver()`. When the frontend issues a `POST` request, it passes a unique `thread_id`; LangGraph invokes its Checkpointer, extracts the last known state dictionary for that session, appends the newest Human message, and routes the LLM to either the `responder` or `lead_collector` nodes without wiping history!

---

## 3. WhatsApp Deployment Strategy

To deploy this local REST architecture onto a social messaging platform like WhatsApp, we would build a robust **Webhook integration bridge** (typically leveraging the official Meta WhatsApp Cloud API or Twilio):

1. **Expose the existing Server via Webhook:**
   Instead of just the web UI accessing `POST /chat`, we would expose a public SSL endpoint (e.g., via Ngrok for testing, or AWS/Vercel for production) that WhatsApp can POST payload events to whenever AutoStream receives a new text.

2. **Map the Identifiers:**
   WhatsApp's Webhook JSON includes a `From` number (the user's phone). Instead of generating a random browser `UUID`, we would pass that unique phone number directly into LangGraph’s `thread_id`! This natively gives WhatsApp users permanent memory persistence across days of chatting.

3. **Process and Return:**
   Inside the newly crafted `/webhook` route, your backend would wait for `graph.invoke()` to complete processing the LangGraph nodes, parse out the final `last_msg.content`, and fire an outbound HTTP request directly back to the WhatsApp API to deliver the text bubble straight to the user's phone.
