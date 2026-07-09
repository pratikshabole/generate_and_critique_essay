
import streamlit as st
import os
from typing import Dict, Any, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq

# Set up page styling
st.set_page_config(page_title="✍️ Agentic Essay Writer", layout="wide")
st.title("✍️ Agentic Essay Writer & Self-Critique Agent")
st.write("This agent generates an initial draft, critiques its own work, and refines it dynamically.")

# ---------------------------------------------------------------------
# 1. State & Agent Definition
# ---------------------------------------------------------------------
class AgentState(TypedDict):
    topic: str
    draft: str
    critique: str
    revision_count: int
    final_essay: str

def get_llm():
    # Pulls API key from environment
    return ChatGroq(temperature=0, model_name='llama-3.3-70b-versatile')

def generate_initial_draft(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    messages = [
        SystemMessage(content="You are an expert academic essayist. Write a concise, compelling 3-paragraph essay on the given topic."),
        HumanMessage(content=f"Write an essay about: {state['topic']}")
    ]
    response = llm.invoke(messages)
    return {"draft": response.content, "revision_count": 1}

def critique_draft(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    messages = [
        SystemMessage(content=(
            "You are a strict, constructive writing editor. Review the essay draft. "
            "Identify structural gaps, clarity improvements, or logical flaws. "
            "If the essay is exceptional and needs no changes, write 'APPROVED'."
        )),
        HumanMessage(content=f"Review this essay draft:\n\n{state['draft']}")
    ]
    response = llm.invoke(messages)
    return {"critique": response.content}

def revise_draft(state: AgentState) -> Dict[str, Any]:
    llm = get_llm()
    messages = [
        SystemMessage(content="You are an adaptive writer. Rewrite the essay draft incorporating all the constructive critique points provided."),
        HumanMessage(content=f"Original Draft:\n{state['draft']}\n\nCritique:\n{state['critique']}")
    ]
    response = llm.invoke(messages)
    return {
        "draft": response.content, 
        "revision_count": state["revision_count"] + 1
    }

# ---------------------------------------------------------------------
# 2. Control Loop Execution
# ---------------------------------------------------------------------
def run_essay_agent(topic: str, max_iterations: int = 3):
    # Initialize state
    state: AgentState = {
        "topic": topic,
        "draft": "",
        "critique": "",
        "revision_count": 0,
        "final_essay": ""
    }
    
    status_box = st.status("🚀 Initializing Agent Loop...", expanded=True)
    
    # Run loop
    while state["revision_count"] < max_iterations:
        current_rev = state["revision_count"] + 1
        
        status_box.write(f"📝 Generating / Modifying Draft (Pass {current_rev})...")
        if state["revision_count"] == 0:
            draft_res = generate_initial_draft(state)
        else:
            draft_res = revise_draft(state)
        state.update(draft_res)
        
        status_box.write(f"🔍 Executing self-critique protocol on Draft {current_rev}...")
        critique_res = critique_draft(state)
        state.update(critique_res)
        
        # Break out early if approved
        if "APPROVED" in state["critique"].upper():
            status_box.write("✅ Critique approved the text structural qualities!")
            break
            
    state["final_essay"] = state["draft"]
    status_box.update(label="✨ Agent Pipeline Complete!", state="complete")
    return state

# ---------------------------------------------------------------------
# 3. Streamlit Interface UI Layout
# ---------------------------------------------------------------------
# Sidebar configuration
with st.sidebar:
    st.header("🔑 Authentication & Control")
    api_key = st.text_input("Grok API Key", type="password")
    max_loops = st.slider("Max Revision Cycles", min_value=1, max_value=4, value=2)

# Main container
topic_input = st.text_input("Enter your essay topic or prompt:", value="The socio-economic impacts of space exploration")

if st.button("Generate & Critique Essay"):
    if not api_key:
        st.error("Please provide your Groq API key in the sidebar to proceed.")
    elif not topic_input.strip():
        st.warning("Please input a valid topic.")
    else:
        # Save key to environment variable securely inside execution context
        os.environ["GROQ_API_KEY"] = api_key
        
        # Run agent loop execution
        final_state = run_essay_agent(topic_input, max_iterations=max_loops)
        
        # Separate output displays elegantly inside layouts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📚 Polished Final Essay")
            st.info(f"Total Revision Cycles Completed: {final_state['revision_count']}")
            st.write(final_state["final_essay"])
            
        with col2:
            st.subheader("📋 Last Critique Feedback Report")
            st.markdown(f"```\n{final_state['critique']}\n```")
