__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain import hub
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import List, Literal, Annotated, Optional, Tuple, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ConfigDict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import time
import re
import tempfile
import hashlib

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# Extended state class with chat history and agent-specific states
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    chat_history: List[BaseMessage] = Field(default_factory=list)
    reformulation_count: int = 0
    current_query: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list) 
    generated_answer: Optional[str] = None
    next_step: Optional[str] = None
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Initialize components with configurable parameters
def initialize_system(urls: List[str], uploaded_files: List[Any], 
                     chunk_size: int = 250, k: int = 3, temperature: float = 0.0):
    # Load and process documents
    docs = []
    
    # Load from URLs
    for url in urls:
        try:
            docs.extend(WebBaseLoader(url).load())
        except Exception as e:
            st.error(f"Failed to load {url}: {str(e)}")
    
    # Load from uploaded files
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name

            if os.path.getsize(temp_file_path) == 0:
                st.error(f"Uploaded file {uploaded_file.name} is empty and was skipped.")
                os.unlink(temp_file_path)
                continue

            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == ".txt":
                loader = TextLoader(temp_file_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            else:
                st.error(f"Unsupported file type: {uploaded_file.name}")
                os.unlink(temp_file_path)
                continue

            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata["source"] = uploaded_file.name
            docs.extend(file_docs)
            os.unlink(temp_file_path)  # Clean up temp file
        except Exception as e:
            st.error(f"Failed to load uploaded file {uploaded_file.name}: {str(e)}")
    
    if not docs:
        st.warning("No documents loaded. Using default knowledge sources.")
        default_urls = [
            "https://medium.com/@sridevi.gogusetty/rag-vs-graph-rag-llama-3-1-8f2717c554e6",
            "https://medium.com/@sridevi.gogusetty/retrieval-augmented-generation-rag-gemini-pro-pinecone-1a0a1bfc0534",
            "https://medium.com/@sridevi.gogusetty/introduction-to-ollama-run-llm-locally-data-privacy-f7e4e58b37a0",
            "https://ollama.com/library",
            "https://ai.google.dev/docs/gemini_api_overview",
        ]
        for url in default_urls:
            try:
                docs.extend(WebBaseLoader(url).load())
            except Exception as e:
                st.error(f"Failed to load default URL {url}: {str(e)}")
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs)
    
    # Create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", 
                                             google_api_key=GOOGLE_API_KEY)
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    # The raw retriever instance
    retriever_instance = vectorstore.as_retriever(search_kwargs={"k": k}) 
    
    # The retriever_tool that when invoked, returns a *string* of concatenated content
    retriever_tool_for_display = create_retriever_tool(
        retriever_instance,
        "retrieve_knowledge",
        "Search and return information from the provided knowledge sources.",
    )
    
    search = TavilySearchAPIWrapper()
    web_search_tool = TavilySearchResults(
        api_wrapper=search, 
        max_results=5,
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    )
    
    # Build enhanced LangGraph workflow
    workflow = StateGraph(AgentState)
    
    # Add specialized agent nodes
    workflow.add_node("router", router_agent)
    workflow.add_node("retrieve", retrieve_agent)
    workflow.add_node("reformulate_query", reformulate_agent)
    workflow.add_node("web_search", web_search_agent)
    workflow.add_node("synthesize", synthesize_agent)
    workflow.add_node("generate", generate_agent)
    workflow.add_node("fact_check", fact_check_agent)
    workflow.add_node("safety_check", safety_agent)
    workflow.add_node("ask_clarification", ask_clarification)
    
    # Add edges and conditional routing
    workflow.add_edge(START, "router")
    
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "reformulate": "reformulate_query",
            "web_search": "web_search",
            "clarify": "ask_clarification",
            "generate": "generate"
        }
    )
    
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "relevant": "synthesize",
            "reformulate": "reformulate_query",
            "web_search": "web_search",
            "clarify": "ask_clarification"
        }
    )
    
    workflow.add_conditional_edges(
        "reformulate_query",
        should_retry_retrieval,
        {
            "retrieve": "retrieve",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "synthesize")
    workflow.add_edge("synthesize", "generate")
    workflow.add_edge("generate", "fact_check")
    workflow.add_edge("fact_check", "safety_check")
    workflow.add_edge("safety_check", END)
    workflow.add_edge("ask_clarification", END)
    
    # Return the raw retriever_instance for use in retrieve_agent
    return workflow.compile(), retriever_instance, web_search_tool, temperature, retriever_tool_for_display

# ========================
# SPECIALIZED AGENT FUNCTIONS
# ========================

def router_agent(state: AgentState) -> dict:
    """Decide initial workflow path based on query and history"""
    st.session_state.logs.append("---ROUTER AGENT---")
    
    model = ChatGroq(
        temperature=st.session_state.temperature, 
        model_name="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY
    )
    
    prompt = PromptTemplate(
        template="""As the Router Agent, analyze the user's question and conversation history to determine the best next step.
        
        Conversation History:
        {history}
        
        Current Question: {question}
        
        Choose one of these actions:
        - "retrieve": If question can be answered with known documents (from internal knowledge base)
        - "reformulate": If query needs refinement for better retrieval (internal knowledge base)
        - "web_search": If question requires current/realtime information or broader knowledge beyond internal docs
        - "clarify": If question is ambiguous or needs more details from the user
        - "generate": If question is simple, conversational, or a direct follow-up easily answerable without external tools
        
        Only respond with the action word.""",
        input_variables=["question", "history"]
    )
    
    history_str = "\n".join([f"{m.type}: {m.content}" for m in state.chat_history[-5:]])
    response = model.invoke(prompt.format(question=state.current_query, history=history_str))
    decision = response.content.strip().lower()
    
    st.session_state.logs.append(f"Routing decision: {decision}")
    return {"next_step": decision}

def retrieve_agent(state: AgentState) -> dict:
    """Retrieve relevant documents from vector store"""
    st.session_state.logs.append("---RETRIEVAL AGENT---")
    query = state.current_query
    
    try:
        # Use the raw retriever_instance to get actual Document objects
        docs_list_objects = st.session_state.retriever_instance.get_relevant_documents(query)
        
        # Convert Document objects to a list of dicts for Pydantic AgentState compatibility
        retrieved_content_with_meta = []
        for doc in docs_list_objects:
            retrieved_content_with_meta.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })

        st.session_state.logs.append(f"Retrieved {len(retrieved_content_with_meta)} documents")
        return {"retrieved_docs": retrieved_content_with_meta}
    except Exception as e:
        st.session_state.logs.append(f"Retrieval error: {str(e)}")
        return {"retrieved_docs": []} # Ensure it's an empty list on error

def reformulate_agent(state: AgentState) -> dict:
    """Reformulate query for better retrieval"""
    st.session_state.logs.append("---QUERY REFORMULATION AGENT---")
    
    model = ChatGroq(
        temperature=st.session_state.temperature + 0.2, # Slightly higher temperature for creativity
        model_name="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY
    )
    
    prompt = PromptTemplate(
        template="""Reformulate the user's query to improve document retrieval. Consider:
        - Synonyms and related terms
        - Breaking down complex questions
        - Context from conversation history
        
        Original Query: {query}
        Conversation History: {history}
        Previous Retrieval Results (if any, truncated): {previous_results}
        
        Respond ONLY with the reformulated query.""",
        input_variables=["query", "history", "previous_results"]
    )
    
    history_str = "\n".join([f"{m.type}: {m.content}" for m in state.chat_history[-3:]])
    # Extract content from retrieved_docs for previous_results display
    previous_results_content = "\n\n".join([doc["content"] for doc in state.retrieved_docs]) if state.retrieved_docs else ""
    
    response = model.invoke(prompt.format(
        query=state.current_query,
        history=history_str,
        previous_results=previous_results_content[:500] if previous_results_content else "None" # Truncate for prompt
    ))
    
    new_query = response.content.strip()
    st.session_state.logs.append(f"Reformulated query: {new_query}")
    return {"current_query": new_query, "reformulation_count": state.reformulation_count + 1}

def web_search_agent(state: AgentState) -> dict:
    """Perform web search using Tavily"""
    st.session_state.logs.append("---WEB SEARCH AGENT---")
    query = state.current_query
    
    try:
        results = st.session_state.web_search_tool.invoke({"query": query})
        # Store results as a list of dicts with content and metadata
        web_results_with_meta = []
        for d in results:
            web_results_with_meta.append({
                "content": d["content"],
                "metadata": d.get("metadata", {"source": "web_search"}) # Add a default source
            })
        st.session_state.logs.append(f"Found {len(web_results_with_meta)} web results")
        return {"retrieved_docs": web_results_with_meta}
    except Exception as e:
        st.session_state.logs.append(f"Web search error: {str(e)}")
        return {"retrieved_docs": []} # Ensure it's an empty list on error

def synthesize_agent(state: AgentState) -> dict:
    """Synthesize information from multiple sources"""
    st.session_state.logs.append("---SYNTHESIS AGENT---")
    
    if not state.retrieved_docs:
        st.session_state.logs.append("No documents for synthesis.")
        return {"retrieved_docs": []} # Return empty list if no docs

    model = ChatGroq(
        temperature=st.session_state.temperature, 
        model_name="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY
    )
    
    prompt = PromptTemplate(
        template="""Synthesize key information from these knowledge sources:
        
        {sources}
        
        Focus on answering: {question}
        
        Extract and combine the most relevant facts. Omit irrelevant details.
        Respond with a concise knowledge summary. Include source attribution where possible.
        """,
        input_variables=["sources", "question"]
    )
    
    # Extract content from the list of dicts for the sources prompt
    sources_text = []
    for i, doc in enumerate(state.retrieved_docs):
        source_info = doc["metadata"].get("source", f"Document {i+1}")
        sources_text.append(f"Source {i+1} ({source_info}):\n{doc['content']}")
    sources_content_for_llm = "\n\n".join(sources_text)

    response = model.invoke(prompt.format(
        sources=sources_content_for_llm, 
        question=state.current_query
    ))
    
    synthesized = response.content
    st.session_state.logs.append(f"Synthesized {len(synthesized.split())} word summary")
    # Store synthesized content as a single document-like dict for generation.
    return {"retrieved_docs": [{"content": synthesized, "metadata": {"source": "synthesized_knowledge"}}]}

def generate_agent(state: AgentState) -> dict:
    """Generate final answer from context"""
    st.session_state.logs.append("---GENERATION AGENT---")
    
    if not state.retrieved_docs:
        st.session_state.logs.append("No context available for generation.")
        return {"generated_answer": "I don't have enough information to answer that question."}

    model = ChatGroq(
        temperature=st.session_state.temperature, 
        model_name="llama3-70b-8192",
        groq_api_key=GROQ_API_KEY
    )
    
    prompt = hub.pull("rlm/rag-prompt") # This prompt expects 'context' and 'question'
    rag_chain = prompt | model | StrOutputParser()
    
    # Extract content from the list of dicts for the context
    context_content = "\n\n".join([doc["content"] for doc in state.retrieved_docs])
    
    response = rag_chain.invoke({
        "context": context_content,
        "question": state.current_query
    })
    
    st.session_state.logs.append("Response generated")
    return {"generated_answer": response}

def fact_check_agent(state: AgentState) -> dict:
    """Fact-check generated answer using web search"""
    st.session_state.logs.append("---FACT-CHECK AGENT---")
    
    # Ensure generated_answer is not None before proceeding
    if not state.generated_answer:
        st.session_state.logs.append("No generated answer to fact-check.")
        return {"generated_answer": state.generated_answer or ""}
        
    model = ChatGroq(
        temperature=0, 
        model_name="llama3-70b-8192", # Use a more powerful model for fact-checking
        groq_api_key=GROQ_API_KEY
    )
    
    claims_prompt = """Identify factual claims in this text. List them as bullet points:
    
    {text}
    
    Respond ONLY with the bullet points of claims. Each claim should start with a hyphen '- '."""
    
    claims_response = model.invoke(claims_prompt.format(text=state.generated_answer)).content
    # Filter out empty lines or lines not starting with a bullet point
    claims_list = [c.strip() for c in claims_response.split("\n") if c.strip() and c.lstrip().startswith('-')] 
    st.session_state.logs.append(f"Identified {len(claims_list)} claims to verify")
    
    # Verify each claim with web search
    verified_claims = []
    # Limit to top 3 for efficiency, and ensure there are claims to check
    for claim in claims_list[:3]:
        try:
            # Remove leading hyphen if present before searching
            search_query = claim[2:] if claim.startswith('- ') else claim
            results = st.session_state.web_search_tool.invoke({"query": f"Verify: {search_query}"})
            sources = [d["content"] for d in results][:2]
            
            verification_prompt = f"""Based on these sources, is this claim true?
            Claim: {claim}
            
            Sources:
            {sources}
            
            Respond with "TRUE" or "FALSE" and a brief explanation."""
            
            verdict = model.invoke(verification_prompt).content
            verified_claims.append(f"{claim} → {verdict}")
        except Exception as e:
            st.session_state.logs.append(f"Fact-check for '{claim}' failed: {str(e)}")
            verified_claims.append(f"{claim} → Verification failed (Error: {str(e)})")
    
    # Update answer with verification notes
    if verified_claims:
        verified_text = "\n".join(verified_claims)
        updated_answer = f"{state.generated_answer}\n\n**Fact Check Results:**\n{verified_text}"
    else:
        updated_answer = state.generated_answer + "\n\n**Fact Check:** No specific claims identified or verified."
        
    st.session_state.logs.append(f"Verified {len(verified_claims)} claims")
    return {"generated_answer": updated_answer}

def safety_agent(state: AgentState) -> dict:
    """Check for harmful/inappropriate content"""
    st.session_state.logs.append("---SAFETY AGENT---")
    
    if not state.generated_answer:
        st.session_state.logs.append("No generated answer to safety check.")
        return {"generated_answer": state.generated_answer or ""}

    model = ChatGroq(
        temperature=0, 
        model_name="llama3-70b-8192", 
        groq_api_key=GROQ_API_KEY
    )
    
    prompt = PromptTemplate(
        template="""Analyze this text for harmful, biased, or inappropriate content:
        
        {text}
        
        Respond in this exact format:
        Safety Rating: [SAFE/CONCERN/UNSAFE]
        Issues: [List any issues found, or 'None' if safe]
        Revised Text: [If issues found and revisable, provide a revised version. Otherwise, state 'Not revisable'.]
        
        Example for unsafe:
        Safety Rating: UNSAFE
        Issues: Contains hate speech
        Revised Text: Not revisable
        
        Example for safe:
        Safety Rating: SAFE
        Issues: None
        Revised Text: N/A""",
        input_variables=["text"]
    )
    
    response_content = model.invoke(prompt.format(text=state.generated_answer)).content
    
    safety_rating_match = re.search(r"Safety Rating:\s*\[?(SAFE|CONCERN|UNSAFE)\]?", response_content, re.IGNORECASE)
    safety_rating = safety_rating_match.group(1).upper() if safety_rating_match else "UNKNOWN"

    # Use re.DOTALL to match across multiple lines for Revised Text
    revised_text_match = re.search(r"Revised Text:\s*(.*?)(?=\n[A-Z]|$)", response_content, re.DOTALL | re.IGNORECASE)
    revised_text = revised_text_match.group(1).strip() if revised_text_match else "N/A"

    final_answer_after_safety = state.generated_answer # Default to original

    if safety_rating == "SAFE":
        final_answer_after_safety = state.generated_answer
    elif "Not revisable" in revised_text:
        final_answer_after_safety = "I cannot answer that question due to safety concerns."
    elif revised_text and revised_text != "N/A": 
        final_answer_after_safety = revised_text
    
    st.session_state.logs.append(f"Safety rating: {safety_rating}")
    return {"generated_answer": final_answer_after_safety}

def ask_clarification(state: AgentState) -> dict:
    """Ask user for clarification"""
    st.session_state.logs.append("---ASKING FOR CLARIFICATION---")
    
    clarification = "I need a bit more clarity to answer your question effectively. Could you please rephrase or provide more details?"
    return {"generated_answer": clarification}

# ========================
# DECISION FUNCTIONS
# ========================

def route_decision(state: AgentState) -> str:
    """Determine next step after routing"""
    # This function simply returns the decision made by the router_agent
    return state.next_step

def grade_documents(state: AgentState) -> Literal["relevant", "reformulate", "web_search", "clarify"]:
    """Evaluate document relevance and suggest next action"""
    st.session_state.logs.append("---GRADE DOCUMENTS---")
    
    # If no documents were retrieved at all, they are irrelevant
    if not state.retrieved_docs:
        st.session_state.logs.append("No documents retrieved, defaulting action to 'web_search'.")
        return "web_search" # Fallback if initial retrieval yields nothing

    model = ChatGroq(
        temperature=0, 
        model_name="gemma2-9b-it", # Lighter model for quick classification
        groq_api_key=GROQ_API_KEY
    )
    
    class Grade(BaseModel):
        score: str = Field(description="Relevance: 'relevant', 'partial', 'irrelevant'")
        action: str = Field(description="Next step: 'relevant', 'reformulate', 'web_search', 'clarify'")
    
    llm = model.with_structured_output(Grade)
    
    prompt = PromptTemplate(
        template="""Evaluate retrieved documents for question relevance:
        
        Question: {question}
        Documents: {context}
        
        Score options:
        - relevant: Directly answers question
        - partial: Partially relevant but incomplete, may need more info or different sources
        - irrelevant: Not relevant at all
        
        Action options (based on score and need for more info):
        - synthesize: Use documents as-is to synthesize an answer
        - reformulate: Improve query and retry internal retrieval (if partial/irrelevant but internal knowledge might exist)
        - web_search: Use web search instead (if irrelevant or query is external-facing)
        - clarify: Ask user for clarification (if question is ambiguous)
        
        Respond with ONLY the JSON object conforming to the Grade schema, like {{"score": "relevant", "action": "synthesize"}}.
        """,
        input_variables=["question", "context"]
    )
    
    # Extract content from the list of dicts for the context
    context_content = "\n\n".join([doc["content"] for doc in state.retrieved_docs])
    
    try:
        result = llm.invoke(prompt.format(
            question=state.current_query,
            context=context_content[:4000] # Truncate long context string for LLM, adjust as needed
        ))
        
        st.session_state.logs.append(f"Relevance: {result.score} → Action: {result.action}")
        # Map 'synthesize' to 'relevant' since that's what the router expects
        if result.action == "synthesize":
            return "relevant"
        return result.action
    except Exception as e:
        st.session_state.logs.append(f"Grading error: {str(e)}. Defaulting to 'web_search'.")
        return "web_search" # Fallback if grading fails

def should_retry_retrieval(state: AgentState) -> Literal["retrieve", "web_search"]:
    """Decide whether to retry internal retrieval after reformulation or go to web search."""
    # This limits the number of query reformulations before falling back to web search
    if state.reformulation_count < 2: # Allow up to 2 reformulations
        st.session_state.logs.append(f"Reformulation count: {state.reformulation_count}. Retrying retrieval.")
        return "retrieve"
    st.session_state.logs.append(f"Reformulation limit reached ({state.reformulation_count}). Falling back to web search.")
    return "web_search"

# ========================
# STREAMLIT APP
# ========================
st.set_page_config(page_title="Multi-Agent RAG System", layout="wide")
st.title("🤖 Advanced Agentic RAG System")
st.caption("Multi-agent collaboration with self-correction and memory")

# Recommendation for better results 
st.info(
    "🔎 **Recommendation:** For the most accurate and relevant answers, please explicitly add your own URLs or upload documents as knowledge sources. "
    "Relying only on default sources may yield less tailored results."
)
with st.sidebar:
    st.markdown(f"""
    **Current Settings:**  
    - Chunk Size: `{st.session_state.get('chunk_size', 250)}`
    - Retriever K: `{st.session_state.get('retriever_k', 3)}`
    - Temperature: `{st.session_state.get('temperature', 0.0)}`
    """)
    with st.expander("⚙️ Adjust Configuration", expanded=False):
        chunk_size = st.slider("Text Chunk Size", 100, 1000, st.session_state.get('chunk_size', 250), 50)
        retriever_k = st.slider("Retriever K Value (Top K Docs)", 1, 10, st.session_state.get('retriever_k', 3))
        temperature = st.slider("LLM Temperature", 0.0, 1.0, st.session_state.get('temperature', 0.0), 0.1)
        st.divider()
    st.subheader("Knowledge Sources")
    st.write("Add URLs (one per line):")
    url_input = st.text_area(
        "Enter URLs",
        height=150,
        value="",
        label_visibility="collapsed"
    )
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True
    )
    reset_params = st.button("Apply Parameters & Update Knowledge")
    st.divider()
    st.subheader("Agent Roles")
    st.markdown("""
    - **Router**: Determines workflow path based on query type.
    - **Retriever**: Fetches relevant documents from internal knowledge base.
    - **Reformulator**: Improves queries for better internal retrieval.
    - **Web Searcher**: Finds real-time, external information.
    - **Synthesizer**: Combines information from various sources.
    - **Generator**: Creates the final answer.
    - **Fact Checker**: Verifies factual claims in the generated answer.
    - **Safety Checker**: Ensures generated content is safe and appropriate.
    - **Clarification**: Asks user for more details if needed.
    """)

# Initialize session state
if "logs" not in st.session_state:
    st.session_state.logs = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "final_answer" not in st.session_state:
    st.session_state.final_answer = ""
if "params_applied" not in st.session_state:
    st.session_state.params_applied = False
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.0
if "knowledge_hash" not in st.session_state:
    st.session_state.knowledge_hash = ""

# Calculate current knowledge hash
def calculate_knowledge_hash(urls, files):
    content = "|".join(sorted(urls))
    for file in files:
        content += file.getvalue().decode(errors="ignore")
    return hashlib.md5(content.encode()).hexdigest()

# Parse URLs from input
urls = [url.strip() for url in url_input.split('\n') if url.strip()]

# Check if knowledge has changed
current_knowledge_hash = calculate_knowledge_hash(urls, uploaded_files)
knowledge_changed = current_knowledge_hash != st.session_state.knowledge_hash

# Initialize or update system
if reset_params or not st.session_state.params_applied or knowledge_changed:
    with st.spinner("Configuring system with new parameters and knowledge sources..."):
        try:
            st.session_state.graph, st.session_state.retriever_instance, st.session_state.web_search_tool, st.session_state.temperature, st.session_state.retriever_tool_for_display = initialize_system(
                urls=urls,
                uploaded_files=uploaded_files,
                chunk_size=chunk_size,
                k=retriever_k,
                temperature=temperature
            )
            st.session_state.params_applied = True
            st.session_state.knowledge_hash = current_knowledge_hash
            st.success("System configured with new knowledge!")
        except Exception as e:
            st.error(f"Configuration failed: {str(e)}")
            st.stop()

# Chat interface
with st.container():
    st.subheader("Chat")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            with st.chat_message("user"):
                st.write(msg.content)
        elif isinstance(msg, AIMessage):
            with st.chat_message("assistant"):
                st.write(msg.content)
    
    if prompt := st.chat_input("Ask about your knowledge sources..."):
        # Add user message to history
        user_msg = HumanMessage(content=prompt)
        st.session_state.chat_history.append(user_msg)
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Prepare agent state
        agent_state = AgentState(
            messages=[user_msg],
            chat_history=st.session_state.chat_history,
            current_query=prompt,
            reformulation_count=0,
            retrieved_docs=[]
        )
        
        # Execute graph
        with st.spinner("Executing multi-agent workflow..."):
            st.session_state.logs = [f"New query: {prompt}"]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                step_count = 0
                max_steps = 15
                current_state = agent_state
                
                # Stream outputs from the graph
                for output in st.session_state.graph.stream(agent_state):
                    node_name = list(output.keys())[0]
                    node_state = output[node_name]
                    
                    status_text.info(f"Executing: **{node_name.replace('_', ' ').title()}**")
                    st.session_state.logs.append(f"Completed node: {node_name}")
                    
                    step_count += 1
                    progress_bar.progress(min(step_count / max_steps, 1.0))
                    time.sleep(0.3)
                    
                    # Store the latest state updates
                    current_state_updates = node_state
                    
                    if step_count >= max_steps:
                        st.session_state.logs.append("⚠️ Safety break: Exceeded maximum steps. Ending workflow.")
                        break
                
                progress_bar.empty()
                status_text.success("✅ Workflow completed!")
                
                # Process final output
                if current_state_updates and 'generated_answer' in current_state_updates:
                    final_answer = current_state_updates['generated_answer']
                    ai_msg = AIMessage(content=final_answer)
                    st.session_state.chat_history.append(ai_msg)
                    st.session_state.final_answer = final_answer
                    
                    with st.chat_message("assistant"):
                        st.write(final_answer)
                else:
                    fallback_message = "I couldn't generate a complete response for your query. Please try rephrasing."
                    ai_msg = AIMessage(content=fallback_message)
                    st.session_state.chat_history.append(ai_msg)
                    st.session_state.final_answer = fallback_message
                    
                    with st.chat_message("assistant"):
                        st.write(fallback_message)

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                status_text.error(f"❌ Execution failed: {str(e)}")
                st.session_state.logs.append(f"ERROR TRACEBACK:\n{error_trace}")
                error_msg = AIMessage(content="Sorry, I encountered an error processing your request. Please check the logs.")
                st.session_state.chat_history.append(error_msg)
                
                with st.chat_message("assistant"):
                    st.write("Sorry, I encountered an error processing your request. Please check the logs.")

# Display results
with st.expander("Execution Details", expanded=False):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Workflow Log")
        log_container = st.container(height=300)
        for log in st.session_state.logs:
            log_container.code(log, language="log")
        
        st.subheader("Active Knowledge Sources")
        if urls or uploaded_files:
            st.markdown("**URLs:**")
            for url in urls:
                st.markdown(f"- [{url}]({url})")
            
            if uploaded_files:
                st.markdown("**Uploaded Files:**")
                for file in uploaded_files:
                    st.markdown(f"- {file.name}")
        else:
            st.markdown("Using default knowledge sources")
    
    with col2:
        st.subheader("Workflow Diagram")
        st.graphviz_chart("""
            digraph {
                node [shape=box, style=rounded]
                start -> router
                router -> retrieve [label="retrieve"]
                router -> reformulate_query [label="reformulate"]
                router -> web_search [label="web_search"]
                router -> ask_clarification [label="clarify"]
                router -> generate [label="generate (simple query)"]
                
                retrieve -> grade_documents [label="docs retrieved"]
                
                grade_documents -> synthesize [label="relevant"]
                grade_documents -> reformulate_query [label="reformulate"]
                grade_documents -> web_search [label="irrelevant / no docs"]
                grade_documents -> ask_clarification [label="clarify"]
                
                reformulate_query -> retrieve [label="retry retrieval (max 2)"]
                reformulate_query -> web_search [label="fallback to web search"]
                
                web_search -> synthesize [label="results found"]
                
                synthesize -> generate [label="summary created"]
                generate -> fact_check [label="answer created"]
                fact_check -> safety_check [label="claims checked"]
                safety_check -> end [label="content safe"]
                
                ask_clarification -> end [label="user asked to clarify"]
            }
        """)
        
        st.subheader("Agent Configuration")
        st.markdown(f"""
        - **Chunk Size**: `{chunk_size}`
        - **Retriever K (Top K Docs)**: `{retriever_k}`
        - **LLM Temperature**: `{temperature}`
        - **Main LLM Model**: `llama3-70b-8192` (Groq)
        - **Grading LLM Model**: `gemma2-9b-it` (Groq)
        """)

# Add reset button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.logs = []
    st.session_state.final_answer = ""
    st.session_state.get("knowledge_hash", None)  # Clear knowledge hash
    st.session_state.params_applied = False
    st.rerun()

st.divider()
st.caption("Advanced Agentic RAG System | Multi-agent collaboration with self-correction")