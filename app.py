import os
import streamlit as st  # The library that turns this Python script into a website/UI
from typing import TypedDict, Annotated  # Imports tools for defining typed state dictionaries and adding extra information to types
from langgraph.graph.message import add_messages  # Imports a function to add new messages to an existing message list
from langgraph.graph import StateGraph, START, END  # Imports tools to create the graph and define its starting and ending points
from langchain_groq import ChatGroq  # Imports the Groq chat model for using an LLM
from langchain_community.document_loaders import PyPDFLoader  # Imports a loader to read and extract text from PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Imports a tool to split large text into smaller chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Imports Hugging Face embeddings to convert text into numerical vectors
from langchain_community.vectorstores import FAISS  # Imports FAISS for storing and searching text embeddings efficiently
from dotenv import load_dotenv

# st.set_page_config MUST be the very first Streamlit command in the file
st.set_page_config(
    page_title="College Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ============================================================================
#  ALL THE CSS THAT MAKES THE APP LOOK "AMAZING" LIVES HERE
#  (This is the only "new" chunk of the file that has nothing to do with
#  your chatbot's brain — it only changes how things look.)
# ============================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Lora:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    :root {
        --bg: #14201A;
        --panel: #203028;
        --panel-border: rgba(201, 162, 39, 0.25);
        --gold: #C9A227;
        --gold-soft: #E4C766;
        --sage: #3F5544;
        --text: #F2EEE1;
        --text-muted: #A9BBA4;
        --radius: 6px;
    }

    /* Hide Streamlit's default chrome for a cleaner look */
    #MainMenu, footer, header {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background-color: var(--bg);
        color: var(--text);
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: #101A14;
        border-right: 1px solid var(--panel-border);
    }
    [data-testid="stSidebar"] * {
        color: var(--text) !important;
    }
    .brand-title {
        font-family: 'Lora', serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--gold-soft);
        margin-bottom: 0;
    }
    .brand-subtitle {
        color: var(--text-muted) !important;
        font-size: 0.85rem;
        margin-top: 0.1rem;
        margin-bottom: 1.2rem;
    }
    .gold-rule {
        border: none;
        border-top: 1px solid var(--panel-border);
        margin: 1rem 0 1.2rem 0;
    }
    .sidebar-label {
        font-size: 0.78rem;
        color: var(--text-muted) !important;
        margin-bottom: 0.3rem;
    }

    div[data-testid="stSidebar"] .stButton > button {
        background-color: transparent;
        border: 1px solid var(--gold);
        color: var(--gold-soft) !important;
        border-radius: var(--radius);
        width: 100%;
        padding: 0.5rem 0;
        transition: background-color 0.15s ease;
    }
    div[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(201, 162, 39, 0.12);
        border-color: var(--gold-soft);
    }

    /* ---------- Main header ---------- */
    .app-header {
        padding-bottom: 0.6rem;
        margin-bottom: 1.4rem;
        border-bottom: 2px solid var(--gold);
    }
    .app-header h1 {
        font-family: 'Lora', serif;
        font-weight: 700;
        font-size: 2rem;
        color: var(--text);
        margin-bottom: 0.15rem;
    }
    .app-header p {
        color: var(--text-muted);
        font-size: 0.95rem;
        margin: 0;
    }

    /* ---------- Empty state ---------- */
    .empty-state {
        border: 1px dashed var(--panel-border);
        border-radius: var(--radius);
        padding: 2rem;
        text-align: center;
        color: var(--text-muted);
        margin-top: 1rem;
    }
    .empty-state span {
        font-family: 'Lora', serif;
        color: var(--gold-soft);
        font-size: 1.1rem;
    }

    /* ---------- Chat messages (native st.chat_message, themed) ---------- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    [data-testid="stChatMessage"] {
        border-radius: var(--radius);
        padding: 0.85rem 1rem;
        margin-bottom: 0.7rem;
        animation: fadeInUp 0.25s ease;
    }
    /* Streamlit gives the user avatar and assistant avatar different, stable
       testids -- we use :has() to theme the whole row based on who sent it */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: rgba(63, 85, 68, 0.35);
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: var(--panel);
        border-left: 3px solid var(--gold);
    }
    [data-testid="stChatMessageAvatarUser"] {
        background-color: var(--sage) !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: var(--gold) !important;
    }

    /* Make the model's markdown (tables, bold, lists) look like a ledger */
    [data-testid="stChatMessageContent"] strong {
        color: var(--gold-soft);
    }
    [data-testid="stChatMessageContent"] table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.6rem 0;
    }
    [data-testid="stChatMessageContent"] th,
    [data-testid="stChatMessageContent"] td {
        border: 1px solid var(--panel-border);
        padding: 0.45rem 0.65rem;
        text-align: left;
    }
    [data-testid="stChatMessageContent"] th {
        background-color: rgba(201, 162, 39, 0.15);
        color: var(--gold-soft);
    }
    [data-testid="stChatMessageContent"] code {
        background-color: rgba(0, 0, 0, 0.25);
        padding: 0.1rem 0.35rem;
        border-radius: 3px;
    }

    /* ---------- Chat input box ---------- */
    [data-testid="stChatInput"] {
        background-color: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: var(--radius);
    }
    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
#  Step 1 - Building the RAG retrievers
#  (unchanged logic — just wrapped in @st.cache_resource so it only runs once)
# ============================================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embeddings = get_embeddings()


@st.cache_resource(show_spinner="Reading and indexing college documents...")
def build_retriever(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(document)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Friendly check so beginners get a clear message instead of a crash if the
# PDFs aren't in the same folder as this script.
missing_files = [f for f in ("academics_handbook.pdf", "fee_structure.pdf") if not os.path.exists(f)]
if missing_files:
    st.error(
        "Missing file(s): " + ", ".join(missing_files) +
        ". Please put them in the same folder as app.py, then refresh the page."
    )
    st.stop()

acedemic_retriever = build_retriever("academics_handbook.pdf")
fee_retriever = build_retriever("fee_structure.pdf")


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)

llm = get_llm()


# ============================================================================
#  Step 2: State  (unchanged)
# ============================================================================
class State(TypedDict):
    programme: str
    messages: Annotated[list, add_messages]
    query_type: str
    retrieved_context: str


# ============================================================================
#  Step 3: Nodes  (unchanged)
# ============================================================================
def classifier_node(state: State) -> dict:
    """Look at the latest user message and decide which path to take."""
    last_message = state['messages'][-1].content

    prompt = (
        "Classify the following student query into exactly one category: "
        "'academic', 'fee', or 'general'.\n\n"
        "Use 'academic' for questions about attendance, exams, grading, credits, "
        "promotion, course structure, summer training, or degree requirements.\n"
        "Use 'fee' for questions about tuition, payment, refund, late charges, "
        "scholarships, or any money-related topic.\n"
        "Use 'general' for greetings, casual talk, or anything not related to "
        "the college rules or fee.\n\n"
        f"Query: {last_message}\n\n"
        "Return only one word: academic, fee, or general."
    )
    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    if "academic" in category:
        category = "academic"
    elif "fee" in category:
        category = "fee"
    else:
        category = "general"
    return {"query_type": category}


def academic_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the academics handbook."""
    query = state["messages"][-1].content
    docs = acedemic_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def fee_rag_node(state: State) -> dict:
    """Retrieves relevant chunks from the fee structure PDF."""
    query = state['messages'][-1].content
    docs = fee_retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"retrieved_context": context}


def general_node(state: State) -> dict:
    """Answers directly using the LLM's own knowledge, no retrieval needed."""
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}


def response_node(state: State) -> dict:
    """Generates the final answer, personalized using the student's programme."""
    query = state["messages"][-1].content
    programme = state.get("programme", "Unknown")
    context = state["retrieved_context"]

    if context == "NO_RETRIEVAL_NEEDED":
        prompt = (
            f"You are a friendly college assistant talking to a {programme} student. "
            f"Answer this question using your own general knowledge:\n\n{query}"
        )
    else:
        prompt = (
            f"You are a college assistant helping a {programme} student. "
            f"Use the following context from the official college documents to answer "
            f"the question accurately. If the context mentions specific figures for "
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Give a clear, friendly, and precise answer."
        )

    response = llm.invoke(prompt)
    return {"messages": [("ai", response.content.strip())]}


# ============================================================================
#  Step 4: Router  (unchanged)
# ============================================================================
def route_query(state: State):
    if state['query_type'] == 'academic':
        return "academic_rag"
    elif state['query_type'] == "fee":
        return "fee_rag"
    else:
        return "general"


# ============================================================================
#  Step 5-8: Build and compile the graph  (unchanged, wrapped so it builds once)
# ============================================================================
@st.cache_resource(show_spinner=False)
def get_graph_app():
    graph = StateGraph(State)

    graph.add_node("classifier", classifier_node)
    graph.add_node("academic_rag", academic_rag_node)
    graph.add_node("fee_rag", fee_rag_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "classifier")
    graph.add_conditional_edges("classifier", route_query)
    graph.add_edge("academic_rag", "response")
    graph.add_edge("fee_rag", "response")
    graph.add_edge("general", "response")
    graph.add_edge("response", END)

    return graph.compile()

graph_app = get_graph_app()


# ============================================================================
#  Step 9: The Streamlit UI  (this replaces your console input()/print() loop)
# ============================================================================

PROGRAMMES = ["BCA", "BBA", "B.Com (H)"]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "content": str}
if "programme" not in st.session_state:
    st.session_state.programme = PROGRAMMES[0]

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<p class="brand-title">🎓 College Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-subtitle">Ask about academics, fees, or anything else</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-label">Your programme</p>', unsafe_allow_html=True)
    st.session_state.programme = st.radio(
        "Your programme",
        PROGRAMMES,
        index=PROGRAMMES.index(st.session_state.programme),
        label_visibility="collapsed",
    )

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)

    if st.button("🗑️  Start new conversation"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown('<hr class="gold-rule">', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-subtitle">Powered by LangGraph + Groq</p>',
        unsafe_allow_html=True,
    )

# ---------- Main header ----------
st.markdown(
    f"""
    <div class="app-header">
        <h1>Welcome, {st.session_state.programme} student</h1>
        <p>Ask me about attendance, exams, fees, scholarships, or anything else about college life.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


def render_message(role: str, content: str):
    # st.chat_message gives us the avatar + row automatically (styled via the
    # CSS above). st.markdown() is what actually turns **bold**, pipe tables,
    # and bullet lists into real formatting instead of raw symbols.
    with st.chat_message(role):
        if role == "assistant":
            # unsafe_allow_html=True lets through any raw <br> tags the model
            # uses inside table cells for line breaks.
            st.markdown(content, unsafe_allow_html=True)
        else:
            st.markdown(content)


# ---------- Chat history panel ----------
chat_panel = st.container(height=480)
with chat_panel:
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div class="empty-state">
                <span>No messages yet</span><br><br>
                Try asking: "How many backlogs are allowed for promotion?"
                or "What's the last date to pay semester fees?"
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.chat_history:
            render_message(msg["role"], msg["content"])

# ---------- Chat input (pinned at the bottom) ----------
user_prompt = st.chat_input("Type your question here...")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    with st.spinner("Checking the college records..."):
        result = graph_app.invoke({
            "programme": st.session_state.programme,
            "messages": [("human", user_prompt)],
        })
        answer = result["messages"][-1].content

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()