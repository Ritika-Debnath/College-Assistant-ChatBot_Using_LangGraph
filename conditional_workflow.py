# A CONDITIONAL WORKFLOW: where the next step is decided based on a condition. The graph looks at the current state and chooses which path to take.
# In a sequential workflow, the path is fixed - A → B → C, every single time.
# In a conditional workflow, the path changes - A → (check something) → go to B or go to C depending on the answer.
#Now usually to create a conditional workflow you need 2 pieces. 
# ---> Piece 1 - A router function 
# ---> Piece 2 - "add_conditional_edges"(which calls the router function to decide).


import os  
from typing import TypedDict, Annotated  # Imports tools for defining typed state dictionaries and adding extra information to types
from langgraph.graph.message import add_messages  # Imports a function to add new messages to an existing message list
from langgraph.graph import StateGraph, START, END  # Imports tools to create the graph and define its starting and ending points
from langchain_groq import ChatGroq  # Imports the Groq chat model for using an LLM
from langchain_community.document_loaders import PyPDFLoader  # Imports a loader to read and extract text from PDF files
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Imports a tool to split large text into smaller chunks
from langchain_huggingface import HuggingFaceEmbeddings  # Imports Hugging Face embeddings to convert text into numerical vectors
from langchain_community.vectorstores import FAISS  # Imports FAISS for storing and searching text embeddings efficiently
from dotenv import load_dotenv 
load_dotenv()

# Step 1 - Building the RAG retrievers :
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Load the Hugging Face model for converting text into numerical vectors

def build_retriever(pdf_path: str):      # Define a function that creates a retriever from a PDF file
    loader = PyPDFLoader(pdf_path)       # Create a PDF loader using the given file path
    document = loader.load()             # Load the PDF and extract its text
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)       # Create 800-character chunks with 100-character overlap
    chunks = splitter.split_documents(document)                 # Split the PDF documents into smaller chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)      # Convert chunks into embeddings and store them in FAISS
    return vectorstore.as_retriever(search_kwargs={"k": 4})     # Create a retriever that returns the 4 most relevant chunks

acedemic_retriever = build_retriever("academics_handbook.pdf")  # Create a retriever for the academic handbook
fee_retriever = build_retriever("fee_structure.pdf")            # Create a retriever for the fee structure document

# Set the LLM:

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.4)  # Create the Groq LLM with a moderate response creativity


# Step 2: Create State:
class State(TypedDict):     # Define the structure of data passed between graph nodes
    programme: str          # Store the student's programme name
    messages: Annotated[list, add_messages]       # Store the conversation messages
    query_type: str                               # Store the category of the user's query
    retrieved_context: str                        # Store the information retrieved from the documents


# Step 3: Create Nodes:
# ================= NODE 1 ====================
def classifier_node(state: State) -> dict:      # Define a node that classifies the user's question
    """Look at the latest user message and decide which path to take."""        # Explain the purpose of this node
    last_message = state["messages"][-1].content        # Get the latest user message

    prompt = (       # Create a prompt for classifying the user's question
        "Classify the following student query into exactly one category: "  
        "'academic', 'fee', or 'general'.\n\n"  
        "Use 'academic' for questions about attendance, exams, grading, credits, " 
        "promotion, course structure, summer training, or degree requirements.\n"  
        "Use 'fee' for questions about tuition, payment, refund, late charges, " 
        "scholarships, or any money-related topic.\n"  
        "Use 'general' for greetings, casual talk, or anything not related to "  
        "the college rules or fee.\n\n"  
        f"Query: {last_message}\n\n"                         # Add the user's question to the prompt
        "Return only one word: academic, fee, or general."   # Ask the LLM to return only the category
    )
    response = llm.invoke(prompt)                       # Send the "classifier_node" prompt to the LLM
    category = response.content.strip().lower()         # Clean and convert the response to lowercase

    if "academic" in category:       # Check if the response indicates an academic query
        category = "academic"        # Set the query type to academic
    elif "fee" in category:          # Check if the response indicates a fee query
        category = "fee"             # Set the query type to fee
    else:                            # Handle all other types of queries
        category = "general"         # Set the query type to general
    return {"query_type": category}  # Return the detected query category


# ================= NODE 2 ====================
def academic_rag_node(state: State) -> dict:                            # Define a node for retrieving academic information
    """Retrieves relevant chunks from the academics handbook."""        # Explain the purpose of this node
    query = state["messages"][-1].content                               # Get the latest user question
    docs = acedemic_retriever.invoke(query)                             # Search the academic handbook for relevant chunks
    context = "\n\n".join([doc.page_content for doc in docs])           # Combine retrieved chunks into one context
    return {"retrieved_context": context}                               # Store the retrieved academic information in the state


# ================= NODE 3 ====================
def fee_rag_node(state: State) -> dict:  # Define a node for retrieving fee information
    """Retrieves relevant chunks from the fee structure PDF."""  # Explain the purpose of this node
    query = state["messages"][-1].content  # Get the latest user question
    docs = fee_retriever.invoke(query)     # Search the fee document for relevant chunks
    context = "\n\n".join([doc.page_content for doc in docs])  # Combine retrieved chunks into one context
    return {"retrieved_context": context}  # Store the retrieved fee information in the state


# ================= NODE 4 ====================
def general_node(state: State) -> dict:          # Define a node for general questions
    """Answers directly using the LLM's own knowledge, no retrieval needed."""  
    return {"retrieved_context": "NO_RETRIEVAL_NEEDED"}  # Mark that no document context is needed


# ================= NODE 5 ====================
def response_node(state: State) -> dict:            # Define a node that generates the final answer
    """Generates the final answer, personalized using the student's programme."""  
    query = state["messages"][-1].content               # Get the user's latest question
    programme = state.get("programme", "Unknown")       # Get the student's programme from the state
    context = state["retrieved_context"]                # Get the retrieved document information

    if context == "NO_RETRIEVAL_NEEDED":        # Check if the question does not need document information
        prompt = (  
            f"You are a friendly college assistant talking to a {programme} student. "  # Personalize the assistant for the student
            f"Answer this question using your own general knowledge:\n\n{query}"        # Add the user's question
        )
    else:       # Handle academic or fee questions requiring retrieved information
        prompt = (  
            f"You are a college assistant helping a {programme} student. "                       # Tell the LLM which student it is helping
            f"Use the following context from the official college documents to answer "     
            f"the question accurately. If the context mentions specific figures for "  
            f"different programmes, highlight the one relevant to {programme} if possible.\n\n"  # Ask for relevant programme information
            f"Context:\n{context}\n\n"          # Provide the retrieved document context
            f"Question: {query}\n\n"            # Provide the user's question
            f"Give a clear, friendly, and precise answer."   
        )

    response = llm.invoke(prompt)                               # Send the final prompt to the LLM
    return {"messages": [("ai", response.content.strip())]}     # Add the generated answer to the conversation



# Step 4: Creating Router Function:
def route_query(state: State):                  # Define a function that decides which node to visit next
    if state["query_type"] == "academic":       # Check if the query is academic
        return "academic_rag"                   # Route the query to the academic RAG node
    elif state["query_type"] == "fee":          # Check if the query is about fees
        return "fee_rag"                        # Route the query to the fee RAG node
    else:                                       # Handle general questions
        return "general"                        # Route the query to the general node


# Step 5: Create the Graph
graph = StateGraph(State)  # Create a LangGraph using the defined State structure


# Step 6: Add Nodes in the Graph
graph.add_node("classifier", classifier_node)           # Add the classifier node to the graph
graph.add_node("academic_rag", academic_rag_node)       # Add the academic retrieval node
graph.add_node("fee_rag", fee_rag_node)                 # Add the fee retrieval node
graph.add_node("general", general_node)                 # Add the general question node
graph.add_node("response", response_node)               # Add the final response node


# Step 7: Add Edges in the Graph:-
graph.add_edge(START, "classifier")             # Start the graph by sending the query to the classifier

# Conditional Edge:
graph.add_conditional_edges("classifier", route_query)  # Route the query based on its classified category

# Add the 3 Nodes to the "Response" using edge:
graph.add_edge("academic_rag", "response")          # Send academic results to the response node
graph.add_edge("fee_rag", "response")               # Send fee results to the response node
graph.add_edge("general", "response")               # Send general queries to the response node

# Add the "response" with the END:
graph.add_edge("response", END)                     # End the graph after generating the final response



# Step 8: Compile the Graph
app = graph.compile()  # Compile the graph so it can be executed



# ==============*************===============
# ==============*************===============
# FROM 177-208 LINE OF CODE YOU DON'T NEED BEACUSE WE WILL USE STREAMLIT FOR FRONTEND , SO THIS PART YOU CAN USE FOR TERMINAL USE:
# Step 9: Run the Code
print("Welcome to the College Assistant\n\n")  # Display a welcome message

print("Which programme are you in?")           # Ask the student to select their programme
print("1. BCA")  
print("2. BBA")  
print("3. B.Com (H)")  

choice = input("\nEnter 1, 2 or 3: ")  # Take the student's programme choice

programme_map = {       # Create a dictionary that maps choices to programme names
    "1": "BCA",         # Map option 1 to BCA
    "2": "BBA",         # Map option 2 to BBA
    "3": "B.Com (H)"    # Map option 3 to B.Com (H)
}

student_programme = programme_map.get(choice, "BCA")                # Get the programme or use BCA as the default
print(f"\nGreat! You're set as a {student_programme} student.")     # Confirm the selected programme


# Step 10: Create a Continuous Chatbot
while True:                             # Keep the chatbot running continuously
    user_query = input("You: ")         # Take a question from the student
    if user_query.lower() in ["exit", "quit"]:          # Check if the user wants to stop the chatbot
        break               # Exit the chatbot loop

    result = app.invoke({                       # Run the LangGraph with the user's question
        "programme": student_programme,         # Pass the student's programme to the graph
        "messages": [("human", user_query)]     # Pass the user's question as a human message
    })


    print(f"Assistant: {result['messages'][-1].content}")  # Display the assistant's final response