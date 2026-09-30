# 🎓 College Assistant

A chatbot that answers college students' questions about **academics** and **fees**, using the college's own PDF documents. Built with **LangGraph**, **Groq**, and **Streamlit**.

## How It Works

1. The student picks their programme (BCA, BBA, or B.Com (H)) and asks a question.
2. A **classifier** decides if the question is `academic`, `fee`, or `general`.
3. A **router** sends it down the right path:
   - `academic` → searches `academics_handbook.pdf`
   - `fee` → searches `fee_structure.pdf`
   - `general` → answers directly, with no document search
4. The **response node** writes a friendly answer, personalized for the student's programme.

```
START → classifier → (academic_rag | fee_rag | general) → response → END
```

This is a **conditional workflow**: the path changes depending on the question.

## Tech Stack

- **LangGraph** – builds the workflow
- **Groq** (`openai/gpt-oss-20b`) – the LLM
- **FAISS + HuggingFace embeddings** (`all-MiniLM-L6-v2`) – document search (RAG)
- **Streamlit** – the web interface

## Project Files

```
├── app.py                  # Streamlit web app
├── main.py                 # Terminal version (rename to match your file)
├── academics_handbook.pdf  # Academic rules (you provide)
├── fee_structure.pdf       # Fee details (you provide)
├── .env                    # Your API key
└── requirements.txt
```

## Setup

**1. Install the libraries**

```bash
pip install streamlit langgraph langchain-groq langchain-community langchain-huggingface langchain-text-splitters faiss-cpu pypdf sentence-transformers python-dotenv
```

**2. Add your Groq API key** – create a `.env` file:

```
GROQ_API_KEY=your_key_here
```

**3. Put both PDFs** in the same folder as `app.py`.

## Run

Web app:

```bash
streamlit run app.py
```

Terminal version:

```bash
python main.py
```

## Example Questions

- "How many backlogs are allowed for promotion?"
- "What's the last date to pay semester fees?"
- "Is there any scholarship available?"
