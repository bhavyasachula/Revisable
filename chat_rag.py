from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import uvicorn
import os
import shutil
import json
import sqlite3
import hashlib
from datetime import datetime, timezone

load_dotenv()

vector_db = None
chat_history = []
current_document_id = None

app = FastAPI(title="Revisable API")

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Upload directory ─────────────────────────────────────────────────────────
Upload_Dir = "uploads"
os.makedirs(Upload_Dir, exist_ok=True)
app.mount("/files", StaticFiles(directory=Upload_Dir), name="files")

current_directory = os.path.dirname(os.path.abspath(__file__))
app_db_path = os.path.join(current_directory, "db", "app.sqlite3")
chroma_directory = os.path.join(current_directory, "db", "chroma_db")
os.makedirs(os.path.dirname(app_db_path), exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def load_doc(file_path: str):
    file_path = os.path.abspath(file_path)
    loader = PyPDFLoader(file_path)
    return loader.load()


def db_connect():
    return sqlite3.connect(app_db_path)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def init_app_db():
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                file_url TEXT NOT NULL,
                uploaded_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS artifacts (
                document_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (document_id, kind)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )


def get_active_document_id():
    with db_connect() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = 'active_document_id'"
        ).fetchone()
    return row[0] if row else None


def set_active_document_id(document_id: str):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO app_state(key, value)
            VALUES('active_document_id', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (document_id,),
        )


def save_document(document_id: str, filename: str, file_url: str):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO documents(id, filename, file_url, uploaded_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename = excluded.filename,
                file_url = excluded.file_url,
                uploaded_at = excluded.uploaded_at
            """,
            (document_id, filename, file_url, utc_now()),
        )


def document_exists(document_id: str):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()
    return row is not None


def get_active_document():
    document_id = current_document_id or get_active_document_id()
    if not document_id:
        return None

    with db_connect() as conn:
        row = conn.execute(
            "SELECT id, filename, file_url, uploaded_at FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "filename": row[1],
        "file_url": row[2],
        "uploaded_at": row[3],
    }


def get_cached_artifact(kind: str):
    document_id = current_document_id or get_active_document_id()
    if not document_id:
        return None

    with db_connect() as conn:
        row = conn.execute(
            "SELECT payload FROM artifacts WHERE document_id = ? AND kind = ?",
            (document_id, kind),
        ).fetchone()

    return json.loads(row[0]) if row else None


def set_cached_artifact(kind: str, payload):
    document_id = current_document_id or get_active_document_id()
    if not document_id:
        return

    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO artifacts(document_id, kind, payload, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(document_id, kind) DO UPDATE SET
                payload = excluded.payload,
                updated_at = excluded.updated_at
            """,
            (document_id, kind, json.dumps(payload), utc_now()),
        )


def load_chat_history(document_id: str):
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE document_id = ?
            ORDER BY id ASC
            """,
            (document_id,),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def append_chat_message(document_id: str, role: str, content: str):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages(document_id, role, content, created_at)
            VALUES(?, ?, ?, ?)
            """,
            (document_id, role, content, utc_now()),
        )


def clear_chat_history(document_id: str):
    with db_connect() as conn:
        conn.execute("DELETE FROM chat_messages WHERE document_id = ?", (document_id,))


def hash_file(file_path: str):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collection_name(document_id: str):
    return f"doc_{document_id[:32]}"


def pdf_ingest(file_path: str, document_id: str):
    global vector_db

    print("Processing:", file_path)

    documents = load_doc(file_path)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embedding = get_embedding()

    if vector_db is None:
        vector_db = Chroma.from_documents(
            split_docs,
            embedding=embedding,
            persist_directory=chroma_directory,
            collection_name=collection_name(document_id),
        )
    else:
        vector_db.add_documents(split_docs)

    vector_db.persist()


def get_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def load_persisted_vector_db(document_id: str):
    if not os.path.exists(os.path.join(chroma_directory, "chroma.sqlite3")):
        return None

    return Chroma(
        persist_directory=chroma_directory,
        embedding_function=get_embedding(),
        collection_name=collection_name(document_id),
    )


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")


init_app_db()
current_document_id = get_active_document_id()
if current_document_id:
    chat_history = load_chat_history(current_document_id)
    vector_db = load_persisted_vector_db(current_document_id)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/")
def homepage():
    return {"message": "Revisable API is running"}


@app.get("/state")
def app_state():
    document = get_active_document()
    if not document:
        return {"uploaded": False, "chat_history": []}

    return {
        "uploaded": True,
        "document": document,
        "chat_history": load_chat_history(document["id"]),
        "has_bullets": get_cached_artifact("bullets") is not None,
        "has_flashcards": get_cached_artifact("flashcards") is not None,
    }


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    global chat_history, current_document_id, vector_db

    filename = file.filename
    file_path = os.path.join(Upload_Dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    document_id = hash_file(file_path)
    file_url = f"http://127.0.0.1:8000/files/{filename}"
    already_saved = document_exists(document_id)

    current_document_id = document_id
    set_active_document_id(document_id)
    save_document(document_id, filename, file_url)
    clear_chat_history(document_id)
    chat_history = []

    vector_db = load_persisted_vector_db(document_id) if already_saved else None
    if vector_db is None or not already_saved:
        pdf_ingest(file_path, document_id)

    file.file.close()

    return {
        "message": "Uploaded successfully",
        "document_id": document_id,
        "filename": filename,
        "file_url": file_url,
    }


# ── Chat ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(request: ChatRequest):
    global vector_db, chat_history, current_document_id

    if vector_db is None:
        return {"answer": "Please upload a PDF first."}

    document_id = current_document_id or get_active_document_id()
    if not document_id:
        return {"answer": "Please upload a PDF first."}

    docs = vector_db.similarity_search(request.question, k=4)
    context = "\n".join([doc.page_content for doc in docs])

    llm = get_llm()

    history_text = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in chat_history[-10:]]
    )

    prompt = f"""You are a helpful study assistant. Answer the user's question based on the provided document context.

Conversation history:
{history_text}

Document context:
{context}

User question: {request.question}

Provide a clear, accurate answer based on the document. If the answer isn't in the context, say so honestly."""

    response = llm.invoke(prompt)

    chat_history.append({"role": "user", "content": request.question})
    chat_history.append({"role": "assistant", "content": response.content})
    append_chat_message(document_id, "user", request.question)
    append_chat_message(document_id, "assistant", response.content)

    return {"answer": response.content}


# ── Bullet Points ─────────────────────────────────────────────────────────────
@app.post("/bullet-points")
def bullet_points():
    global vector_db

    if vector_db is None:
        return {"error": "Please upload a PDF first."}

    cached = get_cached_artifact("bullets")
    if cached is not None:
        return {"bullets": cached, "cached": True}

    queries = [
        "main topics and key concepts",
        "important definitions and terminology",
        "core principles and processes",
        "examples and applications",
    ]
    seen = set()
    all_docs = []
    for q in queries:
        for doc in vector_db.similarity_search(q, k=4):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    context = "\n\n".join([doc.page_content for doc in all_docs])

    llm = get_llm()

    prompt = f"""You are an expert study assistant. Based on the document below, generate a comprehensive bullet-point summary organized by topic.

Return ONLY a valid JSON array (no markdown, no explanation) in this exact format:
[
  {{
    "category": "Category Name",
    "points": ["point 1", "point 2", "point 3"]
  }}
]

Rules:
- Create 5 to 8 categories
- Each category should have 3 to 6 concise bullet points
- Pick a relevant emoji for each category
- Keep each point clear and under 20 words
- Cover all major topics in the document

Document content:
{context}"""

    response = llm.invoke(prompt)

    # Strip markdown code fences if present
    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        bullet_data = json.loads(raw)
    except Exception:
        bullet_data = [{"category": "Key Points", "emoji": "📌", "points": [response.content]}]

    set_cached_artifact("bullets", bullet_data)
    return {"bullets": bullet_data}


# ── Flashcards ────────────────────────────────────────────────────────────────
@app.post("/flashcards")
def flashcards():
    global vector_db

    if vector_db is None:
        return {"error": "Please upload a PDF first."}

    cached = get_cached_artifact("flashcards")
    if cached is not None:
        return {"flashcards": cached, "cached": True}

    queries = [
        "definitions and meanings",
        "how does it work processes",
        "what is the purpose use case",
        "key terms and concepts",
    ]
    seen = set()
    all_docs = []
    for q in queries:
        for doc in vector_db.similarity_search(q, k=4):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    context = "\n\n".join([doc.page_content for doc in all_docs])

    llm = get_llm()

    prompt = f"""You are an expert study assistant. Based on the document below, generate 15 to 20 flashcards for active recall studying.

Return ONLY a valid JSON array (no markdown, no explanation) in this exact format:
[
  {{
    "question": "Question here?",
    "answer": "Concise answer here.",
    "difficulty": "easy"
  }}
]

Rules:
- difficulty must be one of: "easy", "medium", "hard"
- Questions should test understanding, not just memorization
- Answers should be concise (1-3 sentences max)
- Mix different question styles: what/why/how/define

Document content:
{context}"""

    response = llm.invoke(prompt)

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        cards = json.loads(raw)
    except Exception:
        cards = [{"question": "Could not generate flashcards", "answer": response.content, "difficulty": "medium"}]

    set_cached_artifact("flashcards", cards)
    return {"flashcards": cards}


if __name__ == "__main__":
    uvicorn.run("chat_rag:app", host="0.0.0.0", port=8000, reload=True)
