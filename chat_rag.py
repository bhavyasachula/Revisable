from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response, Depends
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
import secrets
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
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )
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
                user_id INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(chat_messages)").fetchall()
        ]
        if "user_id" not in columns:
            conn.execute("ALTER TABLE chat_messages ADD COLUMN user_id INTEGER")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )


def app_state_key(user_id: int):
    return f"active_document_id:{user_id}"


def get_active_document_id(user_id: int):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT value FROM app_state WHERE key = ?",
            (app_state_key(user_id),),
        ).fetchone()
    return row[0] if row else None


def set_active_document_id(document_id: str, user_id: int):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO app_state(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (app_state_key(user_id), document_id),
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


def get_active_document(user_id: int):
    document_id = get_active_document_id(user_id)
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


def get_cached_artifact(kind: str, document_id: str | None = None):
    document_id = document_id or current_document_id
    if not document_id:
        return None

    with db_connect() as conn:
        row = conn.execute(
            "SELECT payload FROM artifacts WHERE document_id = ? AND kind = ?",
            (document_id, kind),
        ).fetchone()

    return json.loads(row[0]) if row else None


def set_cached_artifact(kind: str, payload, document_id: str | None = None):
    document_id = document_id or current_document_id
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


def load_chat_history(document_id: str, user_id: int):
    with db_connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE document_id = ? AND user_id = ?
            ORDER BY id ASC
            """,
            (document_id, user_id),
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]


def append_chat_message(document_id: str, user_id: int, role: str, content: str):
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages(document_id, user_id, role, content, created_at)
            VALUES(?, ?, ?, ?, ?)
            """,
            (document_id, user_id, role, content, utc_now()),
        )


def clear_chat_history(document_id: str, user_id: int):
    with db_connect() as conn:
        conn.execute(
            "DELETE FROM chat_messages WHERE document_id = ? AND user_id = ?",
            (document_id, user_id),
        )


def hash_file(file_path: str):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_email(email: str):
    return email.strip().lower()


def hash_password(password: str, salt: bytes | None = None):
    salt = salt or os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return f"{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored_hash: str):
    try:
        salt_hex, digest_hex = stored_hash.split(":", 1)
    except ValueError:
        return False

    expected = hash_password(password, bytes.fromhex(salt_hex)).split(":", 1)[1]
    return secrets.compare_digest(expected, digest_hex)


def public_user(row):
    return {"id": row[0], "name": row[1], "email": row[2]}


def create_session(user_id: int):
    token = secrets.token_urlsafe(32)
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO sessions(token, user_id, created_at)
            VALUES(?, ?, ?)
            """,
            (token, user_id, utc_now()),
        )
    return token


def get_session_user(token: str | None):
    if not token:
        return None

    with db_connect() as conn:
        row = conn.execute(
            """
            SELECT users.id, users.name, users.email
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (token,),
        ).fetchone()
    return public_user(row) if row else None


def get_current_user(request: Request):
    return get_session_user(request.cookies.get("revisable_session"))


def require_user(request: Request):
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Please log in first.")
    return user


def set_session_cookie(response: Response, token: str):
    response.set_cookie(
        "revisable_session",
        token,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 14,
    )


def clear_session_cookie(response: Response):
    response.delete_cookie("revisable_session", httponly=True, samesite="lax")


def collection_name(document_id: str):
    return f"doc_{document_id[:32]}"


def pdf_ingest(file_path: str, document_id: str):
    global vector_db

    print("Processing:", file_path)

    documents = load_doc(file_path)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embedding = get_embedding()

    vector_db = Chroma.from_documents(
        split_docs,
        embedding=embedding,
        persist_directory=chroma_directory,
        collection_name=collection_name(document_id),
    )

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


def activate_user_document(user_id: int):
    global vector_db, chat_history, current_document_id

    document_id = get_active_document_id(user_id)
    if not document_id:
        return None

    if current_document_id != document_id:
        current_document_id = document_id
        chat_history = load_chat_history(document_id, user_id)
        vector_db = load_persisted_vector_db(document_id)

    return document_id


def get_llm():
    return ChatGroq(model="llama-3.3-70b-versatile")


init_app_db()
current_document_id = None


# ── Routes ───────────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    name: str | None = None
    email: str
    password: str


def authenticate_response(response: Response, user_id: int):
    token = create_session(user_id)
    set_session_cookie(response, token)


@app.get("/me")
def me(user=Depends(require_user)):
    return {"user": user}

# signup route
@app.post("/signup")
def signup(request: AuthRequest, response: Response):
    name = (request.name or "").strip()
    email = normalize_email(request.email)
    password = request.password

    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Please enter your name.")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    try:
        with db_connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users(name, email, password_hash, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (name, email, hash_password(password), utc_now()),
            )
            user_id = cursor.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    authenticate_response(response, user_id)
    return {"user": {"id": user_id, "name": name, "email": email}}

#login route 
@app.post("/login")
def login(request: AuthRequest, response: Response):
    email = normalize_email(request.email)

    with db_connect() as conn:
        row = conn.execute(
            "SELECT id, name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if not row or not verify_password(request.password, row[3]):
        raise HTTPException(status_code=401, detail="Invalid email or password.")

    authenticate_response(response, row[0])
    return {"user": {"id": row[0], "name": row[1], "email": row[2]}}

#logout route
@app.post("/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get("revisable_session")
    if token:
        with db_connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    clear_session_cookie(response)
    return {"message": "Logged out"}


@app.get("/")
def homepage():
    return {"message": "Revisable API is running"}


@app.get("/state")
def app_state(user=Depends(require_user)):
    document = get_active_document(user["id"])
    if not document:
        return {"uploaded": False, "chat_history": []}

    return {
        "uploaded": True,
        "document": document,
        "chat_history": load_chat_history(document["id"], user["id"]),
        "has_bullets": get_cached_artifact("bullets", document["id"]) is not None,
        "has_flashcards": get_cached_artifact("flashcards", document["id"]) is not None,
    }


@app.post("/upload")
def upload_file(file: UploadFile = File(...), user=Depends(require_user)):
    global chat_history, current_document_id, vector_db

    filename = file.filename
    file_path = os.path.join(Upload_Dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    document_id = hash_file(file_path)
    file_url = f"http://127.0.0.1:8000/files/{filename}"
    already_saved = document_exists(document_id)

    current_document_id = document_id
    set_active_document_id(document_id, user["id"])
    save_document(document_id, filename, file_url)
    clear_chat_history(document_id, user["id"])
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
def chat(request: ChatRequest, user=Depends(require_user)):
    global vector_db, chat_history, current_document_id

    document_id = activate_user_document(user["id"])
    if not document_id or vector_db is None:
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
    append_chat_message(document_id, user["id"], "user", request.question)
    append_chat_message(document_id, user["id"], "assistant", response.content)

    return {"answer": response.content}


# ── Bullet Points ─────────────────────────────────────────────────────────────

MAX_CONTEXT_CHARS = 4000  # ~1000 tokens — keeps Groq requests well within limits


@app.post("/bullet-points")
def bullet_points(user=Depends(require_user)):
    global vector_db

    document_id = activate_user_document(user["id"])
    if not document_id or vector_db is None:
        return {"error": "Please upload a PDF first."}

    cached = get_cached_artifact("bullets", document_id)
    if cached is not None:
        return {"bullets": cached, "cached": True}

    # Reduced from 4 queries × k=4 → 2 queries × k=3 to limit context size
    queries = [
        "main topics and key concepts",
        "important definitions and processes",
    ]
    seen = set()
    all_docs = []
    for q in queries:
        for doc in vector_db.similarity_search(q, k=3):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    # Truncate total context to stay within token limits
    context_parts = []
    total_chars = 0
    for doc in all_docs:
        if total_chars + len(doc.page_content) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                context_parts.append(doc.page_content[:remaining])
            break
        context_parts.append(doc.page_content)
        total_chars += len(doc.page_content)

    context = "\n\n".join(context_parts)

    llm = get_llm()

    prompt = f"""Based on this document, generate study notes as JSON.

Return ONLY valid JSON (no markdown):
{{
  "topics": [
    {{
      "category": "Topic Name",
      "description": "Why it matters (1 sentence).",
      "points": ["point 1", "point 2", "point 3"]
    }}
  ],
  "overview": "2-3 sentence overview."
}}

Rules: 3-5 topics, 3-5 points each, under 20 words per point.

Document:
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

    if isinstance(bullet_data, list):
        bullet_data = {
            "topics": [
                {
                    "category": item.get("category", "Key Points"),
                    "description": item.get("description", ""),
                    "points": item.get("points", []),
                }
                for item in bullet_data
                if isinstance(item, dict)
            ],
            "overview": "",
        }

    set_cached_artifact("bullets", bullet_data, document_id)
    return {"bullets": bullet_data}


# ── Flashcards ────────────────────────────────────────────────────────────────
@app.post("/flashcards")
def flashcards(user=Depends(require_user)):
    global vector_db

    document_id = activate_user_document(user["id"])
    if not document_id or vector_db is None:
        return {"error": "Please upload a PDF first."}

    cached = get_cached_artifact("flashcards", document_id)
    if cached is not None:
        return {"flashcards": cached, "cached": True}

    # Reduced from 4 queries × k=4 → 2 queries × k=3 to limit context size
    queries = [
        "definitions and key concepts",
        "processes and use cases",
    ]
    seen = set()
    all_docs = []
    for q in queries:
        for doc in vector_db.similarity_search(q, k=3):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc)

    # Truncate total context to stay within token limits
    context_parts = []
    total_chars = 0
    for doc in all_docs:
        if total_chars + len(doc.page_content) > MAX_CONTEXT_CHARS:
            remaining = MAX_CONTEXT_CHARS - total_chars
            if remaining > 100:
                context_parts.append(doc.page_content[:remaining])
            break
        context_parts.append(doc.page_content)
        total_chars += len(doc.page_content)

    context = "\n\n".join(context_parts)

    llm = get_llm()

    prompt = f"""Based on this document, generate flashcards as JSON.

Return ONLY a valid JSON array (no markdown):
[
  {{
    "question": "Question?",
    "answer": "Concise answer.",
    "difficulty": "easy"
  }}
]

Rules: 8-12 cards, difficulty: easy/medium/hard, answers 1-2 sentences max, mix what/why/how.

Document:
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

    set_cached_artifact("flashcards", cards, document_id)
    return {"flashcards": cards}


if __name__ == "__main__":
    uvicorn.run("chat_rag:app", host="0.0.0.0", port=8000, reload=True)
