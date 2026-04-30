from fastapi import FastAPI ,UploadFile, File,HTTPException
from fastapi.staticfiles import StaticFiles
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader,PyPDFLoader
from langchain_community.embeddings  import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import uvicorn
import os
import shutil
load_dotenv()
vector_db = None   

app = FastAPI()

def load_doc(file_path:str):
    file_path = os.path.abspath(file_path)  
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    return docs


vector_db = None   # 🔥 GLOBAL

def pdf_ingest(file_path):
    global vector_db

    print("processing", file_path)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    persistent_directory = os.path.join(current_directory, "db", "chroma_db")

    documents = load_doc(file_path)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if vector_db is None:
        vector_db = Chroma.from_documents(
            split_docs,
            embedding=embedding,
            persist_directory=persistent_directory
        )
    else:
        vector_db.add_documents(split_docs)

    vector_db.persist()
    

Upload_Dir = "uploads"
os.makedirs(Upload_Dir, exist_ok=True)

print("UPLOAD DIR:", os.path.abspath(Upload_Dir))
app.mount("/files",StaticFiles(directory=Upload_Dir), name="files")

@app.post("/upload")
def upload_file(file:UploadFile=File(...)):
    filename=file.filename
    file_path = os.path.join(Upload_Dir,filename)
    print("Saved at:", file_path)
    print("Exists?", os.path.exists(file_path))  

    

    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
        
    pdf_ingest(file_path)
    file.file.close()
    return{
            "message":"uploaded sucessfully",
            "filename":filename,
            "file_url":f"http://127.0.0.1:8000/files/{filename}"
        }
@app.post("/chat")
def chat(question: str):
    global vector_db

    if vector_db is None:
        return {"answer": "Upload a PDF first"}

    docs = vector_db.similarity_search(question, k=4)

    context = "\n".join([doc.page_content for doc in docs])

    llm = ChatGroq(model="openai/gpt-oss-120b")

    prompt = f"""
    Answer only from this context:
    {context}

    Question: {question}
    """

    response = llm.invoke(prompt)

    return {"answer": response.content}

@app.get("/files/{filename}")
def get_file(filename:str):
    file_path = os.path.join(Upload_Dir,filename)


    return{
        "file_url":f"http://127.0.0.1:8000/files/{filename}"
    }
@app.get("/")
def homepage():
    return "hellooo"