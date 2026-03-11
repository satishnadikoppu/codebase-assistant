import os
import time
from git import Repo
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

CLONE_DIR = "../cloned_repo"

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB", "ragdb"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432")
)

register_vector(conn)


def clone_repository(repo_url: str):

    if os.path.exists(CLONE_DIR) and os.listdir(CLONE_DIR):
        print("Repository already cloned.")
        return

    print("Cloning repository...")

    Repo.clone_from(repo_url, CLONE_DIR)

    print("Repository cloned successfully.")


def collect_source_files(repo_path):

    source_files = []

    allowed_extensions = {
        ".py",
        ".js",
        ".ts",
        ".go",
        ".java"
    }

    ignored_dirs = {
        ".git",
        "__pycache__",
        "node_modules",
        "build",
        "dist",
        "tests",
        "test",
        "examples",
        "docs",
        "scripts",
        ".venv",
        "venv"
    }

    for root, dirs, files in os.walk(repo_path):

        # remove ignored directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:

            if any(file.endswith(ext) for ext in allowed_extensions):

                full_path = os.path.join(root, file)
                source_files.append(full_path)

    return source_files


def chunk_code(text, chunk_size=40, overlap=10):

    lines = text.split("\n")
    chunks = []

    start = 0

    while start < len(lines):

        end = start + chunk_size

        chunk = "\n".join(lines[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def process_files(files):

    all_chunks = []

    for file_path in files:

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()

        except Exception:
            continue

        chunks = chunk_code(code)

        for i, chunk in enumerate(chunks):

            all_chunks.append({
                "file_path": file_path,
                "chunk_index": i,
                "content": chunk
            })

    return all_chunks


def create_table():

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS code_chunks (
        id SERIAL PRIMARY KEY,
        file_path TEXT,
        chunk_index INTEGER,
        content TEXT,
        embedding VECTOR(384)
    )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS code_chunks_embedding_idx
        ON code_chunks
        USING hnsw (embedding vector_l2_ops)
        """)

    conn.commit()


def store_chunks(chunks):

    cursor = conn.cursor()

    cursor.execute("DELETE FROM code_chunks")

    # Embed file path + content together so the model knows which file each chunk belongs to.
    # We store only the original content, but the embedding captures the file context too.
    texts = [f"File: {c['file_path']}\n\n{c['content']}" for c in chunks]

    print("Creating embeddings...")

    embeddings = model.encode(texts, batch_size=64)

    print("Storing chunks in database...")

    records = [
        (chunk["file_path"], chunk["chunk_index"], chunk["content"], embedding.tolist())
        for chunk, embedding in zip(chunks, embeddings)
    ]

    cursor.executemany("""
    INSERT INTO code_chunks (file_path, chunk_index, content, embedding)
    VALUES (%s, %s, %s, %s)
    """, records)

    conn.commit()


if __name__ == "__main__":

    repo_url = input("Enter GitHub repository URL: ")

    clone_repository(repo_url)

    files = collect_source_files(CLONE_DIR)

    print(f"Found {len(files)} source files")

    for f in files[:10]:
        print(f)

    create_table()

    chunks = process_files(files)

    print(f"Generated {len(chunks)} code chunks")

    start = time.time()

    store_chunks(chunks)

    print(f"Code chunks stored in vector database in {time.time() - start:.1f}s")
