import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")

conn = psycopg2.connect(
    dbname=os.getenv("POSTGRES_DB", "ragdb"),
    user=os.getenv("POSTGRES_USER", "postgres"),
    password=os.getenv("POSTGRES_PASSWORD", "postgres"),
    host=os.getenv("POSTGRES_HOST", "localhost"),
    port=os.getenv("POSTGRES_PORT", "5432")
)

register_vector(conn)


def is_prose_line(line):
    """Returns True if a line looks like documentation, not code."""
    line = line.strip()
    if not line:
        return False
    return line.startswith(('"""', "'''", '#', 'Read more', 'http', '..'))


def code_density(text):
    """
    Returns a score from 0.0 to 1.0 representing how much of a chunk is real code.
    A score of 1.0 means every line is code. 0.0 means every line is a comment or docstring.
    """
    lines = text.splitlines()

    if not lines:
        return 0.0

    code_line_count = sum(1 for line in lines if line.strip() and not is_prose_line(line))

    return code_line_count / len(lines)


def path_matches_query(file_path, query):
    """Returns True if any word from the query appears in the file path."""
    query_words = query.lower().split()
    file_path_lower = file_path.lower()
    return any(word in file_path_lower for word in query_words)


def search_code(query, top_k=10):

    embedding = model.encode(query)

    cursor = conn.cursor()

    # Fetch more candidates than we need so we can re-rank them
    cursor.execute("""
        SELECT file_path, chunk_index, content
        FROM code_chunks
        ORDER BY embedding <-> %s::vector
        LIMIT 30
    """, (embedding.tolist(),))

    candidates = cursor.fetchall()

    cursor.close()

    # Score each chunk combining code density + file path relevance.
    # Chunks from files whose name matches a query keyword get a 0.3 bonus.
    def score(row):
        file_path, chunk_index, content = row
        density = code_density(content)
        path_bonus = 0.3 if path_matches_query(file_path, query) else 0.0
        return density + path_bonus

    ranked = sorted(candidates, key=score, reverse=True)

    return ranked[:top_k]


def explain_code(question, results):

    from collections import defaultdict

    file_chunks = defaultdict(list)

    for file_path, chunk_index, content in results:
        file_chunks[file_path].append(content)

    context_parts = []

    for file_path, chunks in file_chunks.items():

        joined_chunks = "\n\n".join(chunks)

        context_parts.append(
            f"File: {file_path}\n\n{joined_chunks}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an AI developer assistant.

Use the provided code snippets to answer the question about the codebase.

If the answer cannot be determined from the code snippets, say:
"I cannot determine the answer from the provided code."

Code snippets:
{context}

Question:
{question}

Answer clearly and explain the architecture if relevant.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    question = input("Ask about the codebase: ")

    results = search_code(question)

    answer = explain_code(question, results)

    print("\nAnswer:\n")
    print(answer)

    print("\nSources:\n")

    for file_path, chunk_index, _ in results:
        print(file_path, "chunk", chunk_index)
