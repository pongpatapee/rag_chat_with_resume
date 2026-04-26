import os, glob
from dotenv import load_dotenv
import psycopg2
from pgvector.psycopg2 import register_vector
from google import genai
from google.genai import types
from pypdf import PdfReader

load_dotenv()

ai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_DIM = 768


def get_conn():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    return conn


def setup_db(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                content TEXT,
                source TEXT,
                embedding vector(768)
            )
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS documents_embedding_idx
            ON documents
            USING hnsw (embedding vector_cosine_ops)
        """)

    conn.commit()


def embed(texts):
    result = ai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )

    return result.embeddings


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def extract_pdf_file(filepath):
    reader = PdfReader(filepath)
    return "\n".join(page.extract_text() for page in reader.pages)


def extract_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(filepath):
    if filepath.endswith(".pdf"):
        return extract_pdf_file(filepath)

    return extract_text_file(filepath)


def ingest_file(filepath, conn):
    print(f"Ingesting file {filepath}...")

    text = extract_text(filepath)
    chunks = chunk_text(text)

    embeddings = embed(chunks)

    if not embeddings:
        print("No embeddings")
        return

    with conn.cursor() as cur:
        for chunk, embedding in zip(chunks, embeddings):
            print(chunk, embedding)
            cur.execute(
                "INSERT INTO documents (content, source, embedding) VALUES (%s, %s, %s)",
                (chunk, filepath, embedding.values),
            )
    conn.commit()
    print(f" ->{len(chunks)} chunks stored")


if __name__ == "__main__":
    conn = get_conn()
    setup_db(conn)

    files = glob.glob("docs/**/*.pdf", recursive=True)
    files += glob.glob("docs/**/*.txt", recursive=True)
    files += glob.glob("docs/**/*.md", recursive=True)

    if not files:
        print("no files in docs/ . Add pdf, md, or txt files")
        conn.close()
        exit()

    for f in files:
        ingest_file(f, conn)

    print(f"Ingested {len(files)} files. Done")
    conn.close()
