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
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_DIM = 768
SIMILARITY_THRESHOLD = 0.4
TOP_K = 5


def get_conn():
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    register_vector(conn)
    return conn


def retrieve(question, conn):
    result = ai_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(
            task_type="retrieval_query", output_dimensionality=EMBEDDING_DIM
        ),
    )

    query_embedding = result.embeddings[0].values

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT content, source, 1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            WHERE 1 - (embedding <=> %s::vector) > %s
            ORDER BY similarity DESC
            LIMIT %s
        """,
            (query_embedding, query_embedding, SIMILARITY_THRESHOLD, TOP_K),
        )
        return cur.fetchall()


def build_prompt(question, chunks):
    context = "\n\n---\n\n".join(
        f"[Source: {source}]\n{content}" for content, source, _ in chunks
    )

    return f"""You are a helpful assistant. Answer the question using ONLY the provided documents.
If the answer isn't in the documents, say "I dont' have that information" 

Documents:
{context}

Question: {question}
Answer:"""


def answer(question, conn):
    chunks = retrieve(question, conn)

    if not chunks:
        print("No relevant documents found above similarity threshold")
        return

    print(f"\nRetrieved {len(chunks)} chunks:")
    for _, source, similarity in chunks:
        print(f" {source} (similarity: {similarity:.2f})")

    prompt = build_prompt(question, chunks)

    response = ai_client.models.generate_content(model=LLM_MODEL, contents=prompt)

    print(f"\nAnswer: \n{response.text}")


def main():
    conn = get_conn()
    print("RAG demo: Chat with my resume")
    print("Type quit or exit to terminate the program")

    while True:
        question = input("You: ").strip()

        if question.lower() in ("quit", "exit"):
            break

        if question:
            answer(question, conn)

        print()

    conn.close()


if __name__ == "__main__":
    main()
