import os, glob
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("GOOGLE_API_KEY", ""),
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


def load_file(filepath):
    if filepath.endswith(".pdf"):
        return PyPDFLoader(filepath).load()
    else:
        return TextLoader(filepath).load()


if __name__ == "__main__":
    files = glob.glob("docs/**/*.pdf", recursive=True)
    files += glob.glob("docs/**/*.txt", recursive=True)
    files += glob.glob("docs/**/*.md", recursive=True)

    if not files:
        print("no files in docs/ . Add pdf, md, or txt files")
        exit()

    all_docs = []
    for f in files:
        all_docs += load_file(f)

    chunks = splitter.split_documents(all_docs)

    # creates the tables, embed, and insert in one call
    PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=os.getenv("DATABASE_URL", ""),
    )

    print(f"Ingested {len(files)} files. Done")
