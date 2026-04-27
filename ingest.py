import os, glob, uuid
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

class _SingleBatchEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        return [super().embed_documents([t], **kwargs)[0] for t in texts]


embeddings = _SingleBatchEmbeddings(
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

    # print(chunks)

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=os.getenv("DATABASE_URL", ""),
    )

    inserted_ids = vector_store.add_documents(
        chunks, ids=[str(uuid.uuid4()) for _ in chunks]
    )

    print(f"Chunks generated: {len(chunks)}")
    print(f"IDs returned by add_documents: {len(inserted_ids)}")
    print(f"Ingested {len(files)} files. Done")
