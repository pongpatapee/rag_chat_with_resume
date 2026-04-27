import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

COLLECTION_NAME = "rag_docs"
CONNECTION_STRING = os.getenv("DATABASE_URL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 768))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.4))
TOP_K = int(os.getenv("TOP_K", 5))

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=CONNECTION_STRING,
)

# This is your retrieve() function, but now a reusable object
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": TOP_K, "score_threshold": SIMILARITY_THRESHOLD},
)

# This is your build_prompt() function
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant. Answer the question using the provided documents.
You may calculate, infer, and reason from the information given (e.g. computing years from dates).
If the answer truly isn't derivable from the documents, say "I don't have that information."

Documents:
{context}

Question: {question}
Answer:
"""
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)


def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# This is the full pipeline — the entire query.py from before, in 4 lines
# Read it left to right: retrieve → format → prompt → llm → parse output
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def main():
    print("RAG Demo (LangChain) — Chat with your docs")
    print("Type 'quit' to exit\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if question:
            answer = chain.invoke(question)
            print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
