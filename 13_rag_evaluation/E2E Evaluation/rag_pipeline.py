import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PDF_PATH = "../data/sustainable_development.pdf"
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "nvidia/nemotron-3.5-super-vl:free"


def build_rag_chain():
    docs = PyPDFLoader(PDF_PATH).load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=150
    ).split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
    )

    vectorstore = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_DIR
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    prompt = ChatPromptTemplate.from_template(
        "Use only the context below to answer the question. "
        "If the context does not contain the answer, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}"
    )

    chain = prompt | llm

    return chain, retriever