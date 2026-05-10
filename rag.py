from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import CrossEncoder

import numpy as np
import dotenv
import os

dotenv.load_dotenv()

api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api_key")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



def doc_loader(folder_path):
    docs = []
    source_path = Path(folder_path)

    if source_path.is_dir():
        pdf_paths = sorted(source_path.rglob("*.pdf"))
    else:
        pdf_paths = [source_path]

    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(str(pdf_path))
        docs.extend(loader.load())

    print(len(docs))
    return docs

def text_splitting(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(docs)

    print(len(chunks))
    return chunks




def vector_store(chunks):
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return store

def load_vector_store():
    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

def retrieve_docs(vector_store, query):
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
    relevant_docs = retriever.invoke(query)
    return relevant_docs

def generate_answer(vector_store, query):
    relevant_docs = retrieve_docs(vector_store, query)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that answers questions only from the provided context."
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}"
            ),
        ]
    )

    llm = ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
        api_key=api_key,
        temperature=0,
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    return {
        "answer": answer,
        "sources": [doc.metadata for doc in relevant_docs],
    }
