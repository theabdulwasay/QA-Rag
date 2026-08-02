"""
ingest.py — loads documents from ./data, splits them into chunks,
embeds them, and saves a FAISS vector store to ./vectorstore.

Run this once (and again whenever ./data changes):
    python ingest.py
"""
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
STORE_DIR = "vectorstore"


def main():
    print(f"Loading documents from ./{DATA_DIR} ...")
    loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s).")

    print("Embedding chunks and building FAISS index ...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(STORE_DIR)
    print(f"Vector store saved to ./{STORE_DIR}")


if __name__ == "__main__":
    main()
