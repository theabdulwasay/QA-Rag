"""
ask.py — loads the FAISS vector store built by ingest.py and answers
questions about your documents using a LangChain retrieval chain.

Run:
    python ask.py
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

STORE_DIR = "vectorstore"

PROMPT_TEMPLATE = """You are a helpful assistant answering questions about a
company handbook. Use only the context below to answer. If the answer isn't
in the context, say you don't know.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        STORE_DIR, embeddings, allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def main():
    if not os.path.isdir(STORE_DIR):
        print("No vector store found. Run `python ingest.py` first.")
        return

    chain = build_chain()
    print("Ask questions about the handbook. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
