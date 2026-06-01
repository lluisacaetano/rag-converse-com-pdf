# Demo RAG - Converse com PDF sobre Bike Fitting

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

def main():
    print("=" * 50)
    print("SISTEMA RAG GRATUITO COM OLLAMA")
    print("=" * 50)

    # 1. Carregar PDF
    print("\nCarregando PDF...")
    loader = PyPDFLoader("./bikefitting.pdf")
    documentos = loader.load()
    print(f"PDF carregado com {len(documentos)} página(s)")

    # 2. Dividir em chunks
    print("Dividindo texto em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Texto dividido em {len(chunks)} chunks")

    # 3. Criar banco vetorial
    print("Criando embeddings e indexando no ChromaDB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Banco vetorial criado!")

    # 4. Inicializar LLM
    print("Inicializando modelo de linguagem...\n")
    llm = OllamaLLM(model="llama3", temperature=0)

    print("=" * 50)
    print("RESPONDENDO PERGUNTAS SOBRE O PDF")
    print("=" * 50)

    # Perguntas sobre Bike Fitting
    perguntas = [
        "O que é bike fitting?",
        "Quais são os benefícios do bike fitting para o ciclista?",
        "Como o bike fitting pode ajudar a prevenir lesões?",
        "Quais ajustes são feitos em um bike fitting?"
    ]

    for pergunta in perguntas:
        print(f"\nPERGUNTA: {pergunta}")
        print("-" * 50)

        # Buscar contexto relevante
        docs = vectorstore.similarity_search(pergunta, k=3)
        contexto = "\n\n".join([doc.page_content for doc in docs])

        # Criar prompt
        prompt = f"""Use o contexto abaixo para responder à pergunta.
Responda em português de forma clara e concisa.

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""

        # Gerar resposta
        resposta = llm.invoke(prompt)
        print(f"RESPOSTA: {resposta}")

    print("\n" + "=" * 50)
    print("DEMO FINALIZADA!")
    print("=" * 50)

if __name__ == "__main__":
    main()
