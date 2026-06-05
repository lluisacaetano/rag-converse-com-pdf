# Sistema RAG GRATUITO - Converse com um PDF
# Usa Ollama (modelos locais) - Não precisa de API key!

import os
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM

def carregar_pdf(caminho_pdf):
    """Carrega o PDF usando PyPDFLoader"""
    print(f"Carregando PDF: {caminho_pdf}")
    loader = PyPDFLoader(caminho_pdf)
    documentos = loader.load()
    print(f"PDF carregado com {len(documentos)} página(s)")
    return documentos

def dividir_em_chunks(documentos, chunk_size=1000, chunk_overlap=200):
    """Divide os documentos em chunks menores"""
    print(f"Dividindo texto em chunks (tamanho: {chunk_size}, sobreposição: {chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documentos)
    print(f"Texto dividido em {len(chunks)} chunks")
    return chunks

def criar_banco_vetorial(chunks, persist_directory="./chroma_db"):
    """Cria o banco vetorial ChromaDB com embeddings do Ollama"""
    print("Criando embeddings com Ollama e indexando no ChromaDB...")
    print("(Isso pode demorar um pouco na primeira vez)")

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Banco vetorial criado com sucesso!")
    return vectorstore

def fazer_pergunta(vectorstore, llm, pergunta):
    """Faz uma pergunta ao sistema RAG"""
    print(f"\nPergunta: {pergunta}")
    print("-" * 50)

    # Buscar documentos relevantes
    docs = vectorstore.similarity_search(pergunta, k=3)

    # Montar o contexto
    contexto = "\n\n".join([doc.page_content for doc in docs])

    # Criar o prompt
    prompt = f"""Use o contexto abaixo para responder à pergunta.
Se você não souber a resposta baseada no contexto, diga que não encontrou essa informação no documento.
Responda sempre em português de forma clara e concisa.

Contexto:
{contexto}

Pergunta: {pergunta}

Resposta:"""

    # Invocar o LLM
    resposta = llm.invoke(prompt)
    print(f"Resposta: {resposta}")
    return resposta

def main():
    # Caminho do PDF
    caminho_pdf = "./Bike Fitting- Efeito no desempenho e na incidência de lesões.pdf"

    # Verificar se o PDF existe
    if not os.path.exists(caminho_pdf):
        print(f"Erro: PDF não encontrado em {caminho_pdf}")
        return

    print("=" * 50)
    print("SISTEMA RAG GRATUITO COM OLLAMA")
    print("=" * 50)

    # Pipeline RAG
    # 1. Carregar o PDF
    documentos = carregar_pdf(caminho_pdf)

    # 2. Dividir em chunks
    chunks = dividir_em_chunks(documentos)

    # 3. Criar banco vetorial
    vectorstore = criar_banco_vetorial(chunks)

    # 4. Criar o LLM
    print("Inicializando modelo de linguagem...")
    llm = OllamaLLM(model="llama3", temperature=0)

    # 5. Fazer perguntas
    print("\n" + "=" * 50)
    print("SISTEMA PRONTO - Converse com o PDF!")
    print("=" * 50)

    # Perguntas de exemplo sobre Bike Fitting
    perguntas = [
        "O que é bike fitting?",
        "Quais são os benefícios do bike fitting para o ciclista?",
        "Como o bike fitting pode ajudar a prevenir lesões?"
    ]

    for pergunta in perguntas:
        fazer_pergunta(vectorstore, llm, pergunta)
        print()

    # Modo interativo
    print("\n" + "=" * 50)
    print("MODO INTERATIVO - Digite suas perguntas (ou 'sair' para encerrar)")
    print("=" * 50)

    while True:
        pergunta = input("\nSua pergunta: ").strip()
        if pergunta.lower() in ['sair', 'exit', 'quit']:
            print("Encerrando o sistema. Até logo!")
            break
        if pergunta:
            fazer_pergunta(vectorstore, llm, pergunta)

if __name__ == "__main__":
    main()
