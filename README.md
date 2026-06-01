# RAG - Converse com PDF

Trabalho da disciplina **(2026-1) FIC - Introdução a Agentes Inteligentes**

## Descrição

Sistema RAG (Retrieval-Augmented Generation) que permite fazer perguntas sobre o conteúdo de um PDF utilizando LangChain, ChromaDB e Ollama.

### Funcionalidades:
- Carrega e processa arquivos PDF
- Divide o texto em chunks para indexação
- Cria embeddings e armazena no ChromaDB
- Responde perguntas baseando-se apenas no conteúdo do PDF

## Pré-requisitos

### Instalar Ollama
```bash
# macOS
brew install ollama

# ou baixe em: https://ollama.ai
```

### Baixar modelos necessários
```bash
ollama pull llama3
ollama pull nomic-embed-text
```

## Como Usar

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar
```bash
python demo.py
```

## Exemplo de Saída

```
==================================================
SISTEMA RAG GRATUITO COM OLLAMA
==================================================

Carregando PDF...
PDF carregado com X página(s)
Dividindo texto em chunks...
Texto dividido em Y chunks
Criando embeddings e indexando no ChromaDB...
Banco vetorial criado!

==================================================
RESPONDENDO PERGUNTAS SOBRE O PDF
==================================================

PERGUNTA: O que é bike fitting?
--------------------------------------------------
RESPOSTA: Bike fitting é um processo de ajuste...
```

## Tecnologias

- Python 3.x
- LangChain
- ChromaDB (banco vetorial)
- Ollama (LLM local gratuito)
- PyPDF (leitura de PDFs)

## PDF Utilizado

- `bikefitting.pdf` - Documento sobre Bike Fitting
