# Ingestão e Busca Semântica com LangChain e Postgres

Desafio do módulo **Introdução ao LangChain** (MBA em Engenharia de Software com IA — Full Cycle).

O projeto lê um PDF, divide o texto em chunks, gera embeddings e grava tudo no **PostgreSQL + pgVector**.
Depois, um chat de linha de comando responde perguntas **somente com base no conteúdo do PDF**.

## Como funciona

**Ingestão** (`src/ingest.py`)

1. Carrega o `document.pdf` com `PyPDFLoader`.
2. Divide em chunks de **1000 caracteres com overlap de 150** (`RecursiveCharacterTextSplitter`).
3. Gera o embedding de cada chunk e grava no pgVector (`PGVector`).
   Os chunks recebem IDs fixos (`doc-0`, `doc-1`, ...), então rodar a ingestão de novo atualiza em vez de duplicar.

**Busca** (`src/search.py`) — ao receber uma pergunta:

1. Vetoriza a pergunta e busca os **10 chunks mais relevantes** com `similarity_search_with_score(query, k=10)`.
2. Concatena os chunks no `CONTEXTO` do prompt definido no desafio.
3. Chama a LLM e devolve a resposta.

**Chat** (`src/chat.py`) — loop no terminal que lê a pergunta e imprime a resposta. Digite `sair` para encerrar.

**Provedores** (`src/providers.py`) — funciona com **Gemini** ou **OpenAI**, escolhido pela variável `LLM_PROVIDER`.

| Provedor | Embeddings (padrão) | LLM (padrão) |
|---|---|---|
| Gemini | `models/gemini-embedding-001` | `gemini-3.5-flash-lite` |
| OpenAI | `text-embedding-3-small` | `gpt-5-nano` |

Os modelos podem ser trocados no `.env`. **Atenção:** cada modelo de embedding gera vetores com dimensão própria.
Se trocar o modelo de embedding (ou o provedor) depois da primeira ingestão, apague a collection
(ou use outro `PG_VECTOR_COLLECTION_NAME`, ou `docker compose down -v`) e rode a ingestão de novo.

## Pré-requisitos

- Python 3.10+
- Docker e Docker Compose
- API key da Google ([Google AI Studio](https://aistudio.google.com/apikey)) **ou** da OpenAI ([platform.openai.com](https://platform.openai.com/api-keys))

## Execução

### 1. Clonar e configurar o ambiente virtual

```bash
git clone https://github.com/GabrielSelim/mba-ia-desafio-ingestao-busca.git
cd mba-ia-desafio-ingestao-busca

python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows (PowerShell/cmd)

pip install -r requirements.txt
```

### 2. Configurar as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` e preencha a API key do provedor escolhido:

```env
LLM_PROVIDER=gemini                # ou openai
GOOGLE_API_KEY=sua-chave-google    # se usar Gemini
OPENAI_API_KEY=sua-chave-openai    # se usar OpenAI
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PG_VECTOR_COLLECTION_NAME=documento_pdf
PDF_PATH=document.pdf
```

### 3. Subir o banco de dados

```bash
docker compose up -d
```

Sobe o PostgreSQL com pgVector e cria a extensão `vector` automaticamente.

### 4. Executar a ingestão do PDF

```bash
python src/ingest.py
```

Saída esperada (o número de chunks depende do PDF):

```
Lendo PDF: .../document.pdf
Ingestão concluída: N página(s), M chunk(s) gravados no PGVector.
```

### 5. Rodar o chat

```bash
python src/chat.py
```

Exemplo:

```
Faça sua pergunta:

PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

---

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.

---

PERGUNTA: sair
Até logo!
```

## Estrutura

```
├── docker-compose.yml    # PostgreSQL + pgVector
├── requirements.txt      # Dependências
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Ingestão do PDF
│   ├── search.py         # Busca semântica + prompt + LLM
│   ├── chat.py           # CLI de interação
│   └── providers.py      # Embeddings/LLM/vector store por provedor
├── document.pdf          # PDF para ingestão
└── README.md
```

## Problemas comuns

- **`different vector dimensions` / erro de dimensão:** o modelo de embedding mudou depois da ingestão. Rode
  `docker compose down -v && docker compose up -d` e a ingestão de novo (ou troque `PG_VECTOR_COLLECTION_NAME`).
- **`connection refused` na porta 5432:** o banco não está de pé — confira com `docker compose ps`.
- **Erro de cota/429 no Gemini:** o plano gratuito tem limite de requisições por minuto; aguarde e tente de novo.
