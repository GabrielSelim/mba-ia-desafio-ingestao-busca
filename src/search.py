from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

from providers import get_llm, get_vector_store

TOP_K = 10

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


def search_prompt(question=None):
    """Monta a chain de RAG: vetoriza a pergunta, busca k=10 no PGVector, preenche o prompt e chama a LLM.

    Sem `question`, retorna a chain (use `chain.invoke(pergunta)`); com `question`, retorna a resposta.
    Em caso de erro de configuração, retorna None.
    """
    try:
        store = get_vector_store()
        llm = get_llm()
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao inicializar a busca: {exc}")
        return None

    prompt = PromptTemplate(input_variables=["contexto", "pergunta"], template=PROMPT_TEMPLATE)

    def build_inputs(pergunta: str) -> dict:
        # Vetoriza a pergunta e traz os 10 chunks mais próximos do banco vetorial.
        results = store.similarity_search_with_score(pergunta, k=TOP_K)
        contexto = "\n\n".join(doc.page_content for doc, _score in results)
        return {"contexto": contexto, "pergunta": pergunta}

    chain = RunnableLambda(build_inputs) | prompt | llm | StrOutputParser()

    if question is not None:
        return chain.invoke(question)
    return chain
