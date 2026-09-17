"""Retrieve relevant events from FAISS and generate an answer with Mistral."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import ChatMistralAI

DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_K = 5

SYSTEM_PROMPT = """Tu es l'assistant culturel de Puls-Events. Tu recommandes des \
événements culturels à partir du contexte fourni ci-dessous, extrait de notre base \
d'événements.

Règles :
- Réponds uniquement à partir des événements listés dans le contexte, ne dis rien qui \
n'y figure pas.
- Cite les informations utiles pour chaque événement mentionné : titre, dates, lieu.
- Si aucun événement du contexte ne correspond réellement à la question, dis-le \
clairement plutôt que d'inventer une réponse.
- Réponds en français, de façon concise et naturelle."""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
    ]
)


@dataclass
class RagAnswer:
    answer: str
    sources: list[Document]


def get_llm(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE) -> ChatMistralAI:
    return ChatMistralAI(model=model, temperature=temperature)


def format_docs(documents: list[Document]) -> str:
    blocks = []
    for doc in documents:
        meta = doc.metadata
        header = f"{meta.get('title_fr', 'Événement')} — {meta.get('daterange_fr', '')} — {meta.get('location_city', '')}"
        source = meta.get("canonicalurl")
        block = f"{header}\n{doc.page_content}"
        if source:
            block += f"\nSource : {source}"
        blocks.append(block)
    return "\n\n".join(blocks)


def answer_question(
    vectorstore: VectorStore, llm: BaseChatModel, question: str, k: int = DEFAULT_K
) -> RagAnswer:
    sources = vectorstore.similarity_search(question, k=k)
    context = format_docs(sources)
    messages = PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)
    return RagAnswer(answer=response.content, sources=sources)
