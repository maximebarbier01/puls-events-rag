"""Récupérer les événements pertinents dans FAISS et générer une réponse avec Mistral."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import ChatMistralAI


# Paramètres par défaut du modèle et de la recherche
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_K = 5


# Instructions données à Mistral pour cadrer la réponse
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


# Construction du prompt envoyé au LLM
PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Contexte :\n{context}\n\nQuestion : {question}"),
    ]
)


# Objet utilisé pour retourner à la fois la réponse et les documents sources
@dataclass
class RagAnswer:
    answer: str
    sources: list[Document]


def get_llm(
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ChatMistralAI:
    """Initialiser le modèle Mistral."""

    return ChatMistralAI(
        model=model,
        temperature=temperature,
    )


def format_docs(documents: list[Document]) -> str:
    """Mettre les documents récupérés dans un format lisible par le LLM."""

    blocks = []

    for doc in documents:
        meta = doc.metadata

        # Informations principales de l'événement
        header = (
            f"{meta.get('title_fr', 'Événement')} — "
            f"{meta.get('daterange_fr', '')} — "
            f"{meta.get('location_city', '')}"
        )

        source = meta.get("canonicalurl")

        # On ajoute le contenu du document sous l'en-tête
        block = f"{header}\n{doc.page_content}"

        # Ajout de l'URL si elle est disponible
        if source:
            block += f"\nSource : {source}"

        blocks.append(block)

    # Séparation des événements par une ligne vide
    return "\n\n".join(blocks)


def answer_question(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    question: str,
    k: int = DEFAULT_K,
) -> RagAnswer:
    """Rechercher les événements proches de la question et générer la réponse."""

    # Recherche des k événements les plus proches dans FAISS
    sources = vectorstore.similarity_search(question, k=k)

    # Transformation des documents en contexte textuel pour le LLM
    context = format_docs(sources)

    # Construction du prompt avec le contexte et la question
    messages = PROMPT.format_messages(
        context=context,
        question=question,
    )

    # Génération de la réponse avec Mistral
    response = llm.invoke(messages)

    return RagAnswer(
        answer=response.content,
        sources=sources,
    )
