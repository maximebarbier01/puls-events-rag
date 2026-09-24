"""Récupérer les événements pertinents dans FAISS et générer une réponse avec Mistral."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_mistralai import ChatMistralAI


# Paramètres par défaut du modèle et de la recherche
DEFAULT_MODEL = "mistral-small-latest"
DEFAULT_TEMPERATURE = 0.2
DEFAULT_K = 10


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


def _normalize(text: str) -> str:
    """Mettre un texte en minuscules, sans accents ni apostrophes typographiques."""

    text = text.replace("’", "'").lower()
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def get_known_cities(vectorstore: VectorStore) -> list[str]:
    """Lister les villes présentes dans l'index (métadonnée location_city des chunks).

    Une même ville peut être saisie sous plusieurs graphies dans Open Agenda (« Strasbourg »
    et « STRASBOURG ») : on ne garde qu'une graphie par nom normalisé.
    """

    # Le docstore est propre à FAISS : pour un autre vector store, on ne filtre simplement pas.
    documents = getattr(getattr(vectorstore, "docstore", None), "_dict", {})
    by_key: dict[str, str] = {}
    for doc in documents.values():
        city = doc.metadata.get("location_city")
        if city:
            by_key.setdefault(_normalize(city), city)
    return sorted(by_key.values())


# « Grand Est » ne doit pas être lu comme la commune de Grand (Vosges).
REGION_PATTERN = re.compile(r"grand[\s-]+est")


def detect_city(question: str, cities: list[str]) -> str | None:
    """Retrouver la ville citée dans la question, ou None s'il n'y en a pas.

    La recherche sémantique seule ne sait pas restreindre à « Colmar » : les embeddings
    rapprochent les sens, pas les noms de lieux. On repère donc la ville par correspondance
    de mots entiers (sans tenir compte de la casse ni des accents), en essayant les noms les
    plus longs d'abord pour que « Saint-Dizier » l'emporte sur un nom plus court qu'il contient.
    Le nom de la région (« Grand Est ») est retiré de la question avant la recherche.
    """

    normalized_question = REGION_PATTERN.sub(" ", _normalize(question))
    for city in sorted(cities, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(_normalize(city)) + r"(?!\w)"
        if re.search(pattern, normalized_question):
            return city
    return None


def retrieve(vectorstore: VectorStore, question: str, k: int = DEFAULT_K) -> list[Document]:
    """Rechercher les k événements les plus proches, restreints à la ville citée s'il y en a une."""

    city = detect_city(question, get_known_cities(vectorstore))
    if city:
        city_key = _normalize(city)
        # Le filtre compare les graphies normalisées : « Strasbourg » et « STRASBOURG »
        # désignent la même ville. fetch_k = taille de l'index, car FAISS applique le filtre
        # APRÈS avoir récupéré les fetch_k plus proches voisins.
        filtered = vectorstore.similarity_search(
            question,
            k=k,
            filter=lambda meta: _normalize(meta.get("location_city") or "") == city_key,
            fetch_k=max(vectorstore.index.ntotal, k),
        )
        if filtered:
            return filtered

    # Pas de ville dans la question (ou aucun événement dans cette ville) : recherche simple.
    return vectorstore.similarity_search(question, k=k)


def answer_question(
    vectorstore: VectorStore,
    llm: BaseChatModel,
    question: str,
    k: int = DEFAULT_K,
) -> RagAnswer:
    """Rechercher les événements proches de la question et générer la réponse."""

    # Recherche des k événements les plus proches dans FAISS (filtrés sur la ville citée)
    sources = retrieve(vectorstore, question, k=k)

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
