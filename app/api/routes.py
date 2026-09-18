"""
Routes de l'API : /ask (poser une question) et /rebuild (reconstruire l'index).

La logique métier (RAG, preprocessing, indexation) n'est pas réécrite ici : on
réutilise app.rag.chain et app.vectorstore.build, comme demandé par le sujet
("séparez bien votre logique métier de votre code d'API").
"""

import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from app.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    MetadataResponse,
    RebuildResponse,
    SourceItem,
)
from app.core.config import get_rebuild_token
from app.data.preprocessing import preprocess
from app.rag.chain import DEFAULT_K, DEFAULT_MODEL, answer_question
from app.vectorstore.build import (
    EMBEDDING_MODEL_NAME,
    build_index,
    chunk_documents,
    save_index,
    to_documents,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# --- Dépendances : on lit les objets chargés une seule fois au démarrage de l'app
# (voir app/main.py, lifespan), jamais rechargés à chaque requête. C'est le point de
# vigilance "évitez de relancer toute la chaîne à chaque appel" du sujet. Passer par
# des fonctions Depends() permet aussi à tests/api_test.py de les remplacer par de
# faux objets (FakeEmbeddings, FakeListChatModel) sans toucher au code des routes.


def get_vectorstore(request: Request) -> VectorStore:
    """Récupérer l'index FAISS chargé en mémoire au démarrage de l'API."""
    return request.app.state.vectorstore


def get_llm(request: Request) -> BaseChatModel:
    """Récupérer le client Mistral instancié au démarrage de l'API."""
    return request.app.state.llm


def verify_rebuild_token(x_rebuild_token: str | None = Header(default=None)) -> None:
    """Protéger /rebuild avec un jeton partagé (header X-Rebuild-Token).

    Comportement volontairement "fail-closed" : si la variable d'environnement
    REBUILD_TOKEN n'est pas configurée côté serveur, on refuse toute requête plutôt
    que de laisser l'endpoint ouvert par défaut (point de vigilance du sujet :
    "protégez les endpoints sensibles").
    """
    expected_token = get_rebuild_token()

    if not expected_token:
        raise HTTPException(
            status_code=403,
            detail="/rebuild est désactivé : la variable REBUILD_TOKEN n'est pas configurée.",
        )

    if x_rebuild_token != expected_token:
        raise HTTPException(
            status_code=401, detail="Jeton X-Rebuild-Token invalide ou manquant."
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérifier que l'API est opérationnelle",
    description=(
        "Endpoint de liveness/readiness : confirme que l'index FAISS et le client "
        "Mistral sont bien chargés en mémoire, sans faire aucun appel réseau ni "
        "génération. N'attend aucun paramètre."
    ),
)
def health(
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: BaseChatModel = Depends(get_llm),
) -> HealthResponse:
    return HealthResponse(
        status="ok",
        vectorstore_loaded=vectorstore is not None,
        llm_configured=llm is not None,
    )


@router.get(
    "/metadata",
    response_model=MetadataResponse,
    summary="Informations sur le système RAG en place",
    description=(
        "Donne aux équipes métier une vue sur ce qui alimente /ask (modèle "
        "d'embedding, modèle de génération, taille actuelle de l'index, nombre de "
        "documents récupérés par question) sans avoir à lire le code. N'attend "
        "aucun paramètre."
    ),
)
def metadata(vectorstore: VectorStore = Depends(get_vectorstore)) -> MetadataResponse:
    # vectorstore.index.ntotal reflète toujours l'état courant de l'index, y compris
    # après un /rebuild (pas besoin de le recalculer ni de le stocker à part).
    return MetadataResponse(
        embedding_model=EMBEDDING_MODEL_NAME,
        llm_model=DEFAULT_MODEL,
        nb_chunks_indexed=vectorstore.index.ntotal,
        default_top_k=DEFAULT_K,
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Poser une question à PulsEvents",
    description=(
        "Récupère les événements les plus pertinents dans l'index FAISS, puis "
        "demande à Mistral de générer une réponse en français à partir de ce "
        'contexte. Attend un corps JSON `{"question": "..."}` ; la question ne '
        "peut pas être vide."
    ),
)
def ask(
    payload: AskRequest,
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: BaseChatModel = Depends(get_llm),
) -> AskResponse:
    # La validation "question non vide" est déjà faite par Pydantic (AskRequest) :
    # une requête avec une question vide renvoie automatiquement une 422 avant
    # d'arriver ici.
    try:
        result = answer_question(vectorstore, llm, payload.question)
    except Exception:
        # On ne renvoie jamais la trace brute au client (elle pourrait contenir des
        # détails internes) : on logge côté serveur et on renvoie un message générique.
        logger.exception(
            "Échec de la génération de réponse pour la question : %s", payload.question
        )
        raise HTTPException(
            status_code=500,
            detail="Le système RAG n'a pas pu générer de réponse pour le moment.",
        )

    sources = [
        SourceItem(
            title=doc.metadata.get("title_fr"),
            dates=doc.metadata.get("daterange_fr"),
            city=doc.metadata.get("location_city"),
            url=doc.metadata.get("canonicalurl"),
        )
        for doc in result.sources
    ]

    return AskResponse(answer=result.answer, sources=sources)


@router.post(
    "/rebuild",
    response_model=RebuildResponse,
    summary="Reconstruire l'index vectoriel",
    description=(
        "Relit les données brutes (data/raw), relance le nettoyage puis reconstruit "
        "l'index FAISS avec le modèle d'embedding déjà chargé (pas de rechargement du "
        "modèle, juste ré-indexation). Ne va pas rechercher de nouvelles données sur "
        "Open Agenda : ça reste une étape manuelle séparée "
        "(scripts/00-fetch_openagenda.py), pour ne pas bloquer une requête HTTP sur un "
        "appel réseau externe potentiellement long. Protégé par le header "
        "`X-Rebuild-Token`."
    ),
    dependencies=[Depends(verify_rebuild_token)],
)
def rebuild(request: Request) -> RebuildResponse:
    # Les chemins et l'embeddings viennent de app.state (posés au démarrage par le
    # lifespan de app/main.py) plutôt que d'être importés en dur ici : ça permet aux
    # tests fonctionnels (tests/api_test.py) de les rediriger vers des fichiers
    # temporaires, sans jamais toucher aux vraies données du projet.
    state = request.app.state

    # Étape 1 : re-nettoyer les données brutes déjà présentes sur disque.
    df = preprocess(state.raw_path, state.interim_path)

    # Étape 2 : reconstruire l'index avec les embeddings déjà en mémoire (on ne
    # recharge pas le modèle HuggingFace à chaque rebuild, seulement les vecteurs).
    documents = to_documents(df)
    chunks = chunk_documents(documents)
    new_vectorstore = build_index(chunks, state.embeddings)

    # Étape 3 : on remplace l'index utilisé par /ask immédiatement (pas besoin de
    # redémarrer le serveur), et on le sauvegarde aussi sur disque pour la prochaine
    # fois.
    state.vectorstore = new_vectorstore
    save_index(new_vectorstore, state.index_path)

    return RebuildResponse(
        message="Index reconstruit avec succès.",
        nb_documents=len(df),
        nb_chunks=len(chunks),
    )
