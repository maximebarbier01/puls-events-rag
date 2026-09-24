"""
Point d'entrée de l'API FastAPI (référencé dans pyproject.toml : app.main:app).

Lancement local :
    poetry run uvicorn app.main:app --reload
Documentation Swagger générée automatiquement : http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import INDEX_PATH, INTERIM_DATA_PATH, RAW_DATA_PATH, load_env
from app.rag.chain import get_llm
from app.vectorstore.build import get_embeddings, load_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charger le modèle d'embedding, l'index FAISS et le client Mistral une seule
    fois au démarrage du serveur, et les garder en mémoire (app.state) pendant toute
    la durée de vie du processus.

    C'est ce qui répond au point de vigilance du sujet : "évitez de relancer toute
    la chaîne à chaque appel" — sans ça, chaque question rechargerait le modèle
    d'embedding et l'index à chaque requête, ce qui serait beaucoup trop lent.
    """
    load_env()

    # Chemins réutilisés par la route /rebuild (app/api/routes.py) : posés ici plutôt
    # qu'importés en dur dans les routes, pour rester redirigeables dans les tests.
    app.state.raw_path = RAW_DATA_PATH
    app.state.interim_path = INTERIM_DATA_PATH
    app.state.index_path = INDEX_PATH

    app.state.embeddings = get_embeddings()
    app.state.vectorstore = load_index(INDEX_PATH, app.state.embeddings)
    app.state.llm = get_llm()

    yield

    # Rien à nettoyer explicitement à l'arrêt (pas de connexion réseau/DB à fermer).


app = FastAPI(
    title="Puls-Events RAG API",
    description=(
        "API exposant le système RAG (LangChain + FAISS + Mistral) "
        "de recommandation d'événements culturels à venir dans le Grand Est."
    ),
    version="0.1.1",
    lifespan=lifespan,
)

app.include_router(router)
