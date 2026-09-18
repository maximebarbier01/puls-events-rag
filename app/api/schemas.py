"""
Modèles Pydantic pour les requêtes et réponses de l'API.

Ces modèles servent aussi à générer automatiquement la documentation
Swagger (/docs) : chaque champ documenté ici apparaît dans l'interface interactive.
"""

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Corps attendu par POST /ask."""

    # min_length=1 + le validateur ci-dessous empêchent une question vide ou
    # composée uniquement d'espaces (point de vigilance du sujet).
    question: str = Field(
        ...,
        min_length=1,
        description="Question posée par l'utilisateur, en français.",
        examples=["Quels concerts de musique à Metz samedi 3 octobre 2026 ?"],
    )

    @field_validator("question")
    @classmethod
    def question_non_vide(cls, value: str) -> str:
        """Rejeter une question qui ne contient que des espaces."""
        if not value.strip():
            raise ValueError("La question ne peut pas être vide.")
        return value


class SourceItem(BaseModel):
    """Un événement source utilisé pour construire la réponse.

    On ne renvoie qu'un sous-ensemble des métadonnées internes (pas tout
    doc.metadata brut) : juste ce qui est utile côté client pour afficher/citer
    l'événement.
    """

    title: str | None = None
    dates: str | None = None
    city: str | None = None
    url: str | None = None


class AskResponse(BaseModel):
    """Réponse renvoyée par POST /ask."""

    answer: str = Field(..., description="Réponse générée par le modèle Mistral.")
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="Événements récupérés dans l'index FAISS et utilisés comme contexte.",
    )


class RebuildResponse(BaseModel):
    """Réponse renvoyée par POST /rebuild."""

    message: str
    nb_documents: int = Field(..., description="Nombre d'événements après nettoyage.")
    nb_chunks: int = Field(..., description="Nombre de chunks indexés dans FAISS.")


class HealthResponse(BaseModel):
    """Réponse renvoyée par GET /health."""

    status: str = Field(..., description="'ok' si l'API est opérationnelle.")
    vectorstore_loaded: bool = Field(..., description="L'index FAISS est chargé en mémoire.")
    llm_configured: bool = Field(..., description="Le client Mistral est instancié.")


class MetadataResponse(BaseModel):
    """Réponse renvoyée par GET /metadata.

    Informations sur le système RAG en cours d'exécution, utiles aux équipes
    métier pour savoir sur quelles données/modèles reposent les réponses de
    /ask, sans avoir à lire le code.
    """

    embedding_model: str = Field(..., description="Modèle utilisé pour vectoriser événements et questions.")
    llm_model: str = Field(..., description="Modèle Mistral utilisé pour générer les réponses.")
    nb_chunks_indexed: int = Field(..., description="Nombre de chunks actuellement dans l'index FAISS.")
    default_top_k: int = Field(..., description="Nombre de documents récupérés par défaut à chaque question.")
