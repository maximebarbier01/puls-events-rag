"""
Poser une question en rapport avec l'indice FAISS
et obtenir une réponse générée par Mistral.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from app.rag.chain import answer_question, get_llm
from app.vectorstore.build import get_embeddings, load_index


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Emplacement de l'index FAISS sauvegardé
INDEX_PATH = PROJECT_ROOT / "index/events_faiss"

# Question utilisée si aucune question n'est passée dans le terminal
DEFAULT_QUESTION = "Quels concerts à Strasbourg en octobre 2026 ?"


def main() -> None:

    # Chargement des variables d'environnement, notamment la clé API Mistral
    load_dotenv(PROJECT_ROOT / ".env")

    # Récupération de la question passée en argument
    # Sinon utilisation de la question par défaut
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    print(f"Question : {question}\n")

    # Chargement du modèle d'embeddings
    embeddings = get_embeddings()

    # Chargement de l'index FAISS et des documents associés
    vectorstore = load_index(INDEX_PATH, embeddings)

    # Initialisation du modèle Mistral
    llm = get_llm()

    # Recherche des documents pertinents puis génération de la réponse
    result = answer_question(
        vectorstore,
        llm,
        question,
    )

    # Affichage de la réponse générée
    print(f"Réponse :\n{result.answer}\n")

    # Affichage des événements récupérés dans FAISS
    print(f"Sources ({len(result.sources)}) :")

    for doc in result.sources:
        meta = doc.metadata

        print(
            f"  - {meta.get('title_fr')} | "
            f"{meta.get('location_city')} | "
            f"{meta.get('daterange_fr')}"
        )


if __name__ == "__main__":
    main()
