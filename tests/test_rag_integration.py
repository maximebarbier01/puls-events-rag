"""
Tests d'intégration avec le VRAI modèle Mistral et le VRAI index FAISS — contrairement
au reste de la suite, ces tests coûtent quelques centimes et ont besoin d'un accès
réseau. Ils sont donc désactivés automatiquement si MISTRAL_API_KEY n'est pas
configurée (poetry run pytest les passe silencieusement en "skipped"), et il vaut
mieux les lancer à part plutôt qu'à chaque `pytest` :

    poetry run pytest tests/test_rag_integration.py -v -s

Objectif : vérifier que le VRAI modèle respecte bien la consigne du prompt système
(app/rag/chain.py::SYSTEM_PROMPT) qui lui interdit d'inventer une réponse quand le
contexte récupéré ne correspond pas à la question. Un faux modèle (FakeListChatModel,
utilisé partout ailleurs dans les tests) ne peut pas vérifier ça : il renvoie toujours
la réponse qu'on lui a scriptée, peu importe la question posée.
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from app.rag.chain import answer_question, get_llm
from app.vectorstore.build import get_embeddings, load_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "index/events_faiss"

load_dotenv(PROJECT_ROOT / ".env")

# Mots qui trahissent une hallucination : si le modèle se met à lister de vrais
# ingrédients de brownie, c'est qu'il a inventé une réponse au lieu de refuser.
BROWNIE_RECIPE_WORDS = ["farine", "beurre", "chocolat fondu", "sucre", "œufs", "oeufs"]

# Tournures attendues quand le modèle refuse correctement de répondre. Liste large
# plutôt qu'exhaustive : la formulation exacte d'un LLM varie d'un appel à l'autre
# (vu en pratique : "Aucun concert prévu à Lyon dans le contexte fourni." — d'où
# l'ajout de "aucun concert" et "dans le contexte fourni" après un premier faux négatif).
REFUSAL_HINTS = [
    "ne correspond",
    "aucun événement",
    "aucun concert",
    "pas d'événement",
    "ne dispose pas",
    "je ne peux pas",
    "hors de mon contexte",
    "pas dans le contexte",
    "dans le contexte fourni",
    "n'y a pas",
    "ne figure pas",
    "pas mentionné",
    "aucune information",
]

pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"),
    reason="MISTRAL_API_KEY non configurée — test d'intégration ignoré (voir .env.example).",
)


@pytest.fixture(scope="module")
def real_vectorstore():
    embeddings = get_embeddings()
    return load_index(INDEX_PATH, embeddings)


@pytest.fixture(scope="module")
def real_llm():
    return get_llm()


def test_model_refuses_to_invent_a_brownie_recipe(real_vectorstore, real_llm):
    """Une question totalement hors-sujet (recette de cuisine) ne doit pas donner
    lieu à une vraie recette, même si le contexte récupéré est sans rapport."""
    result = answer_question(real_vectorstore, real_llm, "Quelle est la recette du brownie au chocolat ?")

    print(f"\nRéponse obtenue :\n{result.answer}\n")

    answer_lower = result.answer.lower()
    recipe_words_found = [w for w in BROWNIE_RECIPE_WORDS if w in answer_lower]

    assert not recipe_words_found, (
        f"Le modèle a halluciné une vraie recette (mots trouvés : {recipe_words_found}) "
        f"au lieu de refuser. Réponse complète : {result.answer}"
    )


def test_model_does_not_invent_an_event_it_does_not_have(real_vectorstore, real_llm):
    """Question sur un événement hors périmètre (ville hors Grand Est) : le modèle ne
    doit pas prétendre avoir un vrai concert à Lyon alors que le corpus ne couvre que
    le Grand Est."""
    result = answer_question(
        real_vectorstore, real_llm, "Quels concerts sont prévus à Lyon ce week-end ?"
    )

    print(f"\nRéponse obtenue :\n{result.answer}\n")

    answer_lower = result.answer.lower()
    has_refusal = any(hint in answer_lower for hint in REFUSAL_HINTS)

    # Assertion volontairement informative plutôt que stricte : la formulation d'un
    # LLM varie d'un appel à l'autre. On alerte si aucune tournure de refus n'est
    # détectée, à vérifier manuellement dans la sortie affichée ci-dessus (pytest -s).
    if not has_refusal:
        pytest.fail(
            "Aucune tournure de refus détectée — vérifiez manuellement si le modèle "
            f"a inventé un événement à Lyon. Réponse complète : {result.answer}"
        )
