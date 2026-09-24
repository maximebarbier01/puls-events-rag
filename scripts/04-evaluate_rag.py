"""
Évaluer le système RAG avec Ragas sur le jeu de test annoté (eval/qa_dataset.json).

L'évaluation s'exécute toujours sur le snapshot figé des données (eval/snapshot/), pas sur
l'index de production : les événements à venir périment chaque jour, et les scores
resteraient sinon incomparables d'un jour à l'autre. L'index du snapshot est reconstruit
en mémoire à chaque exécution (une à deux minutes de vectorisation locale, sans clé API).

Fait de vrais appels à l'API Mistral (génération des réponses + jugement des
métriques) : coûte quelques dizaines de centimes et prend une minute ou deux, ce
n'est volontairement pas un test pytest (pas lancé à chaque `pytest`).
"""

from pathlib import Path

from dotenv import load_dotenv

from app.rag.chain import get_llm
from app.rag.evaluation import (
    SNAPSHOT_DATE,
    SNAPSHOT_FILENAME,
    build_snapshot_vectorstore,
    load_qa_dataset,
    run_evaluation,
)
from app.vectorstore.build import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = PROJECT_ROOT / "eval/snapshot" / SNAPSHOT_FILENAME
QA_DATASET_PATH = PROJECT_ROOT / "eval/qa_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "eval/results.csv"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    qa_pairs = load_qa_dataset(QA_DATASET_PATH)
    print(f"{len(qa_pairs)} questions chargées depuis {QA_DATASET_PATH}")

    embeddings = get_embeddings()
    print(f"Construction de l'index du snapshot ({SNAPSHOT_DATE.date()})...")
    vectorstore = build_snapshot_vectorstore(SNAPSHOT_PATH, SNAPSHOT_DATE, embeddings)
    llm = get_llm()

    print("Évaluation en cours (appels réels à Mistral)...")
    results_df = run_evaluation(qa_pairs, vectorstore, llm, embeddings)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"Détail par question sauvegardé dans {RESULTS_PATH}")

    print("\nMoyennes par métrique :")
    numeric_columns = results_df.select_dtypes(include="number").columns
    for col in numeric_columns:
        print(f"  {col}: {results_df[col].mean():.3f}")


if __name__ == "__main__":
    main()
