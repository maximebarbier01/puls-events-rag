"""Ask a question against the FAISS index and get a Mistral-generated answer."""
import sys
from pathlib import Path

from dotenv import load_dotenv

from app.rag.chain import answer_question, get_llm
from app.vectorstore.build import get_embeddings, load_index

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "index/events_faiss"
DEFAULT_QUESTION = "Quels concerts de musique à Metz ce week-end ?"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    print(f"Question : {question}\n")

    embeddings = get_embeddings()
    vectorstore = load_index(INDEX_PATH, embeddings)
    llm = get_llm()

    result = answer_question(vectorstore, llm, question)

    print(f"Réponse :\n{result.answer}\n")
    print(f"Sources ({len(result.sources)}) :")
    for doc in result.sources:
        meta = doc.metadata
        print(f"  - {meta.get('title_fr')} | {meta.get('location_city')} | {meta.get('daterange_fr')}")


if __name__ == "__main__":
    main()
