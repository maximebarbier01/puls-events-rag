"""
Évaluation automatisée du système RAG avec Ragas.

Ragas a besoin d'un LLM "juge" pour noter les réponses (fidélité, pertinence...).
Par défaut il suppose OpenAI ; on lui fournit notre propre modèle Mistral et nos
propres embeddings HuggingFace via les wrappers LangChain de Ragas, pour rester
cohérent avec le reste du projet (pas de dépendance à un second fournisseur).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

from app.rag.chain import answer_question


def load_qa_dataset(path: Path) -> list[dict]:
    """Charger le jeu de questions/réponses de référence (eval/qa_dataset.json)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_ragas_dataset(
    qa_pairs: list[dict], vectorstore: VectorStore, llm: BaseChatModel
) -> EvaluationDataset:
    """Rejouer chaque question sur le VRAI pipeline RAG (pas de raccourci) pour
    obtenir une réponse et des sources fraîches, et construire le dataset Ragas."""
    samples = []

    for pair in qa_pairs:
        result = answer_question(vectorstore, llm, pair["question"])

        samples.append(
            SingleTurnSample(
                user_input=pair["question"],
                response=result.answer,
                retrieved_contexts=[doc.page_content for doc in result.sources],
                reference=pair["reference_answer"],
            )
        )

    return EvaluationDataset(samples=samples)


def get_ragas_metrics() -> list:
    """Les 4 métriques demandées par le sujet : pertinence, fidélité au contexte,
    et couverture documentaire (precision + recall du contexte récupéré)."""
    return [
        Faithfulness(),
        AnswerRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ]


def run_evaluation(
    qa_pairs: list[dict],
    vectorstore: VectorStore,
    llm: BaseChatModel,
    embeddings: Embeddings,
) -> pd.DataFrame:
    """Construire le dataset Ragas et lancer l'évaluation. Retourne un DataFrame
    avec une ligne par question et une colonne par métrique."""
    dataset = build_ragas_dataset(qa_pairs, vectorstore, llm)

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

    result = evaluate(
        dataset=dataset,
        metrics=get_ragas_metrics(),
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return result.to_pandas()
