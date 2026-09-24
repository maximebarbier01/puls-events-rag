"""
Construit eval/qa_dataset.json : questions + réponses de référence.

Chaque réponse de référence est déduite MÉCANIQUEMENT du snapshot figé (filtre pandas sur
la ville, le type d'événement, la période), sans jamais passer par le LLM : les références
sont donc indépendantes du système évalué. Ce script est gardé pour pouvoir rejouer et
justifier chaque référence.

    poetry run python eval/build_references.py
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.data.preprocessing import preprocess
from app.rag.evaluation import SNAPSHOT_DATE, SNAPSHOT_FILENAME

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = PROJECT_ROOT / "eval/snapshot" / SNAPSHOT_FILENAME
OUTPUT_PATH = PROJECT_ROOT / "eval/qa_dataset.json"


def _utc(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def load_snapshot_events() -> pd.DataFrame:
    """Événements du snapshot après le même nettoyage que la production, à la date figée."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        df = preprocess(SNAPSHOT_PATH, Path(tmp_dir) / "events.parquet", reference_date=SNAPSHOT_DATE)
    df = df.reset_index(drop=True)
    df["_begin"] = pd.to_datetime(df["firstdate_begin"], utc=True, errors="coerce")
    df["_end"] = pd.to_datetime(df["lastdate_end"], utc=True, errors="coerce")
    keywords = df["keywords_fr"].apply(lambda k: " ".join(k) if k is not None else "")
    # Le type d'événement se lit dans le titre et les mots-clés (pas dans toute la description,
    # où « concert » peut apparaître à propos d'un tout autre événement).
    df["_label"] = (df["title_fr"].fillna("") + " " + keywords).str.lower()
    return df


def active_between(df: pd.DataFrame, start: datetime, end: datetime) -> pd.Series:
    """Événements ayant au moins une journée dans [start, end]."""
    return (df["_begin"] <= end) & (df["_end"] >= start)


def starts_on(df: pd.DataFrame, day: datetime) -> pd.Series:
    return df["_begin"].dt.date == day.date()


def format_event(row: pd.Series) -> str:
    return f"- {row['title_fr']} — {row['daterange_fr']} — {row['location_name']}, {row['location_city']}"


def reference_from(df: pd.DataFrame, mask: pd.Series, intro: str, none_text: str) -> str:
    matches = df[mask].sort_values("_begin")
    if matches.empty:
        return none_text
    return intro + "\n" + "\n".join(format_event(row) for _, row in matches.iterrows())


OCTOBER = (_utc(2026, 10, 1), _utc(2026, 10, 31, 23, 59))


def build_dataset(df: pd.DataFrame) -> list[dict]:
    def city(name: str) -> pd.Series:
        # Open Agenda mélange les graphies (« Strasbourg » / « STRASBOURG ») : comparaison sans casse.
        return df["location_city"].str.lower() == name.lower()

    def label(*words: str) -> pd.Series:
        # Mots entiers, singulier ou pluriel : sans \b, « conte » matcherait « contemporaine ».
        pattern = r"\b(?:" + "|".join(words) + r")s?\b"
        return df["_label"].str.contains(pattern, regex=True)

    in_october = active_between(df, *OCTOBER)
    free = df["content"].str.lower().str.contains("gratuit")
    wheelchair = df["content"].str.lower().str.contains("handicap moteur")

    specs = [
        (
            "Quels concerts à Strasbourg en octobre 2026 ?",
            city("Strasbourg") & label("concert") & in_october,
            "Voici les concerts à Strasbourg en octobre 2026 :",
            "Aucun concert à Strasbourg en octobre 2026 n'est présent dans les données.",
        ),
        (
            "Quels concerts à Nancy en octobre 2026 ?",
            city("Nancy") & label("concert") & in_october,
            "Voici les concerts à Nancy en octobre 2026 :",
            "Aucun concert à Nancy en octobre 2026 n'est présent dans les données.",
        ),
        (
            "Quelles expositions à Bar-le-Duc en octobre 2026 ?",
            city("Bar-le-Duc") & label("exposition") & in_october,
            "Voici les expositions à Bar-le-Duc en octobre 2026 :",
            "Aucune exposition à Bar-le-Duc en octobre 2026 n'est présente dans les données.",
        ),
        (
            "Quelles expositions à Reims en octobre 2026 ?",
            city("Reims") & label("exposition") & in_october,
            "Voici les expositions à Reims en octobre 2026 :",
            "Aucune exposition à Reims en octobre 2026 n'est présente dans les données.",
        ),
        (
            "Quels ateliers à Colmar en octobre 2026 ?",
            city("Colmar") & label("atelier") & in_october,
            "Voici les ateliers à Colmar en octobre 2026 :",
            "Aucun atelier à Colmar en octobre 2026 n'est présent dans les données.",
        ),
        (
            "Quelles visites à Colmar en octobre 2026 ?",
            city("Colmar") & label("visite") & in_october,
            "Voici les visites à Colmar en octobre 2026 :",
            "Aucune visite à Colmar en octobre 2026 n'est présente dans les données.",
        ),
        (
            "Quels contes à Colmar en octobre 2026 ?",
            city("Colmar") & label("conte") & in_october,
            "Voici les contes à Colmar en octobre 2026 :",
            "Aucun conte à Colmar en octobre 2026 n'est présent dans les données.",
        ),
        (
            "Quelles conférences à Reims en octobre 2026 ?",
            city("Reims") & label("conférence") & in_october,
            "Voici les conférences à Reims en octobre 2026 :",
            "Aucune conférence à Reims en octobre 2026 n'est présente dans les données.",
        ),
        (
            "Y a-t-il un film à voir à Mulhouse en octobre 2026 ?",
            city("Mulhouse") & label("film") & in_october,
            "Voici les films à Mulhouse en octobre 2026 :",
            "Aucun film à Mulhouse en octobre 2026 n'est présent dans les données.",
        ),
        (
            "Quels événements gratuits à Strasbourg le samedi 10 octobre 2026 ?",
            city("Strasbourg") & free & starts_on(df, _utc(2026, 10, 10)),
            "Voici les événements gratuits à Strasbourg le samedi 10 octobre 2026 :",
            "Aucun événement gratuit à Strasbourg le samedi 10 octobre 2026 n'est présent dans les données.",
        ),
        (
            "Quels événements accessibles aux personnes à mobilité réduite à Strasbourg le samedi 3 octobre 2026 ?",
            city("Strasbourg") & wheelchair & active_between(df, _utc(2026, 10, 3), _utc(2026, 10, 3, 23, 59)),
            "Voici les événements accessibles aux personnes à mobilité réduite à Strasbourg le samedi 3 octobre 2026 :",
            "Aucun événement accessible aux personnes à mobilité réduite à Strasbourg le samedi 3 octobre 2026 n'est présent dans les données.",
        ),
        (
            "Quels concerts à Metz le samedi 3 octobre 2026 ?",
            city("Metz") & label("concert") & starts_on(df, _utc(2026, 10, 3)),
            "Voici les concerts à Metz le samedi 3 octobre 2026 :",
            "Aucun concert à Metz le samedi 3 octobre 2026 n'est présent dans les données.",
        ),
    ]

    dataset = []
    for question, mask, intro, none_text in specs:
        dataset.append(
            {
                "question": question,
                "reference_answer": reference_from(df, mask, intro, none_text),
                "n_expected_events": int(mask.sum()),
            }
        )

    # Cas limites dont la bonne réponse est un refus, indépendamment des données.
    dataset.append(
        {
            "question": "Quels concerts à Paris en octobre 2026 ?",
            "reference_answer": "Aucun concert à Paris n'est présent dans les données : la base ne couvre que la région Grand Est.",
            "n_expected_events": 0,
        }
    )
    dataset.append(
        {
            "question": "Quelle est la recette du brownie au chocolat ?",
            "reference_answer": "Aucun événement dans les données ne propose de recette de brownie au chocolat : la question est hors du périmètre des événements culturels.",
            "n_expected_events": 0,
        }
    )
    return dataset


def main() -> None:
    df = load_snapshot_events()
    dataset = build_dataset(df)
    OUTPUT_PATH.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(dataset)} questions écrites dans {OUTPUT_PATH} (snapshot du {SNAPSHOT_DATE.date()}, {len(df)} événements)")
    for item in dataset:
        print(f"\n[{item['n_expected_events']} événement(s)] {item['question']}")
        print(item["reference_answer"])


if __name__ == "__main__":
    main()
