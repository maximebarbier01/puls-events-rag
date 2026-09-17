"""
Nettoyer, filtrer et structurer les événements bruts d'Open Agenda
avant leur découpage en blocs et leur indexation.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

RECENCY_WINDOW_DAYS = 365
CANCELED_STATUS_ID = 6
FULL_STATUS_ID = 5

# Sources identifiées dans l'exportation « Moselle » qui ne concernent pas des événements culturels :
# France Travail, salons de recrutement, chambres d'agriculture,
# économie numérique, pour les PME, etc.
# Puls-Events étant une plateforme dédiée aux événements culturels,
# ces sources sont exclues même si elles répondent aux critères de filtrage
# (géographiques, de dates et de statut).
EXCLUDED_ORIGINAGENDA_TITLES = {
    "Mes événements France Travail",
    "SolutionsCSE",
    "Chambre d'agriculture de la Moselle",
    "Chambre d'agriculture Grand-Est",
    "2026 : Journées Nationales de l'Agriculture Coopérative  U",
    "Journées Nationales de l'Agriculture 2026",
    "Agenda France Num du numérique pour les TPE PME",
    "Printemps Bio 2026",
    "Ambassadeurs IA",
    "Ensemble, dialoguons - Édition 2026 | Banque de France",
    "Mécénat en Grand Est",
    "Semaine des métiers du tourisme 2026",
}

OUTPUT_COLUMNS = [
    "uid",
    "content",
    "title_fr",
    "firstdate_begin",
    "lastdate_end",
    "daterange_fr",
    "location_city",
    "location_name",
    "location_lat",
    "location_lon",
    "canonicalurl",
    "is_full",
    "keywords_fr",
]


def load_raw_events(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def filter_recent_events(
    df: pd.DataFrame, reference_date: datetime, days: int = RECENCY_WINDOW_DAYS
) -> pd.DataFrame:
    """Conserve les événements remontants jusqu'à `days`, ainsi que tous les événements à venir (sans limite).."""
    cutoff = pd.Timestamp(reference_date - timedelta(days=days))
    cutoff = (
        cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    )
    lastdate_end = pd.to_datetime(df["lastdate_end"], utc=True)
    return df[lastdate_end >= cutoff].copy()


def _status_id(status_raw) -> int | None:
    if not status_raw:
        return None
    try:
        return json.loads(status_raw)["id"]
    except (TypeError, ValueError, KeyError):
        return None


def filter_valid_status(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"].apply(_status_id) != CANCELED_STATUS_ID].copy()


def filter_cultural_events(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["originagenda_title"].isin(EXCLUDED_ORIGINAGENDA_TITLES)].copy()


def clean_html(text) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True)


def _as_text(value) -> str:
    """Normaliser un champ qu'OpenAgenda peut renvoyer sous forme de scalaire ou de liste/tableau."""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _has_value(value) -> bool:
    if isinstance(value, (list, tuple, np.ndarray)):
        return len(value) > 0
    if value is None:
        return False
    if isinstance(value, float) and pd.isna(value):
        return False
    return bool(value)


def build_content_text(row: pd.Series) -> str:
    parts = [row.get("title_fr") or ""]

    dates = row.get("daterange_fr")
    if _has_value(dates):
        parts.append(f"Dates : {_as_text(dates)}")

    location = ", ".join(
        b for b in (row.get("location_name"), row.get("location_city")) if b
    )
    if location:
        parts.append(f"Lieu : {location}")

    description = clean_html(row.get("longdescription_fr")) or (
        row.get("description_fr") or ""
    )
    if description:
        parts.append(description)

    keywords = row.get("keywords_fr")
    if _has_value(keywords):
        parts.append("Mots-clés : " + _as_text(keywords))

    conditions = row.get("conditions_fr")
    if _has_value(conditions):
        parts.append(f"Conditions : {_as_text(conditions)}")

    accessibility = row.get("accessibility_label_fr")
    if _has_value(accessibility):
        parts.append(f"Accessibilité : {_as_text(accessibility)}")

    if _status_id(row.get("status")) == FULL_STATUS_ID:
        parts.append("Cet événement affiche complet.")

    return "\n".join(str(p) for p in parts if p)


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[OUTPUT_COLUMNS].copy()


def preprocess(
    raw_path: Path, output_path: Path, reference_date: datetime | None = None
) -> pd.DataFrame:
    reference_date = reference_date or datetime.now(timezone.utc)

    df = load_raw_events(raw_path)
    df = filter_valid_status(df)
    df = filter_cultural_events(df)
    df = filter_recent_events(df, reference_date)

    df["is_full"] = df["status"].apply(lambda s: _status_id(s) == FULL_STATUS_ID)
    df["content"] = df.apply(build_content_text, axis=1)
    df = df[df["content"].str.strip().astype(bool) & df["title_fr"].notna()]

    df = select_output_columns(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df
