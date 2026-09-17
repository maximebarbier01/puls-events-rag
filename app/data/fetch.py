"""
Récupérer les événements publics issus du jeu de données Open Agenda
via l'API Explore d'OpenDataSoft.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

API_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/records"
)
PAGE_SIZE = 100


def _flatten_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    if "location_coordinates" not in df.columns:
        df["location_lat"] = None
        df["location_lon"] = None
        return df

    coords = df.pop("location_coordinates")
    df["location_lat"] = coords.apply(
        lambda c: c.get("lat") if isinstance(c, dict) else None
    )
    df["location_lon"] = coords.apply(
        lambda c: c.get("lon") if isinstance(c, dict) else None
    )
    return df


def fetch_events(
    department: str,
    min_last_date: str | None = None,
    page_size: int = PAGE_SIZE,
) -> pd.DataFrame:
    """Parcourt l'API Explore pour un service, en limitant éventuellement la recherche par date."""
    where = f'location_department="{department}"'
    if min_last_date:
        where += f" AND lastdate_end >= date'{min_last_date}'"

    records: list[dict] = []
    offset = 0
    while True:
        response = requests.get(
            API_URL,
            params={"where": where, "limit": page_size, "offset": offset},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        results = payload["results"]
        records.extend(results)

        offset += page_size
        if not results or offset >= payload["total_count"]:
            break
        time.sleep(0.2)

    df = pd.DataFrame.from_records(records)
    return _flatten_coordinates(df)


def save_events(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
