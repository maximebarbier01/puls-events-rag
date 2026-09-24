"""Unit tests for app.data.preprocessing."""
import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from app.data.preprocessing import (
    EXCLUDED_ORIGINAGENDA_PREFIXES,
    EXCLUDED_ORIGINAGENDA_TITLES,
    build_content_text,
    clean_html,
    filter_cultural_events,
    filter_upcoming_events,
    filter_valid_status,
    preprocess,
    select_output_columns,
)

REFERENCE_DATE = datetime(2026, 9, 17, tzinfo=timezone.utc)


def _status(status_id: int, label_fr: str) -> str:
    return json.dumps({"id": status_id, "label": {"fr": label_fr}})


@pytest.fixture
def sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {  # kept: upcoming, scheduled, cultural
                "uid": "1",
                "title_fr": "Concert au Centre Pompidou-Metz",
                "description_fr": "Un concert.",
                "longdescription_fr": "<p>Un concert <br>exceptionnel.</p>",
                "keywords_fr": ["musique", "concert"],
                "conditions_fr": "Gratuit",
                "accessibility_label_fr": None,
                "daterange_fr": "Samedi 20 septembre",
                "firstdate_begin": pd.Timestamp("2026-09-20", tz="UTC"),
                "lastdate_end": pd.Timestamp("2026-09-20", tz="UTC"),
                "location_name": "Centre Pompidou-Metz",
                "location_city": "Metz",
                "location_lat": 49.11,
                "location_lon": 6.18,
                "canonicalurl": "https://openagenda.com/culture/events/1",
                "originagenda_title": "Centre Pompidou-Metz",
                "status": _status(1, "Programmé"),
            },
            {  # kept: within the 1-year history window
                "uid": "2",
                "title_fr": "Exposition passée",
                "description_fr": "desc",
                "longdescription_fr": None,
                "keywords_fr": None,
                "conditions_fr": None,
                "accessibility_label_fr": "handicap moteur",
                "daterange_fr": "Il y a 6 mois",
                "firstdate_begin": REFERENCE_DATE - pd.Timedelta(days=180),
                "lastdate_end": REFERENCE_DATE - pd.Timedelta(days=180),
                "location_name": "Galerie X",
                "location_city": "Metz",
                "location_lat": None,
                "location_lon": None,
                "canonicalurl": "https://openagenda.com/culture/events/2",
                "originagenda_title": "Culture et sport en Grand Est",
                "status": _status(1, "Programmé"),
            },
            {  # excluded: too old (> 1 year)
                "uid": "3",
                "title_fr": "Trop ancien",
                "description_fr": "desc",
                "longdescription_fr": None,
                "keywords_fr": None,
                "conditions_fr": None,
                "accessibility_label_fr": None,
                "daterange_fr": "Il y a 2 ans",
                "firstdate_begin": REFERENCE_DATE - pd.Timedelta(days=800),
                "lastdate_end": REFERENCE_DATE - pd.Timedelta(days=800),
                "location_name": "Galerie Y",
                "location_city": "Metz",
                "location_lat": None,
                "location_lon": None,
                "canonicalurl": "https://openagenda.com/culture/events/3",
                "originagenda_title": "Centre Pompidou-Metz",
                "status": _status(1, "Programmé"),
            },
            {  # excluded: canceled
                "uid": "4",
                "title_fr": "Annulé",
                "description_fr": "desc",
                "longdescription_fr": None,
                "keywords_fr": None,
                "conditions_fr": None,
                "accessibility_label_fr": None,
                "daterange_fr": "Demain",
                "firstdate_begin": REFERENCE_DATE + pd.Timedelta(days=1),
                "lastdate_end": REFERENCE_DATE + pd.Timedelta(days=1),
                "location_name": "Salle Z",
                "location_city": "Metz",
                "location_lat": None,
                "location_lon": None,
                "canonicalurl": "https://openagenda.com/culture/events/4",
                "originagenda_title": "Centre Pompidou-Metz",
                "status": _status(6, "Annulé"),
            },
            {  # excluded: non-cultural source (France Travail)
                "uid": "5",
                "title_fr": "Session de recrutement LIDL",
                "description_fr": "desc",
                "longdescription_fr": None,
                "keywords_fr": ["Recrutement"],
                "conditions_fr": None,
                "accessibility_label_fr": None,
                "daterange_fr": "Lundi",
                "firstdate_begin": REFERENCE_DATE + pd.Timedelta(days=2),
                "lastdate_end": REFERENCE_DATE + pd.Timedelta(days=2),
                "location_name": "Agence Metz",
                "location_city": "Metz",
                "location_lat": None,
                "location_lon": None,
                "canonicalurl": "https://openagenda.com/francetravail/events/5",
                "originagenda_title": "Mes événements France Travail",
                "status": _status(1, "Programmé"),
            },
            {  # kept: far future, sold out
                "uid": "6",
                "title_fr": "Festival complet",
                "description_fr": "desc",
                "longdescription_fr": "<p>Festival</p>",
                "keywords_fr": ["festival"],
                "conditions_fr": "30€",
                "accessibility_label_fr": None,
                "daterange_fr": "Été prochain",
                "firstdate_begin": REFERENCE_DATE + pd.Timedelta(days=400),
                "lastdate_end": REFERENCE_DATE + pd.Timedelta(days=400),
                "location_name": "Parc",
                "location_city": "Metz",
                "location_lat": 49.1,
                "location_lon": 6.2,
                "canonicalurl": "https://openagenda.com/culture/events/6",
                "originagenda_title": "Centre Pompidou-Metz",
                "status": _status(5, "Complet"),
            },
        ]
    )


def test_filter_upcoming_events_keeps_only_events_not_yet_ended(sample_events):
    # Par défaut (HISTORY_DAYS = 0) : les événements déjà terminés (uid 2 et 3) sortent,
    # tous les événements à venir restent, même très lointains (uid 6, dans 400 jours).
    result = filter_upcoming_events(sample_events, REFERENCE_DATE)
    assert set(result["uid"]) == {"1", "4", "5", "6"}


def test_filter_upcoming_events_can_keep_a_history_window(sample_events):
    result = filter_upcoming_events(sample_events, REFERENCE_DATE, days=365)
    assert set(result["uid"]) == {"1", "2", "4", "5", "6"}


def test_filter_valid_status_excludes_canceled_but_keeps_full(sample_events):
    result = filter_valid_status(sample_events)
    assert "4" not in set(result["uid"])
    assert "6" in set(result["uid"])


def test_filter_cultural_events_excludes_known_non_cultural_sources(sample_events):
    result = filter_cultural_events(sample_events)
    assert "5" not in set(result["uid"])
    assert not set(result["originagenda_title"]) & EXCLUDED_ORIGINAGENDA_TITLES


def test_filter_cultural_events_excludes_accommodation_catalog_by_prefix(sample_events):
    catalog = sample_events.iloc[[0]].copy()
    catalog["uid"] = "catalogue-1"
    # Titre réel des données, avec l'apostrophe typographique (’) : c'est elle qui avait
    # fait échouer une première version du filtre, validée sur une apostrophe droite.
    catalog["originagenda_title"] = (
        "Catalogue départemental des structures d’accueil et d’hébergement - Vosges"
    )
    df = pd.concat([sample_events, catalog], ignore_index=True)

    result = filter_cultural_events(df)

    assert "catalogue-1" not in set(result["uid"])
    assert "1" in set(result["uid"])


def test_filter_upcoming_events_drops_out_of_range_dates_instead_of_crashing(sample_events):
    # Vu dans les données réelles : une offre d'emploi datée de l'an 2503, hors des bornes
    # de pandas (~2262) : le nettoyage ne doit pas planter, l'événement est simplement écarté.
    bad = sample_events.iloc[[0]].copy()
    bad["uid"] = "annee-2503"
    bad["lastdate_end"] = "2503-03-26T15:30:00+00:00"
    df = pd.concat([sample_events.astype({"lastdate_end": object}), bad], ignore_index=True)

    result = filter_upcoming_events(df, REFERENCE_DATE)

    assert "annee-2503" not in set(result["uid"])
    assert "1" in set(result["uid"])


def test_clean_html_strips_tags():
    assert clean_html("<p>Un concert <br>exceptionnel.</p>") == "Un concert exceptionnel."
    assert clean_html(None) == ""


def test_build_content_text_handles_nulls_and_includes_key_fields(sample_events):
    sparse_row = sample_events.iloc[1]
    content = build_content_text(sparse_row)
    assert "Exposition passée" in content
    assert "Metz" in content

    full_row = sample_events.iloc[5]
    assert "complet" in build_content_text(full_row).lower()


def test_select_output_columns_fills_missing_columns(sample_events):
    df = sample_events.copy()
    df["content"] = df.apply(build_content_text, axis=1)
    df["is_full"] = False

    result = select_output_columns(df)

    assert "location_lat" in result.columns
    assert len(result) == len(df)


def test_preprocess_end_to_end(tmp_path, sample_events):
    raw_path = tmp_path / "raw.parquet"
    output_path = tmp_path / "processed.parquet"
    sample_events.to_parquet(raw_path)

    result = preprocess(raw_path, output_path, reference_date=REFERENCE_DATE)

    assert output_path.exists()
    assert set(result["uid"]) == {"1", "6"}
    assert result["content"].str.strip().astype(bool).all()
    assert result["title_fr"].notna().all()
    assert not result["content"].str.contains("<").any()


def test_preprocess_decodes_html_entities_in_titles(tmp_path, sample_events):
    """Open Agenda livre parfois des titres avec des entités non décodées (« Jér&#244;me »)."""
    sample_events.loc[sample_events["uid"] == "1", "title_fr"] = "Rencontre avec Jér&#244;me Clément &amp; L&rsquo;équipe"
    raw_path = tmp_path / "raw.parquet"
    sample_events.to_parquet(raw_path)

    result = preprocess(raw_path, tmp_path / "processed.parquet", reference_date=REFERENCE_DATE)

    title = result.loc[result["uid"] == "1", "title_fr"].iloc[0]
    assert title == "Rencontre avec Jérôme Clément & L’équipe"
    content = result.loc[result["uid"] == "1", "content"].iloc[0]
    assert "&#244;" not in content and "&rsquo;" not in content
