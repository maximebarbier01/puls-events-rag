"""
Récupérer les événements bruts d'Open Agenda à venir pour la région cible
et les enregistrer dans le répertoire data/raw/.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.data.fetch import fetch_events, save_events
from app.data.preprocessing import HISTORY_DAYS

REGION = "Grand Est"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data/raw/evenements-publics-openagenda.parquet"


def main() -> None:
    min_last_date = (
        (datetime.now(timezone.utc) - timedelta(days=HISTORY_DAYS))
        .date()
        .isoformat()
    )
    print(
        f"Récupération des événements pour la région {REGION!r} (lastdate_end >= {min_last_date})..."
    )

    df = fetch_events(REGION, min_last_date=min_last_date)
    print(f"Récupération de {len(df)} lignes, enregistré sur {OUTPUT_PATH}")
    save_events(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
