"""Fetch raw Open Agenda events for the target department and save them to data/raw/."""
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.data.fetch import fetch_events, save_events
from app.data.preprocessing import RECENCY_WINDOW_DAYS

DEPARTMENT = "Moselle"
OUTPUT_PATH = Path("data/raw/evenements-publics-openagenda.parquet")


def main() -> None:
    min_last_date = (datetime.now(timezone.utc) - timedelta(days=RECENCY_WINDOW_DAYS)).date().isoformat()
    print(f"Fetching events for department={DEPARTMENT!r} (lastdate_end >= {min_last_date})...")

    df = fetch_events(department=DEPARTMENT, min_last_date=min_last_date)
    print(f"Fetched {len(df)} rows, saving to {OUTPUT_PATH}")
    save_events(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
