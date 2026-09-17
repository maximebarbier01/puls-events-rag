"""Build data/processed/events_clean.parquet from the raw Open Agenda export."""
from pathlib import Path

from app.data.preprocessing import preprocess

RAW_PATH = Path("data/raw/evenements-publics-openagenda.parquet")
OUTPUT_PATH = Path("data/processed/events_clean.parquet")


def main() -> None:
    df = preprocess(RAW_PATH, OUTPUT_PATH)
    print(f"Wrote {len(df)} cleaned events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
