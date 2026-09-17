"""Build data/interim/events_clean.parquet from the raw Open Agenda export."""
from pathlib import Path

from app.data.preprocessing import preprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data/raw/evenements-publics-openagenda.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/interim/events_clean.parquet"


def main() -> None:
    df = preprocess(RAW_PATH, OUTPUT_PATH)
    print(f"Wrote {len(df)} cleaned events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
