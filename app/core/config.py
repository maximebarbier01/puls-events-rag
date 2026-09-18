"""
Configuration centralisée de l'application : chemins du projet et variables
d'environnement. On évite volontairement d'ajouter la dépendance pydantic-settings
pour un simple besoin de config sur un POC — os.environ + python-dotenv suffisent,
comme dans les scripts CLI (scripts/03-ask.py).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Racine du projet, calculée depuis ce fichier (app/core/config.py -> app/core -> app -> racine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Emplacements des données et de l'index, réutilisés par les scripts et l'API
RAW_DATA_PATH = PROJECT_ROOT / "data/raw/evenements-publics-openagenda.parquet"
INTERIM_DATA_PATH = PROJECT_ROOT / "data/interim/events_clean.parquet"
INDEX_PATH = PROJECT_ROOT / "index/events_faiss"


def load_env() -> None:
    """Charger les variables du fichier .env à la racine du projet.

    Appelé explicitement au démarrage de l'API (pas au moment de l'import de ce
    module), pour que le comportement reste prévisible et facile à tester.
    """
    load_dotenv(PROJECT_ROOT / ".env")


def get_rebuild_token() -> str | None:
    """Lire le jeton attendu pour protéger l'endpoint /rebuild.

    Retourne None si la variable n'est pas configurée : dans ce cas, l'endpoint doit
    refuser toute requête (fail-closed) plutôt que d'être ouvert par défaut.
    """
    return os.environ.get("REBUILD_TOKEN")
