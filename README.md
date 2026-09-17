# Puls-Events RAG

POC d'assistant intelligent capable de répondre à des questions sur des événements
culturels à venir, en s'appuyant sur un système RAG (Retrieval-Augmented Generation)
combinant recherche vectorielle (FAISS) et génération de réponse en langage naturel
(Mistral), orchestré avec LangChain. Les événements proviennent de l'API
[Open Agenda](https://data.opendatasoft.com/api/explore/v2.1/console).

Mission réalisée pour Puls-Events dans le cadre du parcours OpenClassrooms
"Concevez et déployez un système RAG".

## Table des matières

- [Objectifs](#objectifs)
- [Zone et période couvertes](#zone-et-période-couvertes)
- [Structure du projet](#structure-du-projet)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Reproduction depuis zéro](#reproduction-depuis-zéro)
- [Tests](#tests)

## Objectifs

- Récupérer et nettoyer les événements culturels via l'API Open Agenda.
- Vectoriser les descriptions d'événements et les indexer dans FAISS.
- Générer des réponses augmentées à une question utilisateur via LangChain + Mistral.
- Exposer le système via une API REST (`/ask`, `/rebuild`).
- Évaluer la qualité des réponses sur un jeu de questions/réponses annoté.

## Zone et période couvertes

- **Zone géographique** : département de la Moselle (Grand Est), via le filtre
  `location_department="Moselle"` de l'API Open Agenda.
- **Fenêtre temporelle** : 1 an d'historique + tous les événements à venir (pas de
  plafond dans le futur), soit `lastdate_end >= aujourd'hui - 365 jours`.
- **Filtre thématique** : les événements sont en plus filtrés pour exclure les sources
  non-culturelles présentes dans le jeu de données brut (ex : sessions de recrutement
  "France Travail", agriculture, cyclisme promotionnel...) — voir
  `EXCLUDED_ORIGINAGENDA_TITLES` dans [app/data/preprocessing.py](app/data/preprocessing.py).
- **Statut** : les événements annulés sont exclus ; les événements complets sont gardés
  mais signalés (`is_full`).
- Résultat : ~1439 événements culturels propres dans
  `data/processed/events_clean.parquet` (à partir de 2829 lignes brutes récupérées via
  `scripts/fetch_openagenda.py`).

## Structure du projet

```text
puls-events-rag/
├── app/                  # code applicatif (package Python)
│   ├── api/              # routes FastAPI (/ask, /rebuild)
│   ├── core/             # configuration, clients (Mistral, etc.)
│   ├── data/             # récupération + nettoyage Open Agenda
│   ├── vectorstore/      # construction / chargement de l'index FAISS
│   └── rag/              # chaîne LangChain (retrieval + génération)
├── scripts/              # scripts exécutables (ex: build_index.py, evaluate_rag.py)
├── tests/                # tests unitaires et fonctionnels
├── data/
│   ├── raw/               # données brutes Open Agenda (non versionné)
│   └── processed/         # données nettoyées prêtes à l'indexation (non versionné)
├── eval/                 # jeu de questions/réponses annoté
├── docs/                 # rapport technique, présentation
├── pyproject.toml / poetry.lock   # dépendances (source de vérité)
├── requirements.txt / requirements-dev.txt  # export pour reproduction sans Poetry
└── .env.example           # variables d'environnement attendues
```

## Prérequis

- Python 3.11
- [Poetry](https://python-poetry.org/docs/#installation) (recommandé) **ou** `pip` +
  `venv` avec les fichiers `requirements*.txt`
- Une clé API Mistral ([console.mistral.ai](https://console.mistral.ai/))

## Installation

### Option 1 — Poetry (recommandé)

```bash
poetry env use python3.11
poetry install --with dev,eval,notebook
```

### Option 2 — venv + pip

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

## Configuration

Copiez le fichier d'exemple et renseignez votre clé API :

```bash
cp .env.example .env
```

Ne versionnez jamais le fichier `.env` (déjà ignoré via `.gitignore`).

## Reproduction depuis zéro

Pour vérifier que l'environnement fonctionne sur une machine "propre" :

```bash
poetry env remove --all      # ou: rm -rf .venv
poetry install --with dev,eval,notebook
poetry run python -c "import faiss; from langchain_community.vectorstores import FAISS; from langchain_huggingface import HuggingFaceEmbeddings; from mistralai import Mistral; print('OK')"
```

## Tests

```bash
poetry run pytest
```

## Avancement

- [x] Étape 1 — Environnement de développement
- [x] Étape 2 — Pré-processing des données Open Agenda
- [ ] Étape 3 — Base vectorielle FAISS
- [ ] Étape 4 — Intégration LangChain / RAG
- [ ] Étape 5 — API REST
- [ ] Étape 6 — Conteneurisation et démo
