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
- [Utiliser l'API](#utiliser-lapi)
- [Évaluation (Ragas)](#évaluation-ragas)
- [Déploiement local avec Docker](#déploiement-local-avec-docker)

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
  `data/interim/events_clean.parquet` (à partir de 2829 lignes brutes récupérées via
  `scripts/00-fetch_openagenda.py`).

## Structure du projet

```text
puls-events-rag/
├── app/                  # code applicatif (package Python)
│   ├── api/              # routes FastAPI (/ask, /rebuild)
│   ├── core/             # configuration, clients (Mistral, etc.)
│   ├── data/             # récupération + nettoyage Open Agenda
│   ├── vectorstore/      # construction / chargement de l'index FAISS
│   └── rag/              # chaîne LangChain (retrieval + génération)
├── scripts/              # scripts exécutables numérotés (00-fetch, 01-preprocess, 02-build_index...)
├── tests/                # tests unitaires et fonctionnels
├── data/
│   ├── raw/               # données brutes Open Agenda (non versionné)
│   └── interim/           # données nettoyées prêtes à l'indexation (non versionné)
├── index/                 # index vectoriel FAISS régénérable (non versionné)
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

Aucun test n'appelle un vrai modèle d'embedding ni la vraie API Mistral (faux
objets déterministes partout : `tests/conftest.py`, `FakeListChatModel`) — la suite
est rapide, gratuite et reproductible en CI.

## Utiliser l'API

```bash
poetry run uvicorn app.main:app --reload
```

Documentation interactive (Swagger) : http://localhost:8000/docs

Au démarrage, l'API charge une seule fois le modèle d'embedding, l'index FAISS et le
client Mistral (voir le `lifespan` dans `app/main.py`) — ils ne sont jamais rechargés
à chaque requête.

**`POST /ask`** — poser une question :

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts à Metz ce week-end ?"}'
```

**`POST /rebuild`** — reconstruire l'index (relit `data/raw`, renettoie, réindexe).
Protégé par un jeton (`REBUILD_TOKEN` dans `.env`) :

```bash
curl -X POST http://localhost:8000/rebuild -H "X-Rebuild-Token: votre_jeton"
```

## Évaluation (Ragas)

Jeu de test annoté : [eval/qa_dataset.json](eval/qa_dataset.json) — 12 questions avec
réponses de référence, incluant des cas limites volontaires (questions hors périmètre
géographique/thématique) pour vérifier que le système refuse d'halluciner.

```bash
poetry run python scripts/04-evaluate_rag.py
```

Fait de vrais appels à l'API Mistral (génération + jugement des métriques) : coûte
quelques dizaines de centimes, prend une à deux minutes. Ce n'est pas un test pytest
(pas lancé à chaque `pytest`), volontairement séparé pour ne pas mélanger "tests
gratuits rapides" et "évaluation qui coûte et prend du temps".

4 métriques Ragas, jugées par notre propre modèle Mistral (pas OpenAI) via les
wrappers `LangchainLLMWrapper`/`LangchainEmbeddingsWrapper` :
- **Faithfulness** — la réponse est-elle fidèle au contexte récupéré (pas d'invention) ?
- **Answer relevancy** — la réponse correspond-elle bien à la question posée ?
- **Context precision** — les documents récupérés sont-ils pertinents ?
- **Context recall** — le contexte récupéré couvre-t-il la réponse de référence ?

À noter pour l'interprétation : les questions "pièges" (hors-sujet, hors périmètre)
font naturellement chuter `answer_relevancy` même quand le système répond
correctement en refusant — cette métrique compare la réponse à la question par
similarité sémantique, et une réponse de refus courte ne "ressemble" pas à la
question initiale. Un score bas sur ces questions précises n'indique donc pas un
problème de qualité.

Résultats détaillés sauvegardés dans `eval/results.csv` (non versionné,
régénérable). Le workflow
[.github/workflows/evaluate_rag.yml](.github/workflows/evaluate_rag.yml) relance
cette évaluation automatiquement à chaque push sur `main`, ou manuellement
(`workflow_dispatch`) — nécessite le secret de dépôt `MISTRAL_API_KEY`.

## Déploiement local avec Docker

Prérequis : avoir déjà construit les données et l'index **en local** au moins une
fois (l'image Docker ne contient ni les données ni l'index, montés en volumes au
lancement — voir `docker-compose.yml`) :

```bash
poetry run python scripts/00-fetch_openagenda.py
poetry run python scripts/01-preprocess.py
poetry run python scripts/02-build_index.py
```

Avoir aussi un `.env` complet (`MISTRAL_API_KEY`, `REBUILD_TOKEN`).

```bash
docker compose up --build
```

L'API est alors disponible exactement comme en local (mêmes endpoints, même
Swagger) :

```bash
curl http://localhost:8000/docs

curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels concerts à Metz ce week-end ?"}'

curl -X POST http://localhost:8000/rebuild -H "X-Rebuild-Token: votre_jeton"
```

Le conteneur tourne avec un utilisateur non-root, et le modèle d'embedding est mis
en cache dans un volume Docker nommé (`hf_cache`) — les redémarrages suivants
(`docker compose up`, sans `--build`) sont quasi instantanés. `data/` et `index/`
sont montés en volumes (pas copiés dans l'image) : un `/rebuild` déclenché depuis le
conteneur écrit directement sur ces dossiers côté hôte.

```bash
docker compose down
```

## Avancement

- [x] Étape 1 — Environnement de développement
- [x] Étape 2 — Pré-processing des données Open Agenda
- [x] Étape 3 — Base vectorielle FAISS
- [x] Étape 4 — Intégration LangChain / RAG
- [x] Étape 5 — API REST (FastAPI, `/ask`, `/rebuild` protégé, Swagger, `tests/api_test.py`)
- [x] Évaluation Ragas (jeu de test annoté + `scripts/04-evaluate_rag.py` + CI GitHub Actions)
- [x] Étape 6 — Conteneurisation Docker (build + run testés bout en bout : `/docs`, `/ask`, `/rebuild`) ; présentation PowerPoint à faire séparément
