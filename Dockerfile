# Build multi-stage : le stage "builder" installe les dépendances via Poetry dans un
# .venv in-project, le stage "runtime" ne récupère que ce .venv + le code applicatif
# (app/). Les données (data/) et l'index FAISS (index/) ne sont jamais copiés dans
# l'image : ils sont montés en volumes au lancement (voir docker-compose.yml), déjà
# construits en local via scripts/00-fetch_openagenda.py -> 01-preprocess.py ->
# 02-build_index.py.

FROM python:3.11-slim AS builder
WORKDIR /app

RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.in-project true

# Couche dépendances seule d'abord, pour profiter du cache Docker tant que
# pyproject.toml/poetry.lock ne changent pas (un changement de code applicatif ne
# réinvalide pas cette couche).
COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-root --no-interaction

# pyproject.toml déclare readme = "README.md" : Poetry en a besoin pour installer
# le package racine (app/).
COPY README.md ./
COPY app ./app
RUN poetry install --only main --no-interaction


FROM python:3.11-slim AS runtime
WORKDIR /app

# faiss-cpu et torch (via sentence-transformers) chargent une bibliothèque native qui
# dépend de libgomp (runtime OpenMP), absente de l'image slim par défaut.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Utilisateur non-root pour l'exécution du conteneur (bonne pratique de sécurité).
# uid=1000 correspond à l'utilisateur hôte habituel sur Linux/WSL : les volumes
# data/ et index/ montés depuis l'hôte restent accessibles en écriture (nécessaire
# pour l'endpoint /rebuild).
RUN useradd --create-home --uid 1000 appuser

# Le cache HuggingFace (volume nommé hf_cache, voir docker-compose.yml) doit exister
# et appartenir à appuser AVANT le premier montage : Docker copie le contenu du
# répertoire (droits inclus) de l'image vers le volume nommé à sa création, sinon il
# resterait root:root et bloquerait le téléchargement du modèle en tant que non-root.
RUN mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser/.cache

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/app ./app
COPY README.md ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
