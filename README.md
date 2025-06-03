# Drift v2 Risk Dashboard

## Setup

First, make an env, activate it and pip install the dependencies:

```bash
pip install -r requirements.txt
```

You can run locally in multiple terminals like so:

```bash
# Start the backend
gunicorn backend.app:app -c gunicorn_config.py
```

In a different terminal, run the frontend with:

```bash
streamlit run src/main.py
```

Two endpoints, `asset liabilities` and `price shock` are CPU heavy and so are generated in a separate process
(instead of on request) and cached, the backend will serve these cached files.

In a different terminal, you can generate the cache files with:

```bash
./gen.sh
```

Instead of all the above you can also run it in docker with:

```bash
docker compose up --build
```

which will start a process to generate the cache files and then start the backend and frontend.

## Deployment

Pushing should automatically build the docker images and deploy to our k8s cluster.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/drift-labs/driftpy)