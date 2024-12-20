# Drift v2 Risk Dashboard


Quick Start:
1. Copy .env.example to .env and set RPC_URL
2. Create new venv `python -m venv .venv`
3. Activate venv `source .venv/bin/activate`
4. Install dependencies `pip install -r requirements.txt`
5. `export RPC_URL= <YOUR_RPC_URL>`
6. In one terminal, run the backend with `gunicorn backend.app:app -c gunicorn_config.py` (this might take a while to start up)
7. In another terminal, run the frontend with `streamlit run src/main.py`

Current Metrics:
1. Largest perp positions
2. Largest spot borrows
3. Account health distribution
4. Most levered perp positions > $1m notional
5. Most levered spot borrows > $750k notional
