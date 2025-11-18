import glob
import logging
import os
import random
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from backend.api import (
    asset_liability,
    backend_health,
    deposits_api,
    health,
    high_leverage_api,
    liquidation_curves_api,
    market_recommender_api,
    metadata,
    open_interest_api,
    pnl_api,
    positions,
    price_shock,
    snapshot,
    ucache,
    user_retention_explorer_api,
    user_retention_summary_api,
    vaults_api,
    wallet_activity_api,
)
from backend.middleware.cache_middleware import CacheMiddleware
from backend.middleware.readiness import ReadinessMiddleware
from backend.state import BackendState
from backend.tasks.snapshot_watcher import SnapshotWatcher

load_dotenv()
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
logger = logging.getLogger(__name__)

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

state = BackendState()
snapshot_watcher = SnapshotWatcher(state, check_interval=60)  # Check every minute


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.getenv("RPC_URL")
    if not url:
        raise ValueError("RPC_URL environment variable is not set.")
    global state
    state.initialize(url)

    logger.info("Checking if cached vat exists")
    cached_vat_path = sorted(glob.glob("pickles/*"))
    if len(cached_vat_path) > 0:
        logger.info("Loading cached vat")
        await state.load_pickle_snapshot(cached_vat_path[-1])
    else:
        logger.info("No cached vat found, bootstrapping")
        await state.bootstrap()
        await state.take_pickle_snapshot()
    state.ready = True

    time.sleep(random.randint(1, 10))

    await snapshot_watcher.start()
    logger.info("Starting app")
    yield

    state.ready = False
    await snapshot_watcher.stop()
    await state.dc.unsubscribe()
    await state.connection.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(ReadinessMiddleware, state=state)
app.add_middleware(CacheMiddleware, state=state, cache_dir="cache")

app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(
    backend_health.router, prefix="/api/backend-health", tags=["backend-health"]
)
app.include_router(metadata.router, prefix="/api/metadata", tags=["metadata"])
app.include_router(
    liquidation_curves_api.router,
    prefix="/api/liquidation-curves",
    tags=["liquidation-curves"],
)
app.include_router(price_shock.router, prefix="/api/price-shock", tags=["price-shock"])
app.include_router(
    asset_liability.router, prefix="/api/asset-liability", tags=["asset-liability"]
)
app.include_router(snapshot.router, prefix="/api/snapshot", tags=["snapshot"])
app.include_router(ucache.router, prefix="/api/ucache", tags=["ucache"])
app.include_router(deposits_api.router, prefix="/api/deposits", tags=["deposits"])
app.include_router(pnl_api.router, prefix="/api/pnl", tags=["pnl"])
app.include_router(vaults_api.router, prefix="/api/vaults", tags=["vaults"])
app.include_router(positions.router, prefix="/api/positions", tags=["positions"])
app.include_router(
    market_recommender_api.router,
    prefix="/api/market-recommender",
    tags=["market-recommender"],
)
app.include_router(
    open_interest_api.router, prefix="/api/open-interest", tags=["open-interest"]
)
app.include_router(
    high_leverage_api.router, prefix="/api/high-leverage", tags=["high-leverage"]
)
app.include_router(
    user_retention_summary_api.router,
    prefix="/api/user-retention-summary",
    tags=["user-retention-summary"],
)
app.include_router(
    user_retention_explorer_api.router,
    prefix="/api/user-retention-explorer",
    tags=["user-retention-explorer"],
)
app.include_router(
    wallet_activity_api.router,
    prefix="/api/wallet-activity",
    tags=["wallet-activity"],
)


# NOTE: All other routes should be in /api/* within the /api folder. Routes outside of /api are not exposed in k8s
@app.get("/")
async def root():
    return {"message": "risk dashboard backend is online"}
