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
    deposits,
    health,
    liquidation,
    metadata,
    pnl,
    positions,
    price_shock,
    snapshot,
    ucache,
    vaults,
    delist_recommender,
)
from backend.middleware.cache_middleware import CacheMiddleware
from backend.middleware.readiness import ReadinessMiddleware
from backend.state import BackendState
from backend.tasks.snapshot_watcher import SnapshotWatcher

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

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
app.include_router(backend_health.router, prefix="/api/backend-health", tags=["backend-health"])
app.include_router(metadata.router, prefix="/api/metadata", tags=["metadata"])
app.include_router(liquidation.router, prefix="/api/liquidation", tags=["liquidation"])
app.include_router(price_shock.router, prefix="/api/price-shock", tags=["price-shock"])
app.include_router(
    asset_liability.router, prefix="/api/asset-liability", tags=["asset-liability"]
)
app.include_router(snapshot.router, prefix="/api/snapshot", tags=["snapshot"])
app.include_router(ucache.router, prefix="/api/ucache", tags=["ucache"])
app.include_router(deposits.router, prefix="/api/deposits", tags=["deposits"])
app.include_router(pnl.router, prefix="/api/pnl", tags=["pnl"])
app.include_router(vaults.router, prefix="/api/vaults", tags=["vaults"])
app.include_router(positions.router, prefix="/api/positions", tags=["positions"])
app.include_router(delist_recommender.router, prefix="/api/delist-recommender", tags=["delist-recommender"])


# NOTE: All other routes should be in /api/* within the /api folder. Routes outside of /api are not exposed in k8s
@app.get("/")
async def root():
    return {"message": "risk dashboard backend is online"}
