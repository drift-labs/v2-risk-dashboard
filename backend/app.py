import glob
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

from backend.api import (
    asset_liability,
    deposits,
    health,
    liquidation,
    metadata,
    price_shock,
    snapshot,
    ucache,
)
from backend.middleware.cache_middleware import CacheMiddleware
from backend.middleware.readiness import ReadinessMiddleware
from backend.state import BackendState

load_dotenv()
state = BackendState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.getenv("RPC_URL")
    if not url:
        raise ValueError("RPC_URL environment variable is not set.")
    global state
    state.initialize(url)

    print("Checking if cached vat exists")
    cached_vat_path = sorted(glob.glob("pickles/*"))
    if len(cached_vat_path) > 0:
        print("Loading cached vat")
        await state.load_pickle_snapshot(cached_vat_path[-1])
    else:
        print("No cached vat found, bootstrapping")
        await state.bootstrap()
        await state.take_pickle_snapshot()
    state.ready = True
    import random
    import time

    time.sleep(random.randint(1, 10))
    print("Starting app")
    yield

    # Cleanup
    state.ready = False
    await state.dc.unsubscribe()
    await state.connection.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(ReadinessMiddleware, state=state)
app.add_middleware(CacheMiddleware, state=state, cache_dir="cache")

app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(metadata.router, prefix="/api/metadata", tags=["metadata"])
app.include_router(liquidation.router, prefix="/api/liquidation", tags=["liquidation"])
app.include_router(price_shock.router, prefix="/api/price-shock", tags=["price-shock"])
app.include_router(
    asset_liability.router, prefix="/api/asset-liability", tags=["asset-liability"]
)
app.include_router(snapshot.router, prefix="/api/snapshot", tags=["snapshot"])
app.include_router(ucache.router, prefix="/api/ucache", tags=["ucache"])
app.include_router(deposits.router, prefix="/api/deposits", tags=["deposits"])


# NOTE: All other routes should be in /api/* within the /api folder. Routes outside of /api are not exposed in k8s
@app.get("/")
async def root():
    return {"message": "risk dashboard backend is online"}
