import glob
import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.api import (
    asset_liability,
    health,
    liquidation,
    metadata,
    price_shock,
    snapshot,
    ucache,
    risk_metrics,
    debug,
)
from backend.middleware.cache_middleware import CacheMiddleware
from backend.middleware.readiness import ReadinessMiddleware
from backend.state import BackendState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
state = BackendState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    url = os.getenv("RPC_URL")
    if not url:
        raise ValueError("RPC_URL environment variable is not set.")
    
    global state
    logger.info("Initializing backend state...")
    try:
        state.initialize(url)
        
        logger.info("Checking if cached vat exists")
        cached_vat_path = sorted(glob.glob("pickles/*"))
        if len(cached_vat_path) > 0:
            logger.info(f"Loading cached vat from {cached_vat_path[-1]}")
            await state.load_pickle_snapshot(cached_vat_path[-1])
        else:
            logger.info("No cached vat found, bootstrapping")
            await state.bootstrap()
            logger.info("Taking pickle snapshot")
            await state.take_pickle_snapshot()
        
        state.ready = True
        logger.info("Backend state initialization complete")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize backend state: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down...")
        state.ready = False
        if hasattr(state, 'dc'):
            await state.dc.unsubscribe()
        if hasattr(state, 'connection'):
            await state.connection.close()
        logger.info("App shutdown complete")


app = FastAPI(lifespan=lifespan)
app.add_middleware(ReadinessMiddleware, state=state)
app.add_middleware(CacheMiddleware, state=state, cache_dir="cache")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(metadata.router, prefix="/api/metadata", tags=["metadata"])
app.include_router(liquidation.router, prefix="/api/liquidation", tags=["liquidation"])
app.include_router(price_shock.router, prefix="/api/price-shock", tags=["price-shock"])
app.include_router(
    asset_liability.router, prefix="/api/asset-liability", tags=["asset-liability"]
)
app.include_router(snapshot.router, prefix="/api/snapshot", tags=["snapshot"])
app.include_router(ucache.router, prefix="/api/ucache", tags=["ucache"])
app.include_router(risk_metrics.router, prefix="/api/risk-metrics", tags=["risk-metrics"])
app.include_router(debug.router, prefix="/api/debug", tags=["debug"])


# NOTE: All other routes should be in /api/* within the /api folder. Routes outside of /api are not exposed in k8s
@app.get("/")
async def root():
    return {"message": "risk dashboard backend is online"}

@app.middleware("http")
async def add_backend_state(request: Request, call_next):
    """Middleware to attach backend state to request."""
    logger.debug("Adding backend state to request")
    try:
        if not hasattr(request.state, "backend_state"):
            request.state.backend_state = state
            logger.debug("Backend state attached to request")
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Error in backend state middleware: {e}")
        raise
