import logging
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from backend.state import BackendState

logger = logging.getLogger(__name__)

class ReadinessMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, state: BackendState):
        super().__init__(app)
        self.state = state
        logger.info("ReadinessMiddleware initialized")

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Check if the backend state is ready before processing requests.
        """
        logger.debug("Checking state readiness")
        
        # Health, debug, and VAT state endpoints should always be accessible
        if request.url.path in [
            "/api/health",
            "/api/risk-metrics/health",
            "/api/risk-metrics/debug",
            "/api/risk-metrics/vat-state"
        ]:
            return await call_next(request)
            
        # For all other endpoints, check state readiness
        if not self.state.is_ready:
            logger.warning("Backend state not ready")
            return JSONResponse(
                status_code=503,
                content={"result": "miss", "reason": "state_not_ready"}
            )
            
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Error in request processing: {e}")
            raise
