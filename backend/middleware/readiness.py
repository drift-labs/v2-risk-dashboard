from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from backend.state import BackendRequest, BackendState


class ReadinessMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, state: BackendState):
        super().__init__(app)
        self.state = state

    async def dispatch(self, request: BackendRequest, call_next):
        if not self.state.ready and not request.url.path.startswith("/api/backend-health"):
            raise HTTPException(status_code=503, detail="Service is not ready")

        request.state.backend_state = self.state
        response = await call_next(request)
        return response
