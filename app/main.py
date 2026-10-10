"""
CredVoice — Home Credit LAP Qualification Voice Agent API Gateway.

FastAPI entry point providing health check endpoints, API routing structure,
and runtime configuration details.
"""

import uuid

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from app.api.calls import router as calls_router
from app.api.tools import router as tools_router
from app.api.webhooks import router as webhooks_router
from app.config import settings
from app.services.knowledge import get_knowledge_service
from app.services.session_manager import get_session_manager

app = FastAPI(
    title=f"{settings.service_name} API",
    description="AI-powered Voice Agent backend for Home Credit Loan Against Property (LAP) qualification.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS based on settings
cors_origins = settings.cors_origins or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_and_correlation_middleware(request: Request, call_next):
    """
    Middleware for correlation ID tracking and payload body size limiting.
    """
    correlation_id = request.headers.get("x-correlation-id") or str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    # Enforce request body size limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > settings.max_request_body_bytes:
        return Response(
            content='{"detail": "Request payload exceeds maximum allowed size."}',
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            media_type="application/json",
        )

    response: Response = await call_next(request)
    response.headers["x-correlation-id"] = correlation_id
    return response


# Register API Routers
app.include_router(webhooks_router, prefix=f"{settings.api_prefix}/webhooks")
app.include_router(calls_router, prefix=f"{settings.api_prefix}/calls")
app.include_router(tools_router, prefix=f"{settings.api_prefix}/tools")


@app.get("/", tags=["General"])
async def root():
    """Root endpoint returning service identity and configuration summary."""
    return {
        "service": settings.service_name,
        "environment": settings.app_env,
        "version": "1.0.0",
        "description": "Home Credit LAP Qualification Voice Agent Gateway",
        "documentation": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for container probes and deployment verification."""
    return {
        "status": "healthy",
        "service": settings.service_name,
        "environment": settings.app_env,
        "voice_provider": settings.voice_provider,
        "llm_provider": settings.llm_provider
    }


@app.get("/readiness", tags=["Health"])
async def readiness_check():
    """Readiness probe checking knowledge base and session store health."""
    kb = get_knowledge_service()
    entries_count = len(kb.all_entries())
    if entries_count == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base failed to load FAQ entries.",
        )
    return {
        "status": "ready",
        "service": settings.service_name,
        "knowledge_entries": entries_count,
        "is_production": settings.is_production,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
