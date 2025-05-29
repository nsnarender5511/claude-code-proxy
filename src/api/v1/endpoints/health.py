import logging
from fastapi import APIRouter
from src.core.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    logger.debug(
        f"Health check endpoint called. Target LLM Provider: {settings.TARGET_LLM_PROVIDER}"
    )
    return {"status": "ok", "target_llm_provider": settings.TARGET_LLM_PROVIDER}
