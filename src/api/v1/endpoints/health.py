import logging
from fastapi import APIRouter
from src.core.config import settings # Updated import

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check():
    logger.info(f"Health check endpoint called. Target LLM Provider: {settings.TARGET_LLM_PROVIDER}")
    return {"status": "ok", "target_llm_provider": settings.TARGET_LLM_PROVIDER} 