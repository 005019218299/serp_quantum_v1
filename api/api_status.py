from fastapi import APIRouter
from services.api_key_manager import api_key_manager

router = APIRouter()

@router.get("/api-keys/status")
async def get_api_keys_status():
    """Get status of all API keys"""
    return {
        "serpapi": api_key_manager.get_key_stats("serpapi"),
        "serpstack": api_key_manager.get_key_stats("serpstack"),
        "current_keys": {
            "serpapi": api_key_manager.get_active_key("serpapi"),
            "serpstack": api_key_manager.get_active_key("serpstack")
        }
    }