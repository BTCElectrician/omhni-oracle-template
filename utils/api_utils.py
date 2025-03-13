"""
API utilities for making safe API calls.
"""
import asyncio
import logging
import random
from typing import Dict, Any

from services.ai_service import AiRateLimitError, AiConnectionError, AiResponseError

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


async def async_safe_api_call(client, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely call the OpenAI API with retries and backoff.
    This is a legacy function. Consider using the AI service instead.
    
    Args:
        client: OpenAI client
        *args: Positional arguments for the API call
        **kwargs: Keyword arguments for the API call
        
    Returns:
        API response
        
    Raises:
        Exception: If the API call fails after maximum retries
    """
    retries = 0
    delay = 1  # initial backoff

    while retries < MAX_RETRIES:
        try:
            return await client.chat.completions.create(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower():
                logging.warning(f"Rate limit hit, retrying in {delay} seconds...")
                retries += 1
                delay = min(delay * 2, 60)  # cap backoff at 60s
                await asyncio.sleep(delay + random.uniform(0, 1))  # add jitter
            else:
                logging.error(f"API call failed: {e}")
                await asyncio.sleep(RETRY_DELAY)
                retries += 1

    logging.error("Max retries reached for API call")
    raise Exception("Failed to make API call after maximum retries")
