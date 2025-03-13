"""
Batch processor module that handles batch processing of files.
"""
import time
import asyncio
import logging
from typing import List, Dict, Any

from processing.file_processor import process_pdf_async
from utils.constants import get_drawing_type

from config.settings import API_RATE_LIMIT, TIME_WINDOW


async def process_batch_async(
    batch: List[str],
    client,
    output_folder: str,
    templates_created: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """
    Given a batch of PDF file paths, process each one asynchronously,
    respecting the API rate limit (API_RATE_LIMIT calls per TIME_WINDOW).
    
    Args:
        batch: List of PDF file paths
        client: OpenAI client
        output_folder: Output folder for processed files
        templates_created: Dictionary tracking created templates
        
    Returns:
        List of processing results
    """
    tasks = []
    start_time = time.time()
    logger = logging.getLogger(__name__)

    for index, pdf_file in enumerate(batch):
        # Rate-limit control
        if index > 0 and index % API_RATE_LIMIT == 0:
            elapsed = time.time() - start_time
            if elapsed < TIME_WINDOW:
                wait_time = TIME_WINDOW - elapsed
                logger.info(f"Rate limiting: Waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            start_time = time.time()
        
        drawing_type = get_drawing_type(pdf_file)
        tasks.append(
            process_pdf_async(
                pdf_path=pdf_file,
                client=client,
                output_folder=output_folder,
                drawing_type=drawing_type,
                templates_created=templates_created
            )
        )
    
    return await asyncio.gather(*tasks)
