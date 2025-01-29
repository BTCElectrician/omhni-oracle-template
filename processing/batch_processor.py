import time
import asyncio
import logging

from processing.file_processor import process_pdf_async
from utils.constants import get_drawing_type

API_RATE_LIMIT = 60  # Adjust if needed
TIME_WINDOW = 60     # Time window to respect the rate limit

async def process_batch_async(batch, client, output_folder, templates_created):
    """
    Given a batch of PDF file paths, process each one asynchronously,
    respecting the API rate limit (API_RATE_LIMIT calls per TIME_WINDOW).
    """
    tasks = []
    start_time = time.time()

    for index, pdf_file in enumerate(batch):
        # Rate-limit control
        if index > 0 and index % API_RATE_LIMIT == 0:
            elapsed = time.time() - start_time
            if elapsed < TIME_WINDOW:
                await asyncio.sleep(TIME_WINDOW - elapsed)
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
