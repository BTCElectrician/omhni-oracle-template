import os
import sys
import asyncio
import logging
import time

from openai import AsyncOpenAI
from config.settings import OPENAI_API_KEY, get_all_settings
from utils.logging_utils import setup_logging
from processing.job_processor import process_job_site_async
from utils.performance_utils import get_tracker

async def main_async():
    """
    Main async function to handle processing with better error handling.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_folder> [output_folder]")
        return 1
    
    job_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(job_folder, "output")
    
    if not os.path.exists(job_folder):
        print(f"Error: Input folder '{job_folder}' does not exist.")
        return 1
    
    # 1) Set up logging
    setup_logging(output_folder)
    logging.info(f"Processing files from: {job_folder}")
    logging.info(f"Output will be saved to: {output_folder}")
    logging.info(f"Application settings: {get_all_settings()}")
    
    try:
        # 2) Create OpenAI Client (v1.66.3)
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # 3) Record start time
        start_time = time.time()
        
        # 4) Run asynchronous job processing
        await process_job_site_async(job_folder, output_folder, client)
        
        # 5) Calculate total processing time
        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        
        # 6) Generate performance report
        tracker = get_tracker()
        tracker.log_report()
        
        return 0
    except Exception as e:
        logging.error(f"Unhandled exception in main process: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 