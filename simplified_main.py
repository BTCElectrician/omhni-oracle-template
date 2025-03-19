"""
Simplified entry point for the Ohmni Oracle system.
This script uses the simplified PDF processor to process construction drawings.
"""
import os
import sys
import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import the simplified processor
from utils.simplified_pdf_processor import process_job_folder

# Load environment variables
load_dotenv()

# Set up logging
def setup_logging(output_folder: str) -> None:
    """Set up logging to both file and console."""
    os.makedirs(output_folder, exist_ok=True)
    
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"simplified_log_{timestamp}.txt")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    print(f"Logging to: {log_file}")

async def main_async() -> None:
    """Main async function for the application."""
    if len(sys.argv) < 2:
        print("Usage: python simplified_main.py <input_folder> [output_folder]")
        return
    
    job_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(job_folder, "simplified_output")
    
    if not os.path.exists(job_folder):
        print(f"Error: Input folder '{job_folder}' does not exist.")
        return
    
    # Set up logging
    setup_logging(output_folder)
    
    try:
        # Record start time
        start_time = time.time()
        
        logging.info(f"Starting simplified processing for {job_folder}")
        logging.info(f"Output will be saved to {output_folder}")
        
        # Create OpenAI client
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Process job folder
        results = await process_job_folder(job_folder, output_folder, client)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        logging.info(f"Processing complete")
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        logging.info(f"Processed {results['total_files']} files")
        logging.info(f"Successes: {results['successes']}, Failures: {results['failures']}")
        
    except Exception as e:
        logging.error(f"Unhandled exception in main process: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main_async()) 