import os
import logging
import asyncio
from tqdm.asyncio import tqdm

from utils.file_utils import traverse_job_folder
from processing.batch_processor import process_batch_async

async def process_job_site_async(job_folder, output_folder, client):
    """
    Orchestrates processing of a 'job site,' i.e., an entire folder of PDF files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = traverse_job_folder(job_folder)
    logging.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    
    if not pdf_files:
        logging.warning("No PDF files found. Please check the input folder.")
        return
    
    templates_created = {"floor_plan": False}
    batch_size = 10
    total_batches = (len(pdf_files) + batch_size - 1) // batch_size
    
    all_results = []
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1} of {total_batches}")
            
            batch_results = await process_batch_async(batch, client, output_folder, templates_created)
            all_results.extend(batch_results)
            
            successes = [r for r in batch_results if r['success']]
            failures = [r for r in batch_results if not r['success']]
            
            overall_pbar.update(len(batch))
            logging.info(f"Batch completed. Successes: {len(successes)}, Failures: {len(failures)}")
            
            for failure in failures:
                logging.error(f"Failed to process {failure['file']}: {failure['error']}")

    successes = [r for r in all_results if r['success']]
    failures = [r for r in all_results if not r['success']]
    
    logging.info(f"Processing complete. Total successes: {len(successes)}, Total failures: {len(failures)}")
    if failures:
        logging.warning("Failures:")
        for failure in failures:
            logging.warning(f"  {failure['file']}: {failure['error']}")
