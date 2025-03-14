"""
Job processor module that orchestrates processing of a job site.
"""
import os
import logging
import asyncio
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from utils.file_utils import traverse_job_folder
from utils.constants import get_drawing_type
from processing.file_processor import process_pdf_async
from services.extraction_service import ExtractionResult
from config.settings import BATCH_SIZE


async def process_worker(
    queue: asyncio.Queue,
    client,
    output_folder: str,
    templates_created: Dict[str, bool],
    results: List[Dict[str, Any]]
) -> None:
    """
    Worker process that takes jobs from the queue and processes them.
    
    Args:
        queue: Queue of PDF files to process
        client: OpenAI client
        output_folder: Output folder for processed files
        templates_created: Dictionary tracking created templates
        results: List to collect processing results
    """
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            # Get a task from the queue, or break if queue is empty
            try:
                pdf_file, drawing_type = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check if queue is empty before breaking
                if queue.empty():
                    break
                continue
                
            try:
                # Process the PDF
                result = await process_pdf_async(
                    pdf_path=pdf_file,
                    client=client,
                    output_folder=output_folder,
                    drawing_type=drawing_type,
                    templates_created=templates_created
                )
                
                # Add result to results list
                results.append(result)
                
                # Log result
                if result['success']:
                    logger.info(f"Successfully processed {pdf_file}")
                else:
                    logger.error(f"Failed to process {pdf_file}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "file": pdf_file
                })
            finally:
                # Mark task as done
                queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")


async def monitor_progress(
    queue: asyncio.Queue,
    original_size: int,
    progress_bar: tqdm
) -> None:
    """
    Monitor progress of the queue and update the progress bar.
    
    Args:
        queue: Queue to monitor
        original_size: Original size of the queue
        progress_bar: Progress bar to update
    """
    last_size = queue.qsize()
    try:
        while True:
            await asyncio.sleep(0.5)  # Update twice per second
            current_size = queue.qsize()
            if current_size != last_size:
                # Update progress bar with completed items
                completed = original_size - current_size
                progress_bar.n = completed
                progress_bar.refresh()
                last_size = current_size
    except asyncio.CancelledError:
        # This is expected when the task is cancelled
        pass
    except Exception as e:
        logging.error(f"Monitor error: {str(e)}")


async def process_job_site_async(job_folder: str, output_folder: str, client) -> None:
    """
    Orchestrates processing of a 'job site,' i.e., an entire folder of PDF files.
    Uses a dynamic worker pool for optimal throughput.
    
    Args:
        job_folder: Input folder containing PDF files
        output_folder: Output folder for processed files
        client: OpenAI client
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = traverse_job_folder(job_folder)
    logger.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    
    if not pdf_files:
        logger.warning("No PDF files found. Please check the input folder.")
        return
    
    templates_created = {"floor_plan": False}
    
    # Create a queue of PDF files to process
    queue = asyncio.Queue()
    
    # Group files by drawing type for better organization
    files_by_type = {}
    for pdf_file in pdf_files:
        drawing_type = get_drawing_type(pdf_file)
        if drawing_type not in files_by_type:
            files_by_type[drawing_type] = []
        files_by_type[drawing_type].append(pdf_file)
    
    # Add files to queue
    for drawing_type, files in files_by_type.items():
        logger.info(f"Found {len(files)} {drawing_type} drawings")
        for pdf_file in files:
            await queue.put((pdf_file, drawing_type))
    
    # Determine optimal number of workers
    # Use at most BATCH_SIZE workers, but limit based on CPU cores and queue size
    max_workers = min(BATCH_SIZE, os.cpu_count() or 4, len(pdf_files))
    logger.info(f"Starting {max_workers} workers for {len(pdf_files)} files")
    
    # Shared list to collect results
    all_results = []
    
    # Create and start workers
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        # Track original queue size for progress
        original_queue_size = queue.qsize()
        
        # Create workers
        workers = []
        for _ in range(max_workers):
            worker = asyncio.create_task(
                process_worker(queue, client, output_folder, templates_created, all_results)
            )
            workers.append(worker)
        
        # Monitor progress while workers are running
        monitoring_task = asyncio.create_task(
            monitor_progress(queue, original_queue_size, overall_pbar)
        )
        
        # Wait for all tasks to be processed
        await queue.join()
        
        # Cancel worker tasks
        for worker in workers:
            worker.cancel()
        
        # Cancel the monitoring task
        monitoring_task.cancel()
        
        # Wait for all tasks to be cancelled
        await asyncio.gather(*workers, monitoring_task, return_exceptions=True)
        
        # Summarize results
        successes = [r for r in all_results if r.get('success', False)]
        failures = [r for r in all_results if not r.get('success', False)]
        
        logger.info(f"Processing complete. Total successes: {len(successes)}, Total failures: {len(failures)}")
        if failures:
            logger.warning("Failed files:")
            for failure in failures:
                logger.warning(f"  {failure['file']}: {failure.get('error', 'Unknown error')}")
