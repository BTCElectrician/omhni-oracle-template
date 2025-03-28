<file_map>
/Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template
├── config
│   └── settings.py
├── processing
│   ├── __init__.py
│   ├── file_processor.py
│   ├── job_processor.py
│   └── panel_schedule_processor.py
├── services
│   ├── __init__.py
│   ├── ai_service.py
│   ├── extraction_service.py
│   └── storage_service.py
├── templates
│   ├── prompts
│   │   ├── __init__.py
│   │   ├── architectural.py
│   │   ├── electrical.py
│   │   ├── general.py
│   │   ├── mechanical.py
│   │   └── plumbing.py
│   ├── __init__.py
│   ├── a_rooms_template.json
│   ├── base_templates.py
│   ├── e_rooms_template.json
│   ├── prompt_registry.py
│   ├── prompt_templates.py
│   ├── prompt_types.py
│   └── room_templates.py
├── utils
│   ├── __init__.py
│   ├── constants.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   └── performance_utils.py
├── .env.example
├── main.py
└── requirements.txt

</file_map>

<file_contents>
File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/main.py
```py
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
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/requirements.txt
```txt
# Async utilities
aiofiles>=23.2.1
asyncio>=3.4.3
aiohttp~=3.11
aiohappyeyeballs~=2.4.4
aiosignal~=1.3.2
anyio~=4.8.0
jiter~=0.8.2
sniffio~=1.3.1

# HTTP and API clients
httpx~=0.28.1
httpcore~=1.0.7
requests~=2.32.3
urllib3~=2.2.3
yarl~=1.17.0
frozenlist~=1.4.1
multidict~=6.1.0
h11~=0.14.0

# OpenAI
openai~=1.66.3  # For Responses API support
tiktoken~=0.9.0

# PDF processing
PyMuPDF~=1.24.11  # For enhanced text extraction formats and DataFrame support
pypdfium2~=4.30.0
Wand~=0.6.13
pandas>=2.0.0  # For PyMuPDF's Table.to_pandas() functionality

# Data validation and type handling
pydantic~=2.10.5  # Using v2 validation patterns (field_validator)
pydantic_core~=2.27.2
typing_extensions~=4.12.2
annotated-types~=0.7.0
attrs~=24.3.0

# Utilities
python-dotenv~=1.0.1
tqdm~=4.66.5
tenacity~=9.0.0  # For enhanced retry patterns
pillow~=10.4.0

# Security and crypto
certifi~=2024.12.14
cffi~=1.17.1
charset-normalizer~=3.4.1
cryptography~=44.0.0
pycparser~=2.22
distro~=1.9.0

# Azure (legacy, maintaining for backward compatibility)
azure-ai-documentintelligence==1.0.0
azure-core>=1.32.0

# Testing
pytest~=7.4.0
pytest-asyncio~=0.21.1  # For testing asynchronous functions
pytest-cov~=4.1.0  # For test coverage

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/config/settings.py
```py
"""
Application settings loaded from environment variables.
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Processing Configuration
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '60'))
TIME_WINDOW = int(os.getenv('TIME_WINDOW', '60'))

# Processing Mode Configuration
USE_SIMPLIFIED_PROCESSING = os.getenv('USE_SIMPLIFIED_PROCESSING', 'false').lower() == 'true'

# Model Selection Configuration - Define as a function to reload each time
def get_force_mini_model():
    """Always reload from env to get the latest value"""
    load_dotenv(override=True)
    return os.getenv('FORCE_MINI_MODEL', 'false').lower() == 'true'

# Standard definition for backward compatibility
FORCE_MINI_MODEL = get_force_mini_model()

# Template Configuration
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')

# Additional configuration settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

def get_all_settings() -> Dict[str, Any]:
    return {
        "OPENAI_API_KEY": "***REDACTED***" if OPENAI_API_KEY else None,
        "LOG_LEVEL": LOG_LEVEL,
        "BATCH_SIZE": BATCH_SIZE,
        "API_RATE_LIMIT": API_RATE_LIMIT,
        "TIME_WINDOW": TIME_WINDOW,
        "TEMPLATE_DIR": TEMPLATE_DIR,
        "DEBUG_MODE": DEBUG_MODE,
        "USE_SIMPLIFIED_PROCESSING": USE_SIMPLIFIED_PROCESSING,
        "FORCE_MINI_MODEL": get_force_mini_model()  # Always get latest value
    }

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/.env.example
```example
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_key_here

# Azure Document Intelligence Configuration
DOCUMENTINTELLIGENCE_ENDPOINT=<yourEndpoint>
DOCUMENTINTELLIGENCE_API_KEY=<yourKey>

# Optional Configuration
# LOG_LEVEL=INFO
# BATCH_SIZE=10
# API_RATE_LIMIT=60
# TIME_WINDOW=60 

# Model Selection (set to "true" to force mini model for all documents)
FORCE_MINI_MODEL=false 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/__init__.py
```py
# Processing package initialization

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/job_processor.py
```py
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
    results: List[Dict[str, Any]],
    worker_id: int,
    semaphore: asyncio.Semaphore
) -> None:
    """
    Enhanced worker process that takes jobs from the queue and processes them.
    Uses a semaphore to limit concurrent API calls.
    
    Args:
        queue: Queue of PDF files to process
        client: OpenAI client
        output_folder: Output folder for processed files
        templates_created: Dictionary tracking created templates
        results: List to collect processing results
        worker_id: Unique identifier for this worker
        semaphore: Semaphore to limit concurrent API calls
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            # Get a task from the queue, or break if queue is empty
            try:
                pdf_file, drawing_type = await asyncio.wait_for(queue.get(), timeout=1.0)
                logger.info(f"Worker {worker_id} processing {os.path.basename(pdf_file)}")
            except asyncio.TimeoutError:
                # Check if queue is empty before breaking
                if queue.empty():
                    logger.info(f"Worker {worker_id} finishing - queue empty")
                    break
                continue
                
            try:
                # Process the PDF with timeout protection and semaphore
                try:
                    async with semaphore:
                        logger.info(f"Worker {worker_id} acquired semaphore for {os.path.basename(pdf_file)}")
                        result = await asyncio.wait_for(
                            process_pdf_async(
                                pdf_path=pdf_file,
                                client=client,
                                output_folder=output_folder,
                                drawing_type=drawing_type,
                                templates_created=templates_created
                            ),
                            timeout=600  # 10-minute timeout per file
                        )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing {pdf_file} after 10 minutes")
                    result = {
                        "success": False,
                        "error": "Processing timed out after 10 minutes",
                        "file": pdf_file
                    }
                
                # Add result to results list
                results.append(result)
                
                # Log result
                if result['success']:
                    logger.info(f"Worker {worker_id} successfully processed {pdf_file}")
                else:
                    logger.error(f"Worker {worker_id} failed to process {pdf_file}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing {pdf_file}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "file": pdf_file
                })
            finally:
                # Mark task as done
                queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {str(e)}")
            # Continue to next item rather than breaking


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
    Uses prioritized queue processing with file size sorting and concurrency control.
    
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
    
    # Group and prioritize files by drawing type
    files_by_type = {}
    for pdf_file in pdf_files:
        drawing_type = get_drawing_type(pdf_file)
        if drawing_type not in files_by_type:
            files_by_type[drawing_type] = []
        files_by_type[drawing_type].append(pdf_file)
    
    # Define processing priority order
    priority_order = [
        'Architectural',  # Process architectural drawings first
        'Electrical',     # Then electrical
        'Mechanical',     # Then mechanical
        'Plumbing',       # Then plumbing
        'General'         # Other drawings last
    ]
    
    # Add files to queue in priority order, sorting by file size within each group
    for drawing_type in priority_order:
        if drawing_type in files_by_type:
            # Sort files by size (smallest first)
            files = sorted(files_by_type[drawing_type], key=lambda x: os.path.getsize(x))
            logger.info(f"Queueing {len(files)} {drawing_type} drawings (sorted by size)")
            for pdf_file in files:
                await queue.put((pdf_file, drawing_type))
    
    # Add any remaining file types not explicitly prioritized (also sorted by size)
    for drawing_type, files in files_by_type.items():
        if drawing_type not in priority_order:
            # Sort files by size (smallest first)
            files = sorted(files, key=lambda x: os.path.getsize(x))
            logger.info(f"Queueing {len(files)} {drawing_type} drawings (sorted by size)")
            for pdf_file in files:
                await queue.put((pdf_file, drawing_type))
    
    # Create a semaphore to limit concurrent API calls to 5
    semaphore = asyncio.Semaphore(5)
    logger.info("Using semaphore to limit concurrent API calls to 5")
    
    # Determine optimal number of workers
    max_workers = min(BATCH_SIZE, os.cpu_count() or 4, len(pdf_files))
    logger.info(f"Starting {max_workers} workers for {len(pdf_files)} files")
    
    # Shared list to collect results
    all_results = []
    
    # Create and start workers
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        # Track original queue size for progress
        original_queue_size = queue.qsize()
        
        # Create workers with IDs
        workers = []
        for i in range(max_workers):
            worker = asyncio.create_task(
                process_worker(queue, client, output_folder, templates_created, all_results, i+1, semaphore)
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

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/panel_schedule_processor.py
```py
import os
import json
import logging
from typing import Dict, Any, List, Optional
from tqdm.asyncio import tqdm

from services.extraction_service import PyMuPdfExtractor, ExtractionResult
from services.storage_service import FileSystemStorage
from services.ai_service import DrawingAiService, AiRequest, ModelType, optimize_model_parameters

# If you have a performance decorator, you can add it here if desired
# from utils.performance_utils import time_operation

def split_text_into_chunks(text: str, chunk_size: int = None) -> List[str]:
    """
    Returns the full text as a single chunk with no splitting.
    
    Args:
        text: The text to process
        chunk_size: Ignored parameter (kept for backwards compatibility)
        
    Returns:
        List with a single item containing the full text
    """
    return [text]

def normalize_panel_data_fields(panel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced normalization with expanded synonym mappings:
    - 'description', 'loadType', 'load type', 'item', 'equipment' => 'load_name'
    - 'ocp', 'amperage', 'breaker_size', 'amps', 'size' => 'trip'
    - 'circuit_no', 'circuit_number', 'ckt', 'circuit no', 'no' => 'circuit'
    - And other common electrical synonyms
    """
    circuits = panel_data.get("circuits", [])
    new_circuits = []

    for cdict in circuits:
        c = dict(cdict)
        # load_name synonyms
        if "description" in c and "load_name" not in c:
            c["load_name"] = c.pop("description")
        if "loadType" in c and "load_name" not in c:
            c["load_name"] = c.pop("loadType")
        if "load type" in c and "load_name" not in c:
            c["load_name"] = c.pop("load type")
        if "item" in c and "load_name" not in c:
            c["load_name"] = c.pop("item")
        if "equipment" in c and "load_name" not in c:
            c["load_name"] = c.pop("equipment")

        # trip synonyms
        if "ocp" in c and "trip" not in c:
            c["trip"] = c.pop("ocp")
        if "breaker_size" in c and "trip" not in c:
            c["trip"] = c.pop("breaker_size")
        if "amperage" in c and "trip" not in c:
            c["trip"] = c.pop("amperage")
        if "amp" in c and "trip" not in c:
            c["trip"] = c.pop("amp")
        if "amps" in c and "trip" not in c:
            c["trip"] = c.pop("amps")
        if "size" in c and "trip" not in c:
            c["trip"] = c.pop("size")

        # circuit synonyms
        if "circuit_no" in c and "circuit" not in c:
            c["circuit"] = c.pop("circuit_no")
        if "circuit_number" in c and "circuit" not in c:
            c["circuit"] = c.pop("circuit_number")
        if "ckt" in c and "circuit" not in c:
            c["circuit"] = c.pop("ckt")
        if "circuit no" in c and "circuit" not in c:
            c["circuit"] = c.pop("circuit no")
        if "no" in c and "circuit" not in c:
            c["circuit"] = c.pop("no")

        new_circuits.append(c)

    panel_data["circuits"] = new_circuits
    return panel_data

async def process_panel_schedule_content_async(
    extraction_result: ExtractionResult,
    client,
    file_name: str
) -> Optional[Dict[str, Any]]:
    """
    Process panel schedule content from an extraction result.
    
    Args:
        extraction_result: Result of PDF extraction containing text and tables
        client: OpenAI client
        file_name: Name of the file (for logging purposes)
        
    Returns:
        Processed panel data dictionary or None if processing failed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing panel schedule content for {file_name}")

    try:
        tables = extraction_result.tables
        raw_text = extraction_result.raw_text

        if tables:
            logger.info(f"Found {len(tables)} table(s). Using table-based parsing for {file_name}.")
            panels_data = await _parse_tables_without_chunking(tables, client, logger)
        else:
            logger.warning(f"No tables found in {file_name}—fallback to raw text approach.")
            panels_data = await _fallback_raw_text(raw_text, client, logger)

        if not panels_data:
            logger.warning(f"No panel data extracted from {file_name}.")
            return None

        # Return the first panel if there's only one, otherwise return the array
        if len(panels_data) == 1:
            return normalize_panel_data_fields(panels_data[0])
        else:
            # For multiple panels, create a parent object
            result = {
                "panels": [normalize_panel_data_fields(panel) for panel in panels_data]
            }
            return result

    except Exception as ex:
        logger.error(f"Unhandled error processing panel schedule content for {file_name}: {str(ex)}")
        return None

async def process_panel_schedule_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str
) -> Dict[str, Any]:
    """
    Specialized function for panel schedules:
    1) Extract with PyMuPDF
    2) If any tables found, parse them chunk-by-chunk
    3) If no tables, fallback to raw text chunking
    4) Merge partial results & synonyms
    5) Save final JSON to output_folder/drawing_type
    """
    logger = logging.getLogger(__name__)
    file_name = os.path.basename(pdf_path)

    extractor = PyMuPdfExtractor(logger=logger)
    storage = FileSystemStorage(logger=logger)

    with tqdm(total=100, desc=f"[PanelSchedules] {file_name}", leave=False) as pbar:
        try:
            extraction_result = await extractor.extract(pdf_path)
            pbar.update(10)
            if not extraction_result.success:
                err = f"Extraction failed: {extraction_result.error}"
                logger.error(err)
                return {"success": False, "error": err, "file": pdf_path}

            # Process the content
            panels_data = await process_panel_schedule_content_async(
                extraction_result=extraction_result,
                client=client,
                file_name=file_name
            )
            
            pbar.update(60)
            if not panels_data:
                logger.warning(f"No panel data extracted from {file_name}.")
                return {"success": True, "file": pdf_path, "panel_schedule": True, "data": []}

            # Now we place the final JSON in output_folder/drawing_type
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            base_name = os.path.splitext(file_name)[0]
            output_filename = f"{base_name}_panel_schedules.json"
            output_path = os.path.join(type_folder, output_filename)

            await storage.save_json(panels_data, output_path)
            pbar.update(30)

            logger.info(f"Saved panel schedules to {output_path}")
            pbar.update(10)

            return {
                "success": True,
                "file": output_path,
                "panel_schedule": True,
                "data": panels_data
            }

        except Exception as ex:
            pbar.update(100)
            logger.error(f"Unhandled error processing panel schedule {pdf_path}: {str(ex)}")
            return {"success": False, "error": str(ex), "file": pdf_path}

async def _parse_tables_without_chunking(tables: List[Dict[str, Any]], client, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Process all tables as a single unit - no chunking.
    Returns a list of panel objects.
    """
    from services.ai_service import AiRequest, DrawingAiService
    
    ai_service = DrawingAiService(client, drawing_instructions={}, logger=logger)
    all_panels = []

    # Combine all tables into a single content block
    combined_tables = "\n\n".join([tbl_info["content"] for tbl_info in tables if tbl_info["content"].strip()])
    
    if not combined_tables.strip():
        logger.debug("No valid table content found.")
        return []
    
    system_prompt = """
You are an advanced electrical-engineering assistant. I'm giving you tables from a panel schedule in Markdown form.
Analyze all tables as a complete set to identify:
1. Panel metadata (name, voltage, phases, location, etc.)
2. Complete circuit information
3. Any notes or additional specifications

Return valid JSON with:
{
  "panel_name": "...",
  "panel_metadata": { ... all available metadata ... },
  "circuits": [
    { "circuit": "...", "load_name": "...", "trip": "...", "poles": "...", ... }
  ]
}
CRITICAL: Ensure ALL property names (keys) in the JSON output are enclosed in double quotes.
Ensure ALL circuits are documented EXACTLY as shown. Missing or incomplete information can cause dangerous installation errors.
    """.strip()

    # Process the entire set of tables as a single unit
    user_text = f"FULL PANEL SCHEDULE TABLES:\n{combined_tables}"

    # Determine model type for table-based processing - Using GPT_4O_MINI as default
    selected_model_type = ModelType.GPT_4O_MINI
    logger.info(f"Using default model {selected_model_type.value} for panel schedule table processing.")

    request = AiRequest(
        content=user_text,
        model_type=selected_model_type,  # Use the determined model type
        temperature=0.05,  # Use lowest temperature for precision
        max_tokens=4000,
        system_message=system_prompt
    )

    response = await ai_service.process(request)
    if not response.success or not response.content:
        logger.warning(f"GPT parse error on combined tables: {response.error}")
        return []

    try:
        panel_json = json.loads(response.content)
        # Normalize synonyms
        panel_json = normalize_panel_data_fields(panel_json)
        # If we found circuits or a panel name, add it
        if panel_json.get("panel_name") or panel_json.get("circuits"):
            all_panels.append(panel_json)
        
        return all_panels
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error on combined tables: {str(e)}")
        logger.error(f"Raw problematic content: {response.content[:500]}...")
        return []

async def _fallback_raw_text(raw_text: str, client, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    If no tables found, process the entire raw_text as a single unit.
    Return a list of one or more panels if discovered.
    """
    from services.ai_service import AiRequest, DrawingAiService

    ai_service = DrawingAiService(client, drawing_instructions={}, logger=logger)

    fallback_prompt = """
You are an expert electrical engineer analyzing panel schedules. The content below represents a panel schedule 
with potentially unclear formatting. Your goal is to produce a precisely structured JSON representation.

Extract and structure:
1. Panel metadata (name, voltage, phases, location, etc.) in a 'panel_metadata' object
2. ALL circuit information in a 'circuits' array with EACH circuit having:
   - 'circuit': Circuit number EXACTLY as shown
   - 'trip': Breaker/trip size with units
   - 'poles': Number of poles (1, 2, or 3)
   - 'load_name': Complete description of connected load
   - Any additional circuit information present

Return a valid JSON object with:
{
  "panel_name": "Panel ID exactly as shown",
  "panel_metadata": { ... all available metadata ... },
  "circuits": [
    { "circuit": "1", "load_name": "Receptacles Room 101", "trip": "20A", ... },
    ...
  ]
}

CRITICAL: Ensure ALL property names (keys) in the JSON output are enclosed in double quotes.
CRITICAL: Missing circuits or incorrect information can lead to dangerous installation errors.
    """.strip()
    
    # Determine model type for fallback - Using GPT_4O_MINI as default
    # Alternatively, could call optimize_model_parameters here if needed,
    # but a default is simpler for the fallback path.
    selected_model_type = ModelType.GPT_4O_MINI
    logger.info(f"Using default model {selected_model_type.value} for panel schedule fallback.")
    
    request = AiRequest(
        content=raw_text,
        model_type=selected_model_type, # Use the determined model type
        temperature=0.05,  # Use lowest temperature for precision
        max_tokens=4000,
        system_message=fallback_prompt
    )
    
    response = await ai_service.process(request)
    if not response.success or not response.content:
        logger.warning(f"GPT parse error on raw text: {response.error}")
        return []
    
    try:
        panel_json = json.loads(response.content)
        # Normalize fields
        panel_json = normalize_panel_data_fields(panel_json)
        # If we found circuits or a panel name, add it
        all_panels = []
        if panel_json.get("panel_name") or panel_json.get("circuits"):
            all_panels.append(panel_json)
        
        return all_panels
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error on raw text: {str(e)}")
        logger.error(f"Raw problematic content: {response.content[:500]}...")
        return []

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/file_processor.py
```py
import os
import json
import logging
import asyncio
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from services.extraction_service import create_extractor, ExtractionResult
from services.ai_service import (
    process_drawing, 
    optimize_model_parameters, 
    DRAWING_INSTRUCTIONS, 
    detect_drawing_subtype
)
from services.storage_service import FileSystemStorage
from utils.performance_utils import time_operation
from utils.constants import get_drawing_type
from config.settings import USE_SIMPLIFIED_PROCESSING

# Load environment variables
load_dotenv()

def is_panel_schedule(file_name: str, raw_content: str) -> bool:
    """
    Determine if a PDF is likely an electrical panel schedule
    based solely on the file name (no numeric or content checks).
    
    Args:
        file_name (str): Name of the PDF file
        raw_content (str): (Unused) Extracted text content from the PDF
        
    Returns:
        bool: True if the file name contains certain panel-schedule keywords
    """
    panel_keywords = [
        "electrical panel schedule",
        "panel schedule",
        "panel schedules",
        "power schedule",
        "lighting schedule",
        # Hyphenated versions:
        "electrical-panel-schedule",
        "panel-schedule",
        "panel-schedules",
        "power-schedule",
        "lighting-schedule"
    ]
    file_name_lower = file_name.lower()
    return any(keyword in file_name_lower for keyword in panel_keywords)

@time_operation("total_processing")
async def process_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str,
    templates_created: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Process a single PDF asynchronously:
    1) Extract text and tables from PDF
    2) Detect drawing subtype 
    3) Process with appropriate AI approach based on subtype
    4) Save structured JSON output
    """
    file_name = os.path.basename(pdf_path)
    logger = logging.getLogger(__name__)

    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)
            extractor = create_extractor(drawing_type, logger)
            storage = FileSystemStorage(logger)

            extraction_result = await extractor.extract(pdf_path)
            if not extraction_result.success:
                pbar.update(100)
                logger.error(f"Extraction failed for {pdf_path}: {extraction_result.error}")
                return {"success": False, "error": extraction_result.error, "file": pdf_path}

            # Concatenate all extracted content without any truncation
            raw_content = extraction_result.raw_text
            for table in extraction_result.tables:
                raw_content += f"\nTABLE:\n{table['content']}\n"
                
            # Detect drawing subtype based on drawing type and filename
            subtype = detect_drawing_subtype(drawing_type, file_name)
            logger.info(f"Detected drawing subtype: {subtype} for file {file_name}")

            parsed_json = None
            structured_json = None

            # Process based on subtype
            if drawing_type == "Specifications" or "SPECIFICATION" in file_name.upper():
                # Specialized handling for specifications
                logger.info(f"Using optimized specification processing for {file_name}")
                
                # Get optimized parameters
                params = optimize_model_parameters("Specifications", raw_content, file_name)
                
                # Process with the specific system prompt for specifications
                system_message = f"""
                You are processing a SPECIFICATION document. Your only task is to extract the content into a structured JSON format.
                
                {DRAWING_INSTRUCTIONS.get('Specifications')}
                
                Return ONLY valid JSON. Do not include explanations, summaries, or any other text outside the JSON structure.
                """
                
                # Process all content at once - no chunking
                structured_json = await process_drawing(
                    raw_content=raw_content,
                    drawing_type="Specifications",
                    client=client,
                    file_name=file_name
                )
                
                try:
                    parsed_json = json.loads(structured_json)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error in specification {file_name}: {str(e)}")
                    return handle_json_error(structured_json, pdf_path, drawing_type, output_folder, file_name, storage, pbar)
                
                pbar.update(40)
                
            elif "PanelSchedule" in subtype or ("Electrical" in drawing_type and "panel" in file_name.lower()):
                # Import panel processor here to avoid circular imports
                from processing.panel_schedule_processor import process_panel_schedule_content_async
                
                logger.info(f"Processing panel schedule {file_name}")
                try:
                    parsed_json = await process_panel_schedule_content_async(
                        extraction_result=extraction_result,
                        client=client,
                        file_name=file_name
                    )
                    
                    if parsed_json is None:
                        logger.warning(f"Panel schedule processing returned no data for {file_name}")
                        return {"success": False, "error": "Panel schedule processing failed to extract data", "file": pdf_path}
                        
                    pbar.update(40)
                except Exception as e:
                    logger.error(f"Error in panel schedule processing for {file_name}: {str(e)}")
                    return {"success": False, "error": f"Panel schedule error: {str(e)}", "file": pdf_path}
                
            else:
                # Standard processing for other documents
                structured_json = await process_drawing(raw_content, drawing_type, client, file_name)
                pbar.update(40)
                
                try:
                    parsed_json = json.loads(structured_json)
                except json.JSONDecodeError as e:
                    return handle_json_error(structured_json, pdf_path, drawing_type, output_folder, file_name, storage, pbar)

            # Unified saving logic - executed for all document types
            if parsed_json:
                type_folder = os.path.join(output_folder, drawing_type)
                os.makedirs(type_folder, exist_ok=True)
                
                # Determine appropriate filename
                base_name = os.path.splitext(file_name)[0]
                suffix = "_panel_schedules.json" if "PanelSchedule" in subtype else "_structured.json"
                output_filename = f"{base_name}{suffix}"
                output_path = os.path.join(type_folder, output_filename)
                
                await storage.save_json(parsed_json, output_path)
                pbar.update(20)
                logger.info(f"Saved structured data to: {output_path}")

                # Process architectural drawings for room templates if applicable
                if drawing_type == 'Architectural' and 'rooms' in parsed_json:
                    from templates.room_templates import process_architectural_drawing
                    result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                    templates_created['floor_plan'] = True
                    logger.info(f"Created room templates: {result}")

                pbar.update(10)
                return {"success": True, "file": output_path}
            else:
                logger.error(f"No valid JSON produced for {file_name}")
                return {"success": False, "error": "No valid JSON produced", "file": pdf_path}
                
        except Exception as e:
            pbar.update(100)
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}


def handle_json_error(structured_json, pdf_path, drawing_type, output_folder, file_name, storage, pbar):
    """Helper function to handle JSON parse errors"""
    logger = logging.getLogger(__name__)
    pbar.update(100)
    logger.error(f"JSON parse error for {pdf_path}")
    
    if structured_json:
        logger.error(f"Raw API response: {structured_json[:500]}...")  # Log the first 500 chars
        type_folder = os.path.join(output_folder, drawing_type)
        os.makedirs(type_folder, exist_ok=True)
        raw_output_path = os.path.join(type_folder, f"{os.path.splitext(file_name)[0]}_raw_response.json")
        asyncio.create_task(storage.save_text(structured_json, raw_output_path))
    
    return {"success": False, "error": f"JSON parse failed", "file": pdf_path}
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/performance_utils.py
```py
"""
Performance tracking utilities.
"""
import time
import asyncio
import logging
import os
from typing import Dict, Any, List, Tuple, Callable, Optional, TypeVar, Coroutine
from functools import wraps

T = TypeVar('T')


class PerformanceTracker:
    """
    Tracks performance metrics for different operations.
    """
    def __init__(self):
        self.metrics = {
            "extraction": [],
            "ai_processing": [],
            "total_processing": []
        }
        self.logger = logging.getLogger(__name__)
        
    def add_metric(self, category: str, file_name: str, drawing_type: str, duration: float):
        """
        Add a performance metric.
        
        Args:
            category: Category of the operation (extraction, ai_processing, etc.)
            file_name: Name of the file being processed
            drawing_type: Type of drawing
            duration: Duration in seconds
        """
        if category not in self.metrics:
            self.metrics[category] = []
            
        self.metrics[category].append({
            "file_name": file_name,
            "drawing_type": drawing_type,
            "duration": duration
        })
        
    def get_average_duration(self, category: str, drawing_type: Optional[str] = None) -> float:
        """
        Get the average duration for a category.
        
        Args:
            category: Category of the operation
            drawing_type: Optional drawing type filter
            
        Returns:
            Average duration in seconds
        """
        if category not in self.metrics:
            return 0.0
            
        metrics = self.metrics[category]
        if drawing_type:
            metrics = [m for m in metrics if m["drawing_type"] == drawing_type]
            
        if not metrics:
            return 0.0
            
        return sum(m["duration"] for m in metrics) / len(metrics)
        
    def get_slowest_operations(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the slowest operations for a category.
        
        Args:
            category: Category of the operation
            limit: Maximum number of results
            
        Returns:
            List of slow operations
        """
        if category not in self.metrics:
            return []
            
        return sorted(
            self.metrics[category],
            key=lambda m: m["duration"],
            reverse=True
        )[:limit]
        
    def report(self):
        """
        Generate a report of performance metrics.
        
        Returns:
            Dictionary of performance reports
        """
        report = {}
        
        for category in self.metrics:
            if not self.metrics[category]:
                continue
                
            # Get overall average
            overall_avg = self.get_average_duration(category)
            
            # Get averages by drawing type
            drawing_types = set(m["drawing_type"] for m in self.metrics[category])
            type_averages = {
                dt: self.get_average_duration(category, dt)
                for dt in drawing_types
            }
            
            # Get slowest operations
            slowest = self.get_slowest_operations(category, 5)
            
            report[category] = {
                "overall_average": overall_avg,
                "by_drawing_type": type_averages,
                "slowest_operations": slowest,
                "total_operations": len(self.metrics[category])
            }
            
        return report
        
    def log_report(self):
        """
        Log the performance report.
        """
        report = self.report()
        
        self.logger.info("=== Performance Report ===")
        
        for category, data in report.items():
            self.logger.info(f"Category: {category}")
            self.logger.info(f"  Overall average: {data['overall_average']:.2f}s")
            self.logger.info(f"  Total operations: {data['total_operations']}")
            
            self.logger.info("  By drawing type:")
            for dt, avg in data['by_drawing_type'].items():
                self.logger.info(f"    {dt}: {avg:.2f}s")
                
            self.logger.info("  Slowest operations:")
            for op in data['slowest_operations']:
                self.logger.info(f"    {op['file_name']} ({op['drawing_type']}): {op['duration']:.2f}s")
                
        self.logger.info("==========================")


# Create a global instance
tracker = PerformanceTracker()


def time_operation(category: str):
    """
    Decorator to time an operation and add it to the tracker.
    
    Args:
        category: Category of the operation
    """
    def decorator(func):
        # Define common parameter names here so they're available to both wrappers
        file_params = ["pdf_path", "file_path", "path"]
        type_params = ["drawing_type", "type"]
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to determine file name and drawing type from args/kwargs
            file_name = "unknown"
            drawing_type = "unknown"
            
            # Check positional args - this is a heuristic and may need adjustment
            if len(args) > 0 and isinstance(args[0], str) and args[0].endswith(".pdf"):
                file_name = os.path.basename(args[0])
            
            if len(args) > 2 and isinstance(args[2], str):
                drawing_type = args[2]
                
            # Check keyword args
            for param in file_params:
                if param in kwargs and isinstance(kwargs[param], str):
                    file_name = os.path.basename(kwargs[param])
                    break
                    
            for param in type_params:
                if param in kwargs and isinstance(kwargs[param], str):
                    drawing_type = kwargs[param]
                    break
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                tracker.add_metric(category, file_name, drawing_type, duration)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic as async_wrapper for file_name and drawing_type
            file_name = "unknown"
            drawing_type = "unknown"
            
            # Check positional args
            if len(args) > 0 and isinstance(args[0], str) and args[0].endswith(".pdf"):
                file_name = os.path.basename(args[0])
            
            if len(args) > 2 and isinstance(args[2], str):
                drawing_type = args[2]
                
            # Check keyword args
            for param in file_params:
                if param in kwargs and isinstance(kwargs[param], str):
                    file_name = os.path.basename(kwargs[param])
                    break
                    
            for param in type_params:
                if param in kwargs and isinstance(kwargs[param], str):
                    drawing_type = kwargs[param]
                    break
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                tracker.add_metric(category, file_name, drawing_type, duration)
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Get the global tracker
def get_tracker() -> PerformanceTracker:
    """
    Get the global performance tracker.
    
    Returns:
        Global PerformanceTracker instance
    """
    return tracker 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/a_rooms_template.json
```json
{
    "room_id": "",
    "room_name": "",
    "walls": {
      "north": "",
      "south": "",
      "east": "",
      "west": ""
    },
    "ceiling_height": "",
    "dimensions": ""
}

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/logging_utils.py
```py
"""
Logging utilities with structured logging support.
"""
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredLogger:
    """
    Logger that produces structured log messages.
    """
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
        
    def add_context(self, **kwargs):
        """Add context to all log messages."""
        self.context.update(kwargs)
        
    def info(self, message: str, **kwargs):
        """Log an info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log a warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log an error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)
        
    def _log(self, level: int, message: str, **kwargs):
        """Internal method to log a message with context."""
        log_data = {**self.context, **kwargs, "message": message}
        self.logger.log(level, json.dumps(log_data))


def setup_logging(output_folder: str) -> None:
    """
    Configure and initialize logging for the application.
    Creates a 'logs' folder in the output directory.
    
    Args:
        output_folder: Folder to store log files
    """
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"process_log_{timestamp}.txt")
    
    # Configure basic logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler for visibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    print(f"Logging to: {log_file}")


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """
    Get a structured logger with the given name and context.
    
    Args:
        name: Logger name
        context: Optional context dictionary
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, context)

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/file_utils.py
```py
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def traverse_job_folder(job_folder: str) -> List[str]:
    """
    Traverse the job folder and collect all PDF files.
    """
    pdf_files = []
    try:
        for root, _, files in os.walk(job_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        logger.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    except Exception as e:
        logger.error(f"Error traversing job folder {job_folder}: {str(e)}")
    return pdf_files
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/e_rooms_template.json
```json
{
    "room_id": "",
    "room_name": "",
    "circuits": {
        "lighting": [],
        "power": []
    },
    "light_fixtures": {
        "fixture_ids": [],
        "fixture_count": {}
    },
    "outlets": {
        "regular_outlets": 0,
        "controlled_outlets": 0
    },
    "data": 0,
    "floor_boxes": 0,
    "mechanical_equipment": [],
    "switches": {
        "type": "",
        "model": "",
        "dimming": ""
    }
} 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/constants.py
```py
import os

DRAWING_TYPES = {
    'Architectural': ['A', 'AD'],
    'Electrical': ['E', 'ED'],
    'Mechanical': ['M', 'MD'],
    'Plumbing': ['P', 'PD'],
    'Site': ['S', 'SD'],
    'Civil': ['C', 'CD'],
    'Low Voltage': ['LV', 'LD'],
    'Fire Alarm': ['FA', 'FD'],
    'Kitchen': ['K', 'KD'],
    'Specifications': ['SPEC', 'SP']
}

def get_drawing_type(filename: str) -> str:
    """
    Detect the drawing type by examining the first 1-2 letters of the filename.
    """
    basename = os.path.basename(filename).upper()
    
    # Check by prefixes FIRST
    prefix = basename.split('.')[0][:2].upper()
    for dtype, prefixes in DRAWING_TYPES.items():
        if any(prefix.startswith(p.upper()) for p in prefixes):
            return dtype
            
    # Only use Specifications as a fallback if we couldn't determine by prefix
    if "SPEC" in basename or "SPECIFICATION" in basename:
        return "Specifications"
        
    return 'General'

def get_drawing_subtype(filename: str) -> str:
    """
    Detect the drawing subtype based on keywords in the filename.
    """
    filename_lower = filename.lower()
    if "panel schedule" in filename_lower or "electrical schedule" in filename_lower:
        return "electrical_panel_schedule"
    elif "mechanical schedule" in filename_lower:
        return "mechanical_schedule"
    elif "plumbing schedule" in filename_lower:
        return "plumbing_schedule"
    elif "wall types" in filename_lower or "partition types" in filename_lower:
        return "architectural_schedule"
    else:
        return "default"
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/__init__.py
```py
# Processing package initialization

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/ai_service.py
```py
# File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/ai_service.py
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, TypeVar, Generic, List
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.performance_utils import time_operation
from dotenv import load_dotenv
from templates.prompt_types import (
    DrawingCategory,
    ArchitecturalSubtype,
    ElectricalSubtype,
    MechanicalSubtype,
    PlumbingSubtype
)
from templates.prompt_templates import get_prompt_template

# Drawing type-specific instructions with main types and subtypes
# (DRAWING_INSTRUCTIONS dictionary remains the same - keeping it for brevity)
DRAWING_INSTRUCTIONS = {
    # ... (Keep the existing dictionary content) ...
    "Electrical": """
    You are an electrical drawing expert extracting structured information. Focus on:
    
    1. CRITICAL: Extract all metadata from the drawing's title block including:
       - drawing_number: The drawing number identifier 
       - title: The drawing title/description
       - revision: Revision number or letter
       - date: Drawing date
       - job_number: Project/job number
       - project_name: Full project name
    
    2. All panel schedules - capture complete information about:
       - Panel metadata (name, voltage, phases, rating, location)
       - All circuits with numbers, trip sizes, poles, load descriptions
       - Any panel notes or specifications
    
    3. All equipment schedules with:
       - Complete electrical characteristics (voltage, phase, current ratings)
       - Connection types and mounting specifications
       - Part numbers and manufacturers when available
    
    4. Installation details:
       - Circuit assignments and home run information
       - Mounting heights and special requirements
       - Keyed notes relevant to electrical items
       
    Structure all schedule information into consistent field names (e.g., use 'load_name' for descriptions, 
    'circuit' for circuit numbers, 'trip' for breaker sizes).
    
    IMPORTANT: Ensure ALL circuits, equipment items, and notes are captured in your output. Missing information 
    can cause installation errors.
    """,
    
    "Electrical_LIGHTING": """
    You are an expert in electrical lighting analyzing a lighting drawing or fixture schedule.
    
    CRITICAL: Extract all metadata from the drawing's title block, including:
    - drawing_number (e.g., "E1.00")
    - title (e.g., "LIGHTING - FLOOR LEVEL")
    - revision (e.g., "3")
    - date (e.g., "08/15/2024")
    - job_number (e.g., "30J7925")
    - project_name (e.g., "ELECTRIC SHUFFLE")
    
    Capture ALL lighting fixtures with these details:
    - type_mark: The fixture type identifier
    - count: Quantity of this fixture type
    - manufacturer: Fixture manufacturer name
    - product_number: Product/model number
    - description: Complete fixture description
    - finish: Material finish
    - lamp_type: Lamp specification with wattage and color temp
    - mounting: Mounting method
    - dimensions: Physical dimensions with units
    - location: Installation location
    - wattage: Power consumption
    - ballast_type: Driver/ballast type
    - dimmable: Whether fixture is dimmable
    - remarks: Any special notes
    - catalog_series: Full catalog reference
    
    Also document all lighting zones and controls:
    - zone: Zone identifier
    - area: Area served
    - circuit: Circuit number
    - fixture_type: Type of fixture
    - dimming_control: Control type
    - notes: Special conditions
    - quantities_or_linear_footage: Installation quantity
    
    Structure into a clear, consistent JSON format with metadata at the top level:
    {
      "ELECTRICAL": {
        "metadata": {
          "drawing_number": "E1.00",
          "title": "LIGHTING - FLOOR LEVEL",
          "revision": "3",
          "date": "08/15/2024", 
          "job_number": "30J7925",
          "project_name": "ELECTRIC SHUFFLE"
        },
        "LIGHTING_FIXTURE": [...],
        "LIGHTING_ZONE": [...]
      }
    }
    
    Lighting design coordination requires COMPLETE accuracy in fixture specifications.
    Missing or incorrect information can cause ordering errors and installation conflicts.
    """,
    
    "Mechanical": """
    Extract ALL mechanical information with a simplified, comprehensive structure.

1. Create a straightforward JSON structure with these main categories:
   - "equipment": Object containing arrays of ALL mechanical equipment grouped by type
   - "systems": Information about ductwork, piping, and distribution systems
   - "notes": ALL notes, specifications, and requirements
   - "remarks": ALL remarks and numbered references

2. For ANY type of equipment (air handlers, fans, VAVs, pumps, etc.):
   - Group by equipment type using descriptive keys (airHandlers, exhaustFans, chillers, etc.)
   - Include EVERY specification field with its EXACT value - never round or approximate
   - Use camelCase field names based on original headers
   - Always include identification (tag/ID), manufacturer, model, and capacity information
   - Capture ALL performance data (CFM, tonnage, BTU, static pressure, etc.)
   - Include ALL electrical characteristics (voltage, phase, FLA, MCA, etc.)

3. For ALL mechanical information:
   - Preserve EXACT values - never round or approximate
   - Include units of measurement
   - Keep the structure flat and simple
   - Don't skip ANY information shown on the drawing

Example simplified structure:
{
  "equipment": {
    "airHandlingUnits": [
      {
        "id": "AHU-1",
        "manufacturer": "Trane",
        "model": "M-Series",
        "cfm": "10,000",
        // ALL other fields exactly as shown
      }
    ],
    "exhaustFans": [
      // ALL fan data with EVERY field
    ]
  },
  "notes": [
    // ALL notes and specifications
  ],
  "remarks": [
    // ALL remarks and references
  ]
}

CRITICAL: Engineers need EVERY mechanical element and specification value EXACTLY as shown - complete accuracy is essential for proper system design, ordering, and installation.
    """,
    
    "Plumbing": """
    You are an expert AI assistant extracting detailed information from plumbing drawings, schedules, and notes. Your goal is to create a comprehensive and structured JSON output containing ALL relevant information presented.

    Analyze the provided text, which may include various schedules (fixtures, water heaters, pumps, valves, etc.), legends, and general notes. Structure your response into a single JSON object with the following top-level keys:

    1.  `metadata`: (Object) Capture any project identifiers, drawing numbers, titles, dates, or revisions found.
    2.  `fixture_schedule`: (Array of Objects) Extract details for EVERY item listed in the main plumbing fixture schedule(s). Include items like sinks (S1, S2, S3, HS, MS), drains (FD, FS, HD), cleanouts (WCO, FCO, CO), lavatories (SW-05), urinals (SW-03), water closets (SW-01), trap guards (TG), shock arrestors (SA), backflow preventers (DCBP), etc. For each item, include:
        - `fixture_id`: The exact mark or identifier (e.g., "S1", "SW-05", "WCO").
        - `description`: The full description provided.
        - `manufacturer`: Manufacturer name, if available.
        - `model`: Model number, if available.
        - `mounting`: Mounting details.
        - `connections`: (Object) Use the 'Connection Schedule' table to populate waste, vent, cold water (CW), and hot water (HW) sizes where applicable.
        - `notes`: Any specific notes related to this fixture.
    3.  `water_heater_schedule`: (Array of Objects) Extract details for EACH water heater (e.g., WH-1, WH-2). Include:
        - `mark`: The exact identifier (e.g., "WH-1").
        - `location`: Installation location.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `specifications`: (Object) Capture ALL technical specs like storage_gallons, operating_water_temp, tank_dimensions, recovery_rate, electric_power, kW_input, etc.
        - `mounting`: Mounting details (e.g., "Floor mounted").
        - `notes`: (Array of Strings) Capture ALL general notes associated specifically with the water heater schedule.
    4.  `pump_schedule`: (Array of Objects) Extract details for EACH pump (e.g., CP). Include:
        - `mark`: The exact identifier (e.g., "CP").
        - `location`: Installation location.
        - `serves`: What the pump serves.
        - `type`: Pump type (e.g., "IN-LINE").
        - `gpm`: Gallons Per Minute.
        - `tdh_ft`: Total Dynamic Head (in feet).
        - `hp`: Horsepower.
        - `rpm`: Max RPM.
        - `electrical`: Volts/Phase/Cycle.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `notes`: Any remarks or specific notes.
    5.  `mixing_valve_schedule`: (Array of Objects) Extract details for EACH thermostatic mixing valve (e.g., TM). Include:
        - `designation`: Identifier (e.g., "TM").
        - `location`: Service location.
        - `inlet_temp_F`: Hot water inlet temperature.
        - `outlet_temp_F`: Blended water temperature.
        - `pressure_drop_psi`: Pressure drop.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `notes`: Full description or notes.
    6.  `shock_absorber_schedule`: (Array of Objects) Extract details for EACH shock arrestor size listed (e.g., SA-A, SA-B,... SA-F, plus the general SA). Include:
        - `mark`: The exact identifier (e.g., "SA-A", "SA").
        - `fixture_units`: Applicable fixture units range.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `description`: Full description if provided separately.
    7.  `material_legend`: (Object) Capture the pipe material specifications (e.g., "SANITARY SEWER PIPING": "CAST IRON OR SCHEDULE 40 PVC").
    8.  `general_notes`: (Array of Strings) Extract ALL numbered or lettered general notes found in the text (like notes A-T).
    9.  `insulation_notes`: (Array of Strings) Extract ALL notes specifically related to plumbing insulation (like notes A-F).
    10. `symbols`: (Array of Objects, Optional) If needed, extract symbol descriptions.
    11. `abbreviations`: (Array of Objects, Optional) If needed, extract abbreviation definitions.

    CRITICAL:
    - Capture ALL items listed in EVERY schedule table or list. Do not omit any fixtures, equipment, or sizes.
    - Extract ALL general notes and insulation notes sections completely.
    - Preserve the exact details, model numbers, specifications, and text provided.
    - Ensure your entire response is a single, valid JSON object adhering to this structure. Missing information can lead to system failures or installation errors.
    """,
    
    "Architectural": """
    Extract and structure the following information with PRECISE detail:
    
    1. Room information:
       Create a comprehensive 'rooms' array with objects for EACH room, including:
       - 'number': Room number as string (EXACTLY as shown)
       - 'name': Complete room name
       - 'finish': All ceiling finishes
       - 'height': Ceiling height (with units)
       - 'electrical_info': Any electrical specifications
       - 'architectural_info': Additional architectural details
       - 'wall_types': Wall construction for each wall (north/south/east/west)
    
    2. Complete door and window schedules:
       - Door/window numbers, types, sizes, and materials
       - Hardware specifications and fire ratings
       - Frame types and special requirements
    
    3. Wall type details:
       - Create a 'wall_types' array with complete construction details
       - Include ALL layers, thicknesses, and special requirements
       - Document fire and sound ratings
    
    4. Architectural notes:
       - Capture ALL general notes and keyed notes
       - Include ALL finish schedule information
       
    CRITICAL: EVERY room must be captured. Missing rooms can cause major coordination issues.
    For rooms with minimal information, include what's available and note any missing details.
    """,
    
    "Specifications": """
Extract specification content using a clean, direct structure.

1. Create a straightforward 'specifications' array containing objects with:
   - 'section_title': EXACT section number and title (e.g., "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS")
   - 'content': COMPLETE text of the section with ALL parts and subsections
   
2. For the 'content' field:
   - Preserve the EXACT text - no summarizing or paraphrasing
   - Maintain ALL hierarchical structure (PART > SECTION > SUBSECTION)
   - Keep ALL numbering and lettering (1.1, A., etc.)
   - Include ALL paragraphs, tables, lists, and requirements

3. DO NOT add interpretations, summaries, or analysis
   - Your ONLY task is to preserve the original text in the correct sections
   - The structure should be simple and flat (just title + content for each section)
   - Handle each section as a complete unit

Example structure:
{
  "specifications": [
    {
      "section_title": "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS",
      "content": "PART 1 - GENERAL\\n\\n1.1 RELATED DOCUMENTS\\n\\nA. DRAWINGS AND GENERAL PROVISIONS...\\n\\n[COMPLETE TEXT HERE]"
    },
    {
      "section_title": "SECTION 16123 - BUILDING WIRE AND CABLE",
      "content": "PART 1 GENERAL\\n\\n1.01 SECTION INCLUDES\\n\\nA. WIRE AND CABLE...\\n\\n[COMPLETE TEXT HERE]"
    }
  ]
}

CRITICAL: Construction decisions rely on complete, unaltered specifications. Even minor omissions or changes can cause legal and safety issues.
    """,
    
    "General": """
    Extract ALL relevant content and organize into a comprehensive, structured JSON:
    
    1. Identify the document type and organize data accordingly:
       - For schedules: Create arrays of consistently structured objects
       - For specifications: Preserve the complete text with hierarchical structure
       - For drawings: Document all annotations, dimensions, and references
    
    2. Capture EVERY piece of information:
       - Include ALL notes, annotations, and references
       - Document ALL equipment, fixtures, and components
       - Preserve ALL technical specifications and requirements
    
    3. Maintain relationships between elements:
       - Link components to their locations (rooms, areas)
       - Connect items to their technical specifications
       - Reference related notes and details
    
    Structure everything into a clear, consistent JSON format that preserves ALL the original information.
    """,
    
    # Electrical subtypes
    "Electrical_PanelSchedule": """
    You are an expert electrical engineer analyzing panel schedules. Your goal is to produce a precisely structured JSON representation with COMPLETE information.
    
    Extract and structure the following information with PERFECT accuracy:
    
    1. Panel metadata (create a 'panel' object):
       - 'name': Panel name/ID (EXACTLY as written)
       - 'location': Physical location of panel
       - 'voltage': Full voltage specification (e.g., "120/208V Wye", "277/480V")
       - 'phases': Number of phases (1 or 3) and wires (e.g., "3 Phase 4 Wire")
       - 'amperage': Main amperage rating
       - 'main_breaker': Main breaker size if present
       - 'aic_rating': AIC/interrupting rating
       - 'feed': Source information (fed from)
       - Any additional metadata present (enclosure type, mounting, etc.)
       
    2. Circuit information (create a 'circuits' array with objects for EACH circuit):
       - 'circuit': Circuit number or range EXACTLY as shown (e.g., "1", "2-4-6", "3-5")
       - 'trip': Breaker/trip size with units (e.g., "20A", "70 A")
       - 'poles': Number of poles (1, 2, or 3)
       - 'load_name': Complete description of the connected load
       - 'equipment_ref': Reference to equipment ID if available
       - 'room_id': Connected room(s) if specified
       - Any additional circuit information present
       
    3. Additional information:
       - 'panel_totals': Connected load, demand factors, and calculated loads
       - 'notes': Any notes specific to the panel
       
    CRITICAL: EVERY circuit must be documented EXACTLY as shown on the schedule. Missing circuits, incorrect numbering, or incomplete information can cause dangerous installation errors.
    """,
    
    "Electrical_Lighting": """
    You are extracting complete lighting fixture and control information.
    
    1. Create a comprehensive 'lighting_fixtures' array with details for EACH fixture:
       - 'type_mark': Fixture type designation (EXACTLY as shown)
       - 'description': Complete fixture description
       - 'manufacturer': Manufacturer name
       - 'product_number': Model or catalog number
       - 'lamp_type': Complete lamp specification (e.g., "LED, 35W, 3500K")
       - 'mounting': Mounting type and height
       - 'voltage': Operating voltage
       - 'wattage': Power consumption
       - 'dimensions': Complete fixture dimensions
       - 'count': Quantity of fixtures when specified
       - 'location': Installation locations
       - 'dimmable': Dimming capability and type
       - 'remarks': Any special notes or requirements
       
    2. Document all lighting controls with specific details:
       - Switch types and functions
       - Sensors (occupancy, vacancy, daylight)
       - Dimming systems and protocols
       - Control zones and relationships
       
    3. Document circuit assignments:
       - Panel and circuit numbers
       - Connected areas and zones
       - Load calculations
       
    IMPORTANT: Capture EVERY fixture type and ALL specifications. Missing or incorrect information
    can lead to incompatible installations and lighting failure.
    """,
    
    "Electrical_Power": """
    Extract ALL power distribution and equipment connection information with complete detail:
    
    1. Document all outlets and receptacles in an organized array:
       - Type (standard, GFCI, special purpose, isolated ground)
       - Voltage and amperage ratings
       - Mounting height and orientation
       - Circuit assignment (panel and circuit number)
       - Room location and mounting surface
       - NEMA configuration
       
    2. Create a structured array of equipment connections:
       - Equipment type and designation
       - Power requirements (voltage, phase, amperage)
       - Connection method (hardwired, cord-and-plug)
       - Circuit assignment
       - Disconnecting means
       
    3. Detail specialized power systems:
       - UPS connections and specifications
       - Emergency or standby power
       - Isolated power systems
       - Specialty voltage requirements
       
    4. Document all keyed notes related to power:
       - Special installation requirements
       - Code compliance notes
       - Coordination requirements
       
    IMPORTANT: ALL power elements must be captured with their EXACT specifications.
    Electrical inspectors will verify these details during installation.
    """,
    
    "Electrical_FireAlarm": """
    Extract complete fire alarm system information with precise detail:
    
    1. Document ALL devices in a structured array:
       - Device type (smoke detector, heat detector, pull station, etc.)
       - Model number and manufacturer
       - Mounting height and location
       - Zone/circuit assignment
       - Addressable or conventional designation
       
    2. Identify all control equipment:
       - Fire alarm control panel specifications
       - Power supplies and battery calculations
       - Remote annunciators
       - Auxiliary control functions
       
    3. Capture ALL wiring specifications:
       - Circuit types (initiating, notification, signaling)
       - Wire types, sizes, and ratings
       - Survivability requirements
       
    4. Document interface requirements:
       - Sprinkler system monitoring
       - Elevator recall functions
       - HVAC shutdown
       - Door holder/closer release
       
    CRITICAL: Fire alarm systems are life-safety systems subject to strict code enforcement.
    ALL components and functions must be documented exactly as specified to ensure proper operation.
    """,
    
    "Electrical_Technology": """
    Extract ALL low voltage systems information with complete technical detail:
    
    1. Document data/telecom infrastructure in structured arrays:
       - Outlet types and locations
       - Cable specifications (category, shielding)
       - Mounting heights and orientations
       - Pathway types and sizes
       - Equipment rooms, racks, and cabinets
       
    2. Identify security systems with specific details:
       - Camera types, models, and coverage areas
       - Access control devices and door hardware
       - Intrusion detection sensors and zones
       - Control equipment and monitoring requirements
       
    3. Document audiovisual systems:
       - Display types, sizes, and mounting details
       - Audio equipment and speaker layout
       - Control systems and interfaces
       - Signal routing and processing
       
    4. Capture specialty systems:
       - Nurse call or emergency communication
       - Distributed antenna systems (DAS)
       - Paging and intercom
       - Radio and wireless systems
       
    IMPORTANT: Technology systems require precise documentation to ensure proper integration.
    ALL components, connections, and configurations must be captured as specified.
    """,
    
    # Architectural subtypes
    "Architectural_FloorPlan": """
    Extract COMPLETE floor plan information with precise room-by-room detail:
    
    1. Create a comprehensive 'rooms' array with objects for EACH room, capturing:
       - 'number': Room number EXACTLY as shown (including prefixes/suffixes)
       - 'name': Complete room name
       - 'dimensions': Length, width, and area when available
       - 'adjacent_rooms': List of connecting room numbers
       - 'wall_types': Wall construction for each room boundary (north/south/east/west)
       - 'door_numbers': Door numbers providing access to the room
       - 'window_numbers': Window numbers in the room
       
    2. Document circulation paths with specific details:
       - Corridor widths and clearances
       - Stair dimensions and configurations
       - Elevator locations and sizes
       - Exit paths and egress requirements
       
    3. Identify area designations and zoning:
       - Fire-rated separations and occupancy boundaries
       - Smoke compartments
       - Security zones
       - Department or functional areas
       
    CRITICAL: EVERY room must be documented with ALL available information.
    Missing rooms or incomplete details can cause serious coordination issues across all disciplines.
    When room information is unclear or incomplete, note this in the output.
    """,
    
    "Architectural_ReflectedCeiling": """
    Extract ALL ceiling information with complete room-by-room detail:
    
    1. Create a comprehensive 'rooms' array with ceiling-specific objects for EACH room:
       - 'number': Room number EXACTLY as shown
       - 'name': Complete room name
       - 'ceiling_type': Material and system (e.g., "2x2 ACT", "GWB")
       - 'ceiling_height': Height above finished floor (with units)
       - 'soffit_heights': Heights of any soffits or bulkheads
       - 'slope': Ceiling slope information if applicable
       - 'fixtures': Array of ceiling-mounted elements (lights, diffusers, sprinklers)
       
    2. Document ceiling transitions with specific details:
       - Height changes between areas
       - Bulkhead and soffit dimensions
       - Special ceiling features (clouds, islands)
       
    3. Identify ceiling-mounted elements:
       - Lighting fixtures (coordinated with electrical)
       - HVAC diffusers and registers
       - Sprinkler heads and fire alarm devices
       - Specialty items (projector mounts, speakers)
       
    IMPORTANT: Ceiling coordination is critical for clash detection.
    EVERY room must have complete ceiling information to prevent conflicts with mechanical, 
    electrical, and plumbing systems during installation.
    """,
    
    "Architectural_Partition": """
    Extract ALL wall and partition information with precise construction details:
    
    1. Create a comprehensive 'wall_types' array with objects for EACH type:
       - 'type': Wall type designation EXACTLY as shown
       - 'description': Complete description of the wall assembly
       - 'details': Object containing:
         - 'stud_type': Material and thickness (steel, wood)
         - 'stud_width': Dimension with units
         - 'stud_spacing': Spacing with units
         - 'layers': Complete description of all layers from exterior to interior
         - 'insulation': Type and R-value
         - 'total_thickness': Overall dimension with units
       - 'fire_rating': Fire resistance rating with duration
       - 'sound_rating': STC or other acoustic rating
       - 'height': Height designation ("to deck", "above ceiling")
       
    2. Document room-to-wall type relationships:
       - For each room, identify wall types used on each boundary (north/south/east/west)
       - Note any special conditions or variations
       
    3. Identify special wall conditions:
       - Seismic considerations
       - Expansion/control joints
       - Bracing requirements
       - Wall transitions
       
    CRITICAL: Wall type details impact all disciplines (architectural, structural, mechanical, electrical).
    EVERY wall type must be fully documented with ALL construction details to ensure proper installation.
    """,
    
    "Architectural_Details": """
    Extract ALL architectural details with complete construction information:
    
    1. Document each architectural detail:
       - Detail number and reference
       - Complete description of the assembly
       - Materials and dimensions
       - Connection methods and fastening
       - Finish requirements
       
    2. Capture specific assembly information:
       - Waterproofing and flashing details
       - Thermal and moisture protection
       - Acoustic treatments
       - Fire and smoke barriers
       
    3. Document all annotations and notes:
       - Construction requirements
       - Installation sequence
       - Quality standards
       - Reference standards
       
    IMPORTANT: Architectural details provide critical information for proper construction.
    ALL detail information must be captured exactly as specified to ensure code compliance
    and proper installation.
    """,
    
    "Architectural_Schedules": """
    Extract ALL architectural schedules with complete information for each element:
    
    1. Door schedules with comprehensive detail:
       - Door number EXACTLY as shown
       - Type, size (width, height, thickness)
       - Material and finish
       - Fire rating and label requirements
       - Frame type and material
       - Hardware sets and special requirements
       
    2. Window schedules with specific details:
       - Window number and type
       - Dimensions and configuration
       - Glazing type and performance ratings
       - Frame material and finish
       - Operating requirements
       
    3. Room finish schedules:
       - Room number and name
       - Floor, base, wall, and ceiling finishes
       - Special treatments or requirements
       - Finish transitions
       
    4. Accessory and equipment schedules:
       - Item designations and types
       - Mounting heights and locations
       - Material and finish specifications
       - Quantities and installation notes
       
    CRITICAL: Schedule information is used by multiple trades and disciplines.
    EVERY scheduled item must be completely documented with ALL specifications to ensure
    proper procurement and installation.
    """
}

def detect_drawing_subtype(drawing_type: str, file_name: str) -> str:
    """
    Detect more specific drawing subtype based on drawing type and filename.

    Args:
        drawing_type: Main drawing type (Electrical, Architectural, etc.)
        file_name: Name of the file being processed

    Returns:
        More specific subtype or the original drawing type if no subtype detected
    """
    if not file_name or not drawing_type:
        return drawing_type

    file_name_lower = file_name.lower()

    # Enhanced specification detection - check this first for efficiency
    if "specification" in drawing_type.lower() or any(term in file_name_lower for term in
                                                       ["spec", "specification", ".spec", "e0.01"]):
        return DrawingCategory.SPECIFICATIONS.value

    # Electrical subtypes
    if drawing_type == DrawingCategory.ELECTRICAL.value:
        # Panel schedules
        if any(term in file_name_lower for term in ["panel", "schedule", "panelboard", "circuit", "h1", "l1", "k1", "k1s", "21lp-1", "20h-1"]):
            return f"{drawing_type}_{ElectricalSubtype.PANEL_SCHEDULE.value}"
        # Lighting fixtures and controls
        elif any(term in file_name_lower for term in ["light", "lighting", "fixture", "lamp", "luminaire", "rcp", "ceiling"]):
            return f"{drawing_type}_{ElectricalSubtype.LIGHTING.value}"
        # Power distribution
        elif any(term in file_name_lower for term in ["power", "outlet", "receptacle", "equipment", "connect", "riser", "metering"]):
            return f"{drawing_type}_{ElectricalSubtype.POWER.value}"
        # Fire alarm systems
        elif any(term in file_name_lower for term in ["fire", "alarm", "fa", "detection", "smoke", "emergency", "evacuation"]):
            return f"{drawing_type}_{ElectricalSubtype.FIREALARM.value}"
        # Low voltage systems
        elif any(term in file_name_lower for term in ["tech", "data", "comm", "security", "av", "low voltage", "telecom", "network"]):
            return f"{drawing_type}_{ElectricalSubtype.TECHNOLOGY.value}"
        # Specifications (if not caught earlier)
        elif any(term in file_name_lower for term in ["spec", "specification", "requirement"]):
             return DrawingCategory.SPECIFICATIONS.value # Map directly to main spec type

    # Architectural subtypes
    elif drawing_type == DrawingCategory.ARCHITECTURAL.value:
        # Reflected ceiling plans
        if any(term in file_name_lower for term in ["rcp", "ceiling", "reflected"]):
            return f"{drawing_type}_{ArchitecturalSubtype.CEILING.value}"
        # Wall types and partitions
        elif any(term in file_name_lower for term in ["partition", "wall type", "wall-type", "wall", "room wall"]):
            return f"{drawing_type}_{ArchitecturalSubtype.WALL.value}"
        # Floor plans
        elif any(term in file_name_lower for term in ["floor", "plan", "layout", "room"]):
            return f"{drawing_type}_{ArchitecturalSubtype.ROOM.value}"
        # Door and window schedules
        elif any(term in file_name_lower for term in ["door", "window", "hardware", "schedule"]):
            return f"{drawing_type}_{ArchitecturalSubtype.DOOR.value}"
        # Architectural details
        elif any(term in file_name_lower for term in ["detail", "section", "elevation", "assembly"]):
            return f"{drawing_type}_{ArchitecturalSubtype.DETAIL.value}"

    # Mechanical subtypes
    elif drawing_type == DrawingCategory.MECHANICAL.value:
        # Equipment schedules
        if any(term in file_name_lower for term in ["equip", "unit", "ahu", "rtu", "vav", "schedule"]):
            return f"{drawing_type}_{MechanicalSubtype.EQUIPMENT.value}"
        # Ventilation systems
        elif any(term in file_name_lower for term in ["vent", "air", "supply", "return", "diffuser", "grille"]):
            return f"{drawing_type}_{MechanicalSubtype.VENTILATION.value}"
        # Piping systems
        elif any(term in file_name_lower for term in ["pipe", "chilled", "heating", "cooling", "refrigerant"]):
            return f"{drawing_type}_{MechanicalSubtype.PIPING.value}"

    # Plumbing subtypes
    elif drawing_type == DrawingCategory.PLUMBING.value:
        # Fixture schedules
        if any(term in file_name_lower for term in ["fixture", "sink", "toilet", "shower", "schedule"]):
            return f"{drawing_type}_{PlumbingSubtype.FIXTURE.value}"
        # Equipment
        elif any(term in file_name_lower for term in ["equip", "heater", "pump", "water", "schedule"]):
            return f"{drawing_type}_{PlumbingSubtype.EQUIPMENT.value}"
        # Piping systems
        elif any(term in file_name_lower for term in ["pipe", "riser", "water", "sanitary", "vent"]):
            return f"{drawing_type}_{PlumbingSubtype.PIPE.value}"

    # If no subtype detected, return the main type
    return drawing_type

class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

def optimize_model_parameters(
    drawing_type: str,
    raw_content: str,
    file_name: str
) -> Dict[str, Any]:
    """
    Determine optimal model parameters based on drawing type and content.

    Args:
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        raw_content: Raw content from the drawing
        file_name: Name of the file being processed

    Returns:
        Dictionary of optimized parameters
    """
    from dotenv import load_dotenv
    load_dotenv(override=True) # Reload to ensure we get the latest env values

    from config.settings import get_force_mini_model # Import the function instead

    content_length = len(raw_content)

    # Default parameters
    params = {
        "model_type": ModelType.GPT_4O_MINI,
        "temperature": 0.1, # Reduced default temperature for more consistent output
        "max_tokens": 16000, # Default max tokens for mini model
    }

    # Determine the appropriate model based on FORCE_MINI_MODEL flag and content length/type
    use_mini_model = get_force_mini_model()
    use_large_model = False

    # Conditions to potentially upgrade to the larger model (GPT-4O)
    if not use_mini_model:
        if content_length > 50000 or "specification" in drawing_type.lower():
            use_large_model = True
            logging.info(f"Content length ({content_length}) or type ({drawing_type}) suggests using GPT-4o for {file_name}")
        elif ("Electrical" in drawing_type and "PanelSchedule" in drawing_type and content_length > 15000):
             use_large_model = True
             logging.info(f"Complex panel schedule ({content_length}) suggests using GPT-4o for {file_name}")
        elif ("Architectural" in drawing_type and content_length > 20000):
             use_large_model = True
             logging.info(f"Large architectural drawing ({content_length}) suggests using GPT-4o for {file_name}")
        elif ("Mechanical" in drawing_type and content_length > 20000):
             use_large_model = True
             logging.info(f"Large mechanical drawing ({content_length}) suggests using GPT-4o for {file_name}")

    # Set the model type
    if use_large_model:
        params["model_type"] = ModelType.GPT_4O
        # Adjust max_tokens for the larger model
        estimated_input_tokens = min(128000, len(raw_content) // 4) # Cap at 128k tokens maximum
        # Reserve tokens for output, adjust based on model context window (approx 128k input, 4k output generally safe)
        params["max_tokens"] = max(4096, min(16000, 128000 - estimated_input_tokens - 1000)) # Ensure at least 4k, max 16k, within context
    elif use_mini_model:
         logging.info(f"Forcing gpt-4o-mini model for testing: {file_name}")
         # Keep default params["model_type"] = ModelType.GPT_4O_MINI
         # Keep default params["max_tokens"] = 16000

    # Adjust temperature based on drawing type
    if "Electrical" in drawing_type:
        # Panel schedules need lowest temperature for precision
        if "PanelSchedule" in drawing_type:
            params["temperature"] = 0.05
        # Other electrical drawings need precision too
        else:
            params["temperature"] = 0.1
    elif "Architectural" in drawing_type:
         # Requires precision but might need some inference for relationships
         params["temperature"] = 0.1
    elif "Mechanical" in drawing_type:
         # Slightly higher temperature might help with varied schedule formats
         params["temperature"] = 0.2 # Reduced from 0.3 for better consistency
    elif "Specification" in drawing_type:
        # Needs precision, very low temperature
        params["temperature"] = 0.05

    # Safety check: ensure max_tokens is within reasonable limits for the chosen model
    if params["model_type"] == ModelType.GPT_4O_MINI:
        params["max_tokens"] = min(params["max_tokens"], 16000) # Hard cap for mini
    elif params["model_type"] == ModelType.GPT_4O:
         # Let the calculated value stand, but ensure it doesn't exceed common practical limits like 16k unless necessary
         params["max_tokens"] = min(params["max_tokens"], 16000) # Re-cap at 16k for safety/cost unless specifically needed higher


    logging.info(f"Using model {params['model_type'].value} with temperature {params['temperature']} and max_tokens {params['max_tokens']} for {file_name}")

    return params


T = TypeVar('T')

class AiRequest:
    """
    Class to hold AI API request parameters.
    """
    def __init__(
        self,
        content: str,
        model_type: 'ModelType' = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
        system_message: str = ""
    ):
        """
        Initialize an AiRequest.

        Args:
            content: Content to send to the API
            model_type: Model type to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            system_message: System message to use
        """
        self.content = content
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message

class AiResponse(Generic[T]):
    """
    Generic class to hold AI API response data or errors.
    """
    def __init__(self, success: bool = True, content: str = "", parsed_content: Optional[T] = None, error: str = ""):
        """
        Initialize an AiResponse.

        Args:
            success: Whether the API call was successful
            content: Raw content from the API
            parsed_content: Optional parsed content (of generic type T)
            error: Error message if the call failed
        """
        self.success = success
        self.content = content
        self.parsed_content = parsed_content
        self.error = error

class DrawingAiService:
    """
    Specialized AI service for processing construction drawings.
    """
    def __init__(self, client: AsyncOpenAI, drawing_instructions: Dict[str, str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the DrawingAiService.

        Args:
            client: AsyncOpenAI client instance
            drawing_instructions: Optional dictionary of drawing type-specific instructions
            logger: Optional logger instance
        """
        self.client = client
        self.drawing_instructions = drawing_instructions or DRAWING_INSTRUCTIONS
        self.logger = logger or logging.getLogger(__name__)

    def _get_default_system_message(self, drawing_type: str) -> str:
        """
        Get the default system message for the given drawing type using the prompt templates.

        Args:
            drawing_type: Type of drawing (Architectural, Electrical, etc.) or subtype

        Returns:
            System message string
        """
        # Use the prompt template module to get the appropriate template
        return get_prompt_template(drawing_type)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process(self, request: AiRequest) -> AiResponse[Dict[str, Any]]:
        """
        Process an AI request. (Used by panel schedule processor primarily)

        Args:
            request: AiRequest object containing parameters

        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing content of length {len(request.content)} using model {request.model_type.value}")

            response = await self.client.chat.completions.create(
                model=request.model_type.value,
                messages=[
                    {"role": "system", "content": request.system_message},
                    {"role": "user", "content": request.content}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format={"type": "json_object"} # Ensure JSON response
            )

            content = response.choices[0].message.content

            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...") # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}", content=content) # Return raw content on error
        except Exception as e:
            self.logger.error(f"Error during AI processing: {str(e)}")
            return AiResponse(success=False, error=str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process_drawing_with_responses(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI,
        system_message: Optional[str] = None,
    ) -> AiResponse:
        """
        Process a drawing using the OpenAI API, returning an AiResponse.
        (Removed few-shot example logic)

        Args:
            raw_content: Complete raw content from the drawing - NO TRUNCATION
            drawing_type: Type of drawing
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            model_type: Model type to use
            system_message: Optional system message

        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing {drawing_type} drawing with {len(raw_content)} characters using model {model_type.value}")

            messages = [
                {"role": "system", "content": system_message or self._get_default_system_message(drawing_type)},
                {"role": "user", "content": raw_content}
            ]

            response = await self.client.chat.completions.create(
                model=model_type.value,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} # Ensure JSON response
            )

            content = response.choices[0].message.content

            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...") # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}", content=content) # Return raw content on error
        except Exception as e:
            self.logger.error(f"Error processing drawing: {str(e)}")
            return AiResponse(success=False, error=str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process_with_prompt(
        self,
        raw_content: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI,
        system_message: Optional[str] = None,
    ) -> str:
        """
        Process raw content using a specific prompt, ensuring full content is sent to the API.
        Returns the raw JSON string response.
        (Removed few-shot example logic)

        Args:
            raw_content: Raw content from the drawing
            temperature: Temperature parameter for the AI model
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            system_message: Optional custom system message to override default

        Returns:
            Processed content as a JSON string

        Raises:
            JSONDecodeError: If the response is not valid JSON
            ValueError: If the JSON structure is invalid or context length exceeded
            Exception: For other processing errors
        """
        default_system_message = """
        You are an AI assistant specialized in construction drawings. Extract all relevant information from the provided content and organize it into a structured JSON object with these sections:

        - "metadata": An object containing drawing metadata such as "drawing_number", "title", "date", and "revision".
        Include any available information; if a field is missing, omit it.

        - "schedules": An array of schedule objects. Each schedule should have a "type" (e.g., "electrical_panel",
        "mechanical") and a "data" array containing objects with standardized field names. For panel schedules,
        use consistent field names like "circuit" for circuit numbers, "trip" for breaker sizes,
        "load_name" for equipment descriptions, and "poles" for the number of poles.

        - "notes": An array of strings containing any notes or instructions found in the drawing.

        - "specifications": An array of objects, each with a "section_title" and "content" for specification sections.

        - "rooms": For architectural drawings, include an array of rooms with 'number', 'name', 'finish', 'height',
        'electrical_info', and 'architectural_info'.

        CRITICAL REQUIREMENTS:
        1. The JSON output MUST include ALL information from the drawing - nothing should be omitted
        2. Structure data consistently with descriptive field names
        3. Panel schedules MUST include EVERY circuit, with correct circuit numbers, trip sizes, and descriptions
        4. For architectural drawings, ALWAYS include a 'rooms' array with ALL rooms
        5. For specifications, preserve the COMPLETE text in the 'content' field
        6. Ensure the output is valid JSON with no syntax errors

        Construction decisions will be based on this data, so accuracy and completeness are essential.
        """

        # Use the provided system message or fall back to default
        final_system_message = system_message if system_message else default_system_message

        content_length = len(raw_content)
        self.logger.info(f"Processing content of length {content_length} with model {model_type.value}")

        # Context length warnings (remain relevant)
        if content_length > 250000 and model_type == ModelType.GPT_4O_MINI:
            self.logger.warning(f"Content length ({content_length} chars) is large for GPT-4o-mini context window. Consider upgrading model if issues occur.")
        if content_length > 500000 and model_type == ModelType.GPT_4O:
             self.logger.warning(f"Content length ({content_length} chars) is very large, approaching GPT-4o context window limits. Processing may be incomplete.")


        try:
            messages = [
                {"role": "system", "content": final_system_message},
                {"role": "user", "content": raw_content}
            ]

            # Calculate rough token estimate for logging
            estimated_tokens = content_length // 4
            self.logger.info(f"Estimated input tokens: ~{estimated_tokens}")

            try:
                response = await self.client.chat.completions.create(
                    model=model_type.value,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} # Ensure JSON response
                )
                content = response.choices[0].message.content

                # Process usage information if available
                if hasattr(response, 'usage') and response.usage:
                    self.logger.info(f"Token usage - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

                try:
                    # Validate JSON parsing immediately
                    parsed_content = json.loads(content)
                    if not self.validate_json(parsed_content):
                         self.logger.warning("JSON validation failed - missing required keys")
                         # Still return the content, as it might be usable even with missing keys

                    return content
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decoding error: {str(e)}")
                    self.logger.error(f"Raw content received: {content[:500]}...") # Log the first 500 chars for debugging
                    raise # Re-raise the JSONDecodeError

            except Exception as e:
                if "maximum context length" in str(e).lower() or "token limit" in str(e).lower():
                    self.logger.error(f"Context length exceeded: {str(e)}")
                    raise ValueError(f"Content too large for model context window: {str(e)}")
                else:
                    self.logger.error(f"API error: {str(e)}")
                    raise # Re-raise other API errors

        except Exception as e:
            self.logger.error(f"Error preparing or initiating AI processing: {str(e)}")
            raise # Re-raise any other unexpected errors

    def validate_json(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate the JSON structure.

        Args:
            json_data: Parsed JSON data

        Returns:
            True if the JSON has all required keys, False otherwise
        """
        # Basic validation - check for required top-level keys
        # Allow flexibility, just log warnings if keys are missing for now
        required_keys = ["metadata", "schedules", "notes"]
        missing_keys = [key for key in required_keys if key not in json_data]
        if missing_keys:
            self.logger.warning(f"JSON response missing expected keys: {', '.join(missing_keys)}")
            # Return True for now, but logging helps identify inconsistent outputs.
            # Could return False here if strict validation is needed later.

        # Specifications validation - check structure and convert if needed
        if "specifications" in json_data:
            specs = json_data["specifications"]
            if isinstance(specs, list) and specs:
                # Convert string arrays to object arrays if needed
                if isinstance(specs[0], str):
                    self.logger.warning("Converting specifications from string array to object array")
                    json_data["specifications"] = [{"section_title": spec, "content": ""} for spec in specs]
        # No else needed - if 'specifications' isn't there or is empty, that's okay

        # For architectural drawings, log if rooms array is missing
        if isinstance(json_data.get("metadata"), dict) and \
           "architectural" in json_data["metadata"].get("drawing_type", "").lower() and \
           "rooms" not in json_data:
            self.logger.warning("Architectural drawing missing 'rooms' array in JSON response")
            # Again, return True for now, but log the issue.

        return True # Return True unless strict validation requires False on warnings

# --- REMOVED get_example_output function ---

@time_operation("ai_processing")
async def process_drawing(raw_content: str, drawing_type: str, client, file_name: str = "") -> str:
    """
    Use GPT to parse PDF text and table data into structured JSON based on the drawing type.
    (Removed few-shot example logic)

    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        client: OpenAI client
        file_name: Optional name of the file being processed

    Returns:
        Structured JSON as a string

    Raises:
        ValueError: If the content is empty or too large for processing
        JSONDecodeError: If the response is not valid JSON
        Exception: For other processing errors
    """
    if not raw_content:
        logging.warning(f"Empty content received for {file_name}. Cannot process.")
        raise ValueError("Cannot process empty content")

    # Log details about processing task
    content_length = len(raw_content)
    drawing_type = drawing_type or "Unknown"
    file_name = file_name or "Unknown"

    logging.info(f"Starting drawing processing: Type={drawing_type}, File={file_name}, Content length={content_length}")

    try:
        # Detect more specific drawing subtype
        subtype = detect_drawing_subtype(drawing_type, file_name)
        logging.info(f"Detected drawing subtype: {subtype}")

        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)

        # Get optimized parameters for this drawing
        params = optimize_model_parameters(subtype, raw_content, file_name)

        # --- REMOVED call to get_example_output ---

        # Check if this is a specification document
        is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"

        # Determine the appropriate system message based on detected subtype/type
        if is_specification:
            # Simplified system message retrieval for specifications
            system_message = ai_service._get_default_system_message(DrawingCategory.SPECIFICATIONS.value)
            logging.info("Using specification-specific system prompt.")
        else:
            # Get the appropriate system message based on detected subtype
            system_message = ai_service._get_default_system_message(subtype)
            logging.info(f"Using system prompt for subtype: {subtype}")


        # Process the drawing using the simplified prompt method
        try:
            response_str = await ai_service.process_with_prompt(
                raw_content=raw_content,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                model_type=params["model_type"],
                system_message=system_message,
                # --- REMOVED example_output parameter ---
            )

            # Basic validation check after receiving the string
            try:
                json.loads(response_str) # Try parsing to ensure it's valid JSON
                logging.info(f"Successfully processed {subtype} drawing ({len(response_str)} chars output)")
                return response_str
            except json.JSONDecodeError as json_err:
                 logging.error(f"Invalid JSON response from AI service for {file_name}: {json_err}")
                 logging.error(f"Raw response snippet: {response_str[:500]}...")
                 raise # Re-raise the JSON error

        except ValueError as val_err: # Catch context length errors from process_with_prompt
            logging.error(f"Value error processing {file_name}: {val_err}")
            raise # Re-raise the value error
        except Exception as proc_err: # Catch other API/processing errors
            logging.error(f"Error during AI processing call for {file_name}: {proc_err}")
            raise # Re-raise other errors


    except Exception as e:
        logging.error(f"Unexpected error setting up processing for {drawing_type} drawing '{file_name}': {str(e)}")
        raise # Re-raise any setup errors

# --- REMOVED process_drawing_with_examples function ---
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/__init__.py
```py
# Services package initialization 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/extraction_service.py
```py
"""
Extraction service interface and implementations for PDF content extraction.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
import os

import pymupdf as fitz

from utils.performance_utils import time_operation


class ExtractionResult:
    """
    Domain model representing the result of a PDF extraction operation.
    """
    def __init__(
        self,
        raw_text: str,
        tables: List[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.raw_text = raw_text
        self.tables = tables
        self.success = success
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "raw_text": self.raw_text,
            "tables": self.tables,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create an ExtractionResult from a dictionary."""
        return cls(
            raw_text=data.get("raw_text", ""),
            tables=data.get("tables", []),
            success=data.get("success", False),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


class PdfExtractor(ABC):
    """
    Abstract base class defining the interface for PDF extraction services.
    """
    @abstractmethod
    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractionResult containing the extracted content
        """
        pass


class PyMuPdfExtractor(PdfExtractor):
    """
    PDF content extractor implementation using PyMuPDF.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    @time_operation("extraction")
    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text and tables from a PDF file using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractionResult containing the extracted content
        """
        try:
            self.logger.info(f"Starting extraction for {file_path}")
            
            # Use run_in_executor to move CPU-bound work off the main thread
            loop = asyncio.get_event_loop()
            raw_text, tables, metadata = await loop.run_in_executor(
                None, self._extract_content, file_path
            )
            
            # Check if we got any content
            if not raw_text and not tables:
                self.logger.warning(f"No content extracted from {file_path}")
                return ExtractionResult(
                    raw_text="No content could be extracted from this PDF.",
                    tables=[],
                    success=True,  # Still mark as success to continue processing
                    metadata=metadata
                )
            
            self.logger.info(f"Successfully extracted content from {file_path}")
            return ExtractionResult(
                raw_text=raw_text,
                tables=tables,
                success=True,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {str(e)}")
            return ExtractionResult(
                raw_text="",
                tables=[],
                success=False,
                error=str(e)
            )

    def _extract_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Internal method to extract content from a PDF file.
        This method runs in a separate thread.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (raw_text, tables, metadata)
        """
        # Use context manager to ensure document is properly closed
        with fitz.open(file_path) as doc:
            # Extract metadata first
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": len(doc)
            }
            
            # Initialize containers for text and tables
            raw_text = ""
            tables = []
            
            # Process each page individually to avoid reference issues
            for i, page in enumerate(doc):
                # Add page header
                page_text = f"PAGE {i+1}:\n"
                
                # Try block-based extraction first
                try:
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if block[6] == 0:  # Text block (type 0)
                            page_text += block[4] + "\n"
                except Exception as e:
                    self.logger.warning(f"Block extraction error on page {i+1}: {str(e)}")
                    # Fall back to regular text extraction
                    try:
                        page_text += page.get_text() + "\n\n"
                    except Exception as e2:
                        self.logger.warning(f"Error extracting text from page {i+1}: {str(e2)}")
                        page_text += "[Error extracting text from this page]\n\n"
                
                # Add to overall text
                raw_text += page_text
                
                # Extract tables safely
                try:
                    # Find tables on the page
                    table_finder = page.find_tables()
                    if table_finder and hasattr(table_finder, "tables"):
                        for j, table in enumerate(table_finder.tables):
                            try:
                                table_markdown = table.to_markdown()
                                tables.append({
                                    "page": i+1,
                                    "table_index": j,
                                    "content": table_markdown
                                })
                            except Exception as e:
                                self.logger.warning(f"Error converting table {j} on page {i+1}: {str(e)}")
                                tables.append({
                                    "page": i+1,
                                    "table_index": j,
                                    "content": f"[Error extracting table {j} from page {i+1}]"
                                })
                except Exception as e:
                    self.logger.warning(f"Error finding tables on page {i+1}: {str(e)}")
            
            return raw_text, tables, metadata

    async def save_page_as_image(self, file_path: str, page_num: int, output_path: str, dpi: int = 300) -> str:
        """
        Save a PDF page as an image.
        
        Args:
            file_path: Path to the PDF file
            page_num: Page number to extract (0-based)
            output_path: Path to save the image
            dpi: DPI for the rendered image (default: 300)
            
        Returns:
            Path to the saved image
            
        Raises:
            FileNotFoundError: If the file does not exist
            IndexError: If the page number is out of range
            Exception: For any other errors during extraction
        """
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Use run_in_executor to move CPU-bound work off the main thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._save_page_as_image, file_path, page_num, output_path, dpi
            )
            
            return result
        except Exception as e:
            self.logger.error(f"Error saving page as image: {str(e)}")
            raise
            
    def _save_page_as_image(self, file_path: str, page_num: int, output_path: str, dpi: int = 300) -> str:
        """
        Internal method to save a PDF page as an image.
        This method runs in a separate thread.
        
        Args:
            file_path: Path to the PDF file
            page_num: Page number to extract (0-based)
            output_path: Path to save the image
            dpi: DPI for the rendered image
            
        Returns:
            Path to the saved image
        """
        with fitz.open(file_path) as doc:
            if page_num < 0 or page_num >= len(doc):
                raise IndexError(f"Page number {page_num} out of range (0-{len(doc)-1})")
                
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            pixmap.save(output_path)
            
            self.logger.info(f"Saved page {page_num} as image: {output_path}")
            return output_path


class ArchitecturalExtractor(PyMuPdfExtractor):
    """Specialized extractor for architectural drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    @time_operation("extraction")
    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract architectural-specific content from PDF."""
        # Get base extraction using parent method
        result = await super().extract(file_path)
        
        if not result.success:
            return result
            
        # Enhance extraction with architectural-specific processing
        try:
            # Extract room information more effectively
            enhanced_text = self._enhance_room_information(result.raw_text)
            result.raw_text = enhanced_text
            
            # Prioritize tables containing room schedules, door schedules, etc.
            prioritized_tables = self._prioritize_architectural_tables(result.tables)
            result.tables = prioritized_tables
            
            self.logger.info(f"Enhanced architectural extraction for {file_path}")
            return result
        except Exception as e:
            self.logger.warning(f"Error in architectural enhancement for {file_path}: {str(e)}")
            # Fall back to base extraction on error
            return result
    
    def _enhance_room_information(self, text: str) -> str:
        """Extract and highlight room information in text."""
        # Add a marker for room information
        if "room" in text.lower() or "space" in text.lower():
            text = "ROOM INFORMATION DETECTED:\n" + text
        return text
    
    def _prioritize_architectural_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize architectural tables by type."""
        # Prioritize tables likely to be room schedules
        room_tables = []
        other_tables = []
        
        for table in tables:
            content = table.get("content", "").lower()
            if "room" in content or "space" in content or "finish" in content:
                room_tables.append(table)
            else:
                other_tables.append(table)
                
        return room_tables + other_tables


class ElectricalExtractor(PyMuPdfExtractor):
    """Specialized extractor for electrical drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    @time_operation("extraction")
    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract electrical-specific content from PDF."""
        # Get base extraction using parent method
        result = await super().extract(file_path)
        
        if not result.success:
            return result
            
        # Enhance extraction with electrical-specific processing
        try:
            # Focus on panel schedules and circuit information
            enhanced_text = self._enhance_panel_information(result.raw_text)
            result.raw_text = enhanced_text
            
            # Prioritize tables containing panel schedules
            prioritized_tables = self._prioritize_electrical_tables(result.tables)
            result.tables = prioritized_tables
            
            self.logger.info(f"Enhanced electrical extraction for {file_path}")
            return result
        except Exception as e:
            self.logger.warning(f"Error in electrical enhancement for {file_path}: {str(e)}")
            # Fall back to base extraction on error
            return result
    
    def _enhance_panel_information(self, text: str) -> str:
        """Extract and highlight panel information in text."""
        # Add a marker for panel information
        if "panel" in text.lower() or "circuit" in text.lower():
            text = "PANEL INFORMATION DETECTED:\n" + text
        return text
    
    def _prioritize_electrical_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize electrical tables - panel schedules first."""
        # Prioritize tables likely to be panel schedules
        panel_tables = []
        other_tables = []
        
        for table in tables:
            content = table.get("content", "").lower()
            if "circuit" in content or "panel" in content:
                panel_tables.append(table)
            else:
                other_tables.append(table)
                
        return panel_tables + other_tables


class MechanicalExtractor(PyMuPdfExtractor):
    """Specialized extractor for mechanical drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    @time_operation("extraction")
    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract mechanical-specific content from PDF."""
        # Use parent extraction method
        result = await super().extract(file_path)
        
        if not result.success:
            return result
            
        # Enhance extraction with mechanical-specific processing
        try:
            # Focus on equipment schedules
            enhanced_text = self._enhance_equipment_information(result.raw_text)
            result.raw_text = enhanced_text
            
            # Prioritize tables containing equipment schedules
            prioritized_tables = self._prioritize_mechanical_tables(result.tables)
            result.tables = prioritized_tables
            
            self.logger.info(f"Enhanced mechanical extraction for {file_path}")
            return result
        except Exception as e:
            self.logger.warning(f"Error in mechanical enhancement for {file_path}: {str(e)}")
            # Fall back to base extraction on error
            return result
    
    def _enhance_equipment_information(self, text: str) -> str:
        """Extract and highlight equipment information in text."""
        # Add a marker for equipment information
        if "equipment" in text.lower() or "hvac" in text.lower() or "cfm" in text.lower():
            text = "EQUIPMENT INFORMATION DETECTED:\n" + text
        return text
    
    def _prioritize_mechanical_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize mechanical tables - equipment schedules first."""
        # Simple heuristic - look for equipment-related terms
        equipment_tables = []
        other_tables = []
        
        for table in tables:
            content = table.get("content", "").lower()
            if any(term in content for term in ["equipment", "hvac", "cfm", "tonnage"]):
                equipment_tables.append(table)
            else:
                other_tables.append(table)
                
        return equipment_tables + other_tables


def create_extractor(drawing_type: str, logger: Optional[logging.Logger] = None) -> PdfExtractor:
    """
    Factory function to create the appropriate extractor based on drawing type.
    
    Args:
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        logger: Optional logger instance
        
    Returns:
        Appropriate PdfExtractor implementation
    """
    drawing_type = drawing_type.lower() if drawing_type else ""
    
    if "architectural" in drawing_type:
        return ArchitecturalExtractor(logger)
    elif "electrical" in drawing_type:
        return ElectricalExtractor(logger)
    elif "mechanical" in drawing_type:
        return MechanicalExtractor(logger)
    else:
        # Default to the base extractor for other types
        return PyMuPdfExtractor(logger) 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/__init__.py
```py
# Processing package initialization

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/storage_service.py
```py
"""
Storage service interface and implementations for saving processing results.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, BinaryIO
import os
import json
import logging
import aiofiles
import asyncio


class StorageService(ABC):
    """
    Abstract base class defining the interface for storage services.
    """
    @abstractmethod
    async def save_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Save JSON data to a file.
        
        Args:
            data: JSON-serializable data to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def save_text(self, text: str, file_path: str) -> bool:
        """
        Save text data to a file.
        
        Args:
            text: Text content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def save_binary(self, data: bytes, file_path: str) -> bool:
        """
        Save binary data to a file.
        
        Args:
            data: Binary content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read JSON data from a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Parsed JSON data if successful, None otherwise
        """
        pass


class FileSystemStorage(StorageService):
    """
    Storage service implementation using the local file system.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def save_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Save JSON data to a file.
        
        Args:
            data: JSON-serializable data to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file asynchronously
            async with aiofiles.open(file_path, 'w') as f:
                json_str = json.dumps(data, indent=2)
                await f.write(json_str)
                
            self.logger.info(f"Successfully saved JSON to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON to {file_path}: {str(e)}")
            return False

    async def save_text(self, text: str, file_path: str) -> bool:
        """
        Save text data to a file.
        
        Args:
            text: Text content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file asynchronously
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(text)
                
            self.logger.info(f"Successfully saved text to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving text to {file_path}: {str(e)}")
            return False

    async def save_binary(self, data: bytes, file_path: str) -> bool:
        """
        Save binary data to a file.
        
        Args:
            data: Binary content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file asynchronously
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
                
            self.logger.info(f"Successfully saved binary data to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving binary data to {file_path}: {str(e)}")
            return False

    async def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read JSON data from a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Parsed JSON data if successful, None otherwise
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                return None
                
            # Read the file asynchronously
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                
            # Parse JSON
            data = json.loads(content)
            self.logger.info(f"Successfully read JSON from {file_path}")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from {file_path}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading JSON from {file_path}: {str(e)}")
            return None 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompt_registry.py
```py
"""
Registry system for managing prompt templates.
"""
from typing import Dict, Callable, Optional

# Define prompt registry as a dictionary of factories
PROMPT_REGISTRY: Dict[str, Callable[[], str]] = {}

def register_prompt(category: str, subtype: Optional[str] = None):
    """
    Decorator to register a prompt factory function.
    
    Args:
        category: Drawing category (e.g., "Electrical")
        subtype: Drawing subtype (e.g., "PanelSchedule")
        
    Returns:
        Decorator function that registers the decorated function
    """
    key = f"{category}_{subtype}" if subtype else category
    
    def decorator(func: Callable[[], str]):
        PROMPT_REGISTRY[key.upper()] = func
        return func
    
    return decorator

def get_registered_prompt(drawing_type: str) -> str:
    """
    Get prompt using registry with fallbacks.
    
    Args:
        drawing_type: Type of drawing (e.g., "Electrical_PanelSchedule")
        
    Returns:
        Prompt template string
    """
    # Handle case where drawing_type is None
    if not drawing_type:
        return PROMPT_REGISTRY.get("GENERAL", lambda: "")()
        
    # Normalize the key
    key = drawing_type.upper().replace(" ", "_")
    
    # Try exact match first
    if key in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[key]()
    
    # Try main category
    main_type = key.split("_")[0]
    if main_type in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[main_type]()
    
    # Fall back to general
    return PROMPT_REGISTRY.get("GENERAL", lambda: "")()

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompt_types.py
```py
from enum import Enum, auto

class DrawingCategory(Enum):
    """Main drawing categories."""
    ARCHITECTURAL = "Architectural"
    ELECTRICAL = "Electrical"
    MECHANICAL = "Mechanical"
    PLUMBING = "Plumbing"
    GENERAL = "General"
    SPECIFICATIONS = "Specifications"

class ArchitecturalSubtype(Enum):
    """Architectural drawing subtypes."""
    ROOM = "ROOM"
    CEILING = "CEILING"
    WALL = "WALL"
    DOOR = "DOOR"
    DETAIL = "DETAIL"
    DEFAULT = "DEFAULT"

class ElectricalSubtype(Enum):
    """Electrical drawing subtypes."""
    PANEL_SCHEDULE = "PANEL_SCHEDULE"
    LIGHTING = "LIGHTING"
    POWER = "POWER"
    FIREALARM = "FIREALARM"
    TECHNOLOGY = "TECHNOLOGY"
    SPEC = "SPEC"
    DEFAULT = "DEFAULT"

class MechanicalSubtype(Enum):
    """Mechanical drawing subtypes."""
    EQUIPMENT = "EQUIPMENT"
    VENTILATION = "VENTILATION"
    PIPING = "PIPING"
    DEFAULT = "DEFAULT"

class PlumbingSubtype(Enum):
    """Plumbing drawing subtypes."""
    FIXTURE = "FIXTURE"
    EQUIPMENT = "EQUIPMENT"
    PIPE = "PIPE"
    DEFAULT = "DEFAULT"

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompt_templates.py
```py
"""
Main interface module for accessing prompt templates.
"""

from typing import Dict, Optional

# Import prompt dictionaries from each category
from templates.prompts.architectural import ARCHITECTURAL_PROMPTS
from templates.prompts.electrical import ELECTRICAL_PROMPTS
from templates.prompts.mechanical import MECHANICAL_PROMPTS
from templates.prompts.plumbing import PLUMBING_PROMPTS
from templates.prompts.general import GENERAL_PROMPT

# Import registry for more flexible prompt retrieval
from templates.prompt_registry import get_registered_prompt

# Mapping of main drawing types to prompt dictionaries (for backward compatibility)
PROMPT_CATEGORIES = {
    "Architectural": ARCHITECTURAL_PROMPTS,
    "Electrical": ELECTRICAL_PROMPTS, 
    "Mechanical": MECHANICAL_PROMPTS,
    "Plumbing": PLUMBING_PROMPTS
}

def get_prompt_template(drawing_type: str) -> str:
    """
    Get the appropriate prompt template based on drawing type.
    
    Args:
        drawing_type: Type of drawing (e.g., "Architectural", "Electrical_PanelSchedule")
        
    Returns:
        Prompt template string appropriate for the drawing type
    """
    # Default to general prompt if no drawing type provided
    if not drawing_type:
        return GENERAL_PROMPT
    
    # Try to get prompt from registry first (preferred method)
    registered_prompt = get_registered_prompt(drawing_type)
    if registered_prompt:
        return registered_prompt
    
    # Legacy fallback using dictionaries
    # Parse drawing type to determine category and subtype
    parts = drawing_type.split('_', 1)
    main_type = parts[0]
    
    # If main type not recognized, return general prompt
    if main_type not in PROMPT_CATEGORIES:
        return GENERAL_PROMPT
    
    # Get prompt dictionary for this main type
    prompt_dict = PROMPT_CATEGORIES[main_type]
    
    # Determine subtype (if any)
    subtype = parts[1].upper() if len(parts) > 1 else "DEFAULT"
    
    # Return the specific subtype prompt if available, otherwise the default for this category
    return prompt_dict.get(subtype, prompt_dict["DEFAULT"])

def get_available_subtypes(main_type: Optional[str] = None) -> Dict[str, list]:
    """
    Get available subtypes for a main drawing type or all types.
    
    Args:
        main_type: Optional main drawing type (e.g., "Architectural")
        
    Returns:
        Dictionary of available subtypes by main type
    """
    if main_type and main_type in PROMPT_CATEGORIES:
        # Return subtypes for specific main type
        return {main_type: list(PROMPT_CATEGORIES[main_type].keys())}
    
    # Return all subtypes by main type
    return {category: list(prompts.keys()) for category, prompts in PROMPT_CATEGORIES.items()}

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/__init__.py
```py
"""
Prompt template package for different drawing types.
"""

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/base_templates.py
```py
"""
Base prompt templates to reduce duplication across specific drawing types.
"""

BASE_DRAWING_TEMPLATE = """
You are extracting information from a {drawing_type} drawing.
Document ALL elements following this general structure, adapting to project-specific terminology.

EXTRACTION PRIORITIES:
1. Identify and extract ALL {element_type} elements
2. Document specifications EXACTLY as shown for each element
3. Preserve ALL notes, reference numbers, and special requirements

{specific_instructions}

EXAMPLE STRUCTURE (adapt based on what you find in the drawing):
{example_structure}

CRITICAL INSTRUCTIONS:
1. CAPTURE everything in the drawing
2. PRESERVE original terminology and organization
3. GROUP similar elements together in logical sections
4. DOCUMENT all specifications and detailed information
5. Ensure your entire response is a single, valid JSON object.

{industry_context}
"""

SCHEDULE_TEMPLATE = """
You are extracting {schedule_type} information from {drawing_category} drawings. 
Document ALL {item_type} following the structure in this example, while adapting to project-specific terminology.

EXTRACTION PRIORITIES:
1. Capture EVERY {item_type} with ALL specifications
2. Document ALL {key_properties} EXACTLY as shown
3. Include ALL notes, requirements, and special conditions

EXAMPLE OUTPUT STRUCTURE (field names may vary by project):
{example_structure}

CRITICAL INSTRUCTIONS:
1. EXTRACT all {item_type}s shown on the {source_location}
2. PRESERVE exact {preservation_focus}
3. INCLUDE all technical specifications and requirements 
4. ADAPT the structure to match this specific drawing
5. MAINTAIN the overall hierarchical organization shown in the example
6. EXTRACT ALL metadata from the drawing's title block, including drawing_number, title, revision, date, job_number, and project_name
7. Format your output as a complete, valid JSON object.

{stake_holders} rely on this information for {use_case}.
Complete accuracy is essential for {critical_purpose}.
"""

def create_general_template(drawing_type, element_type, instructions, example, context):
    """Create a general prompt template with the provided parameters."""
    return BASE_DRAWING_TEMPLATE.format(
        drawing_type=drawing_type,
        element_type=element_type,
        specific_instructions=instructions,
        example_structure=example,
        industry_context=context
    )

def create_schedule_template(
    schedule_type, 
    drawing_category,
    item_type,
    key_properties,
    example_structure,
    source_location,
    preservation_focus,
    stake_holders,
    use_case,
    critical_purpose
):
    """Create a schedule template with the provided parameters."""
    return SCHEDULE_TEMPLATE.format(
        schedule_type=schedule_type,
        drawing_category=drawing_category,
        item_type=item_type,
        key_properties=key_properties,
        example_structure=example_structure,
        source_location=source_location,
        preservation_focus=preservation_focus,
        stake_holders=stake_holders,
        use_case=use_case,
        critical_purpose=critical_purpose
    )

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/electrical.py
```py
"""
Electrical prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Electrical")
def default_electrical_prompt():
    """Default prompt for electrical drawings."""
    return create_general_template(
        drawing_type="ELECTRICAL",
        element_type="electrical",
        instructions="""
Focus on identifying and extracting ALL electrical elements (panels, fixtures, devices, connections, etc.).
""",
        example="""
{
  "ELECTRICAL": {
    "metadata": {
      "drawing_number": "E101",
      "title": "ELECTRICAL FLOOR PLAN",
      "date": "2023-05-15",
      "revision": "2"
    },
    "elements": {
      "panels": [],
      "fixtures": [],
      "devices": [],
      "connections": []
    },
    "notes": []
  }
}
""",
        context="Electrical engineers and installers rely on this information for proper system design and construction."
    )

@register_prompt("Electrical", "PANEL_SCHEDULE")
def panel_schedule_prompt():
    """Prompt for electrical panel schedules."""
    return create_schedule_template(
        schedule_type="panel schedule",
        drawing_category="electrical",
        item_type="circuit",
        key_properties="circuit assignments, breaker sizes, and load descriptions",
        example_structure="""
{
  "ELECTRICAL": {
    "PANEL_SCHEDULE": {
      "panel": {
        "name": "K1S",
        "voltage": "120/208 Wye",
        "phases": 3,
        "main_breaker": "30 A Main Breaker",
        "marks": "K1S",
        "aic_rating": "65K",
        "type": "MLO",
        "rating": "600 A",
        "specifications": {
          "sections": "1 Section(s)",
          "nema_enclosure": "Nema 1 Enclosure",
          "amps": "125 Amps",
          "phases": "3 Phase 4 Wire",
          "voltage": "480Y/277V",
          "frequency": "50/60 Hz",
          "interrupt_rating": "65kA Fully Rated",
          "incoming_feed": "Bottom",
          "fed_from": "1 inch conduit with 4#10's and 1#10 ground",
          "mounting": "Surface Mounted",
          "circuits_count": 12
        },
        "circuits": [
          {
            "circuit": 1,
            "load_name": "E-117(*)",
            "trip": "15 A",
            "poles": 1,
            "wires": 4,
            "info": "GFCI Circuit Breaker",
            "load_classification": "Kitchen Equipment",
            "connected_load": "1200 VA",
            "demand_factor": "65.00%",
            "equipment_ref": "E01",
            "room_id": ["Room_2104", "Room_2105"]
          }
        ],
        "panel_totals": {
          "total_connected_load": "5592 VA",
          "total_estimated_demand": "3635 VA", 
          "total_connected_amps": "16 A",
          "total_estimated_demand_amps": "10 A"
        }
      }
    }
  }
}
""",
        source_location="panel schedule",
        preservation_focus="circuit numbers, trip sizes, and load descriptions",
        stake_holders="Electrical engineers and installers",
        use_case="critical electrical system design and installation",
        critical_purpose="preventing safety hazards and ensuring proper function"
    )

@register_prompt("Electrical", "LIGHTING")
def lighting_fixture_prompt():
    """Prompt for lighting fixtures."""
    return create_schedule_template(
        schedule_type="lighting fixture",
        drawing_category="electrical",
        item_type="lighting fixture",
        key_properties="CRITICAL: Extract all metadata from the drawing's title block, including drawing_number, title, revision, date, job_number, and project_name, placing it in the 'metadata' object. Also capture model numbers, descriptions, and performance data for fixtures.",
        example_structure="""
{
  "ELECTRICAL": {
    "metadata": {
      "drawing_number": "E1.00",
      "title": "LIGHTING - FLOOR LEVEL",
      "revision": "3",
      "date": "08/15/2024",
      "job_number": "30J7925",
      "project_name": "ELECTRIC SHUFFLE"
    },
    "LIGHTING_FIXTURE": {
      "type_mark": "CL-US-18",
      "count": 13,
      "manufacturer": "Mullan",
      "product_number": "MLP323",
      "description": "Essense Vintage Prismatic Glass Pendant Light",
      "finish": "Antique Brass",
      "lamp_type": "E27, 40W, 120V, 2200K",
      "mounting": "Ceiling",
      "dimensions": "15.75\\" DIA x 13.78\\" HEIGHT",
      "location": "Restroom Corridor and Raised Playspace",
      "wattage": "40W",
      "ballast_type": "LED Driver",
      "dimmable": "Yes",
      "remarks": "Refer to architectural",
      "catalog_series": "RA1-24-A-35-F2-M-C"
    },
    "LIGHTING_ZONE": {
      "zone": "Z1",
      "area": "Dining 103",
      "circuit": "L1-13",
      "fixture_type": "LED",
      "dimming_control": "ELV",
      "notes": "Shuffleboard Tables 3,4",
      "quantities_or_linear_footage": "16"
    }
  }
}
""",
        source_location="fixture schedule",
        preservation_focus="fixture types, models, and specifications",
        stake_holders="Lighting designers and electrical contractors",
        use_case="product selection, energy calculations, and installation",
        critical_purpose="proper lighting design and code compliance"
    )

@register_prompt("Electrical", "POWER")
def power_connection_prompt():
    """Prompt for power connections."""
    return create_schedule_template(
        schedule_type="power connection",
        drawing_category="electrical",
        item_type="power connection",
        key_properties="circuit assignments, breaker sizes, and loads",
        example_structure="""
{
  "ELECTRICAL": {
    "POWER_CONNECTION": {
      "item": "E101A",
      "connection_type": "JUNCTION BOX",
      "quantity": 2,
      "description": "Door Heater / Conden. Drain Line Heater / Heated Vent Port",
      "breaker_size": "15A",
      "voltage": "120",
      "phase": 1,
      "mounting": "Ceiling",
      "height": "108\\"",
      "current": "7.4A",
      "remarks": "Branch to connection, verify compressor location"
    },
    "HOME_RUN": {
      "id": "HR1",
      "circuits": [
        "28N",
        "47",
        "49",
        "51",
        "N"
      ]
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="circuit assignments and specifications",
        stake_holders="Electrical contractors",
        use_case="proper installation and coordination",
        critical_purpose="proper power distribution and equipment function"
    )

@register_prompt("Electrical", "SPEC")
def electrical_spec_prompt():
    """Prompt for electrical specifications."""
    return create_schedule_template(
        schedule_type="specification",
        drawing_category="electrical",
        item_type="specification section",
        key_properties="requirements, standards, and installation details",
        example_structure="""
{
  "ELECTRICAL": {
    "ELECTRICAL_SPEC": {
      "section": "16050",
      "title": "BASIC ELECTRICAL MATERIALS AND METHODS",
      "details": [
        "Installation completeness",
        "Compliance with NEC, OSHA, IEEE, UL, NFPA, and local codes",
        "Submittals for proposed schedule and deviations",
        "Listed and labeled products per NFPA 70",
        "Uniformity of manufacturer for similar equipment",
        "Coordination with construction and other trades",
        "Trenching and backfill requirements",
        "Warranty: Minimum one year",
        "Safety guards and equipment arrangement",
        "Protection of materials and apparatus"
      ],
      "subsection_details": {
        "depth_and_backfill_requirements": {
          "details": [
            "Trenches support on solid ground",
            "First backfill layer: 6 inches above the top of the conduit with select fill or pea gravel",
            "Minimum buried depth: 24 inches below finished grade for underground cables per NEC"
          ]
        },
        "wiring_requirements_and_wire_sizing": {
          "section": "16123",
          "details": [
            "Wire and cable for 600 volts and less",
            "Use THHN in metallic conduit for dry interior locations",
            "Use THWN in non-metallic conduit for underground installations",
            "Solid conductors for feeders and branch circuits 10 AWG and smaller",
            "Stranded conductors for control circuits",
            "Minimum conductor size for power and lighting circuits: 12 AWG",
            "Use 10 AWG for longer branch circuits as specified",
            "Conductor sizes are based on copper unless indicated as aluminum"
          ]
        }
      }
    }
  }
}
""",
        source_location="document",
        preservation_focus="section numbers, titles, and requirements",
        stake_holders="Contractors and installers",
        use_case="code compliance and proper installation",
        critical_purpose="meeting building code requirements"
    )

# Dictionary of all electrical prompts for backward compatibility
ELECTRICAL_PROMPTS = {
    "DEFAULT": default_electrical_prompt(),
    "PANEL_SCHEDULE": panel_schedule_prompt(),
    "LIGHTING": lighting_fixture_prompt(),
    "POWER": power_connection_prompt(),
    "SPEC": electrical_spec_prompt()
}

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/general.py
```py
"""
General prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template

@register_prompt("General")
def general_prompt():
    """General purpose prompt for any drawing type."""
    return create_general_template(
        drawing_type="construction",
        element_type="construction",
        instructions="""
Extract ALL elements following a logical structure, while adapting to project-specific terminology.
""",
        example="""
{
  "metadata": {
    "drawing_number": "X101",
    "title": "DRAWING TITLE",
    "date": "2023-05-15",
    "revision": "2"
  },
  "schedules": [
    {
      "type": "schedule_type",
      "data": [
        {"item_id": "X1", "description": "Item description", "specifications": "Technical details"}
      ]
    }
  ],
  "notes": ["Note 1", "Note 2"]
}
""",
        context="Engineers need EVERY element and specification value EXACTLY as shown - complete accuracy is essential for proper system design, ordering, and installation."
    )

# Register general prompt to ensure it's always available
GENERAL_PROMPT = general_prompt()

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/mechanical.py
```py
"""
Mechanical prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Mechanical")
def default_mechanical_prompt():
    """Default prompt for mechanical drawings."""
    return create_general_template(
        drawing_type="MECHANICAL",
        element_type="mechanical",
        instructions="""
Focus on identifying and extracting ALL mechanical elements (equipment, ductwork, piping, etc.) with their specifications.
""",
        example="""
{
  "MECHANICAL": {
    "metadata": {
      "drawing_number": "M101",
      "title": "MECHANICAL PLAN",
      "date": "2023-05-15",
      "revision": "2"
    },
    "equipment": {
      "air_handling_units": [],
      "fans": [],
      "vav_boxes": [],
      "pumps": []
    },
    "distribution": {
      "ductwork": [],
      "piping": []
    },
    "notes": []
  }
}
""",
        context="Mechanical engineers and contractors rely on this information for proper system design, coordination, and installation."
    )

@register_prompt("Mechanical", "EQUIPMENT")
def equipment_schedule_prompt():
    """Prompt for mechanical equipment schedules."""
    return create_schedule_template(
        schedule_type="equipment",
        drawing_category="mechanical",
        item_type="mechanical equipment",
        key_properties="model numbers, capacities, and performance data",
        example_structure="""
{
  "MECHANICAL": {
    "EQUIPMENT": {
      "equipment_id": "AHU-1",
      "type": "AIR HANDLING UNIT",
      "manufacturer": "Trane",
      "model": "CSAA012",
      "capacity": {
        "cooling": "12 Tons",
        "heating": "150 MBH",
        "airflow": "4,800 CFM"
      },
      "electrical": {
        "voltage": "460/3/60",
        "fla": "22.4 A",
        "mca": "28 A",
        "mocp": "45 A"
      },
      "dimensions": "96\" L x 60\" W x 72\" H",
      "weight": "2,500 lbs",
      "location": "Roof",
      "accessories": [
        "Economizer",
        "VFD",
        "MERV 13 Filters"
      ],
      "notes": "Provide seismic restraints per detail M5.1"
    }
  }
}
""",
        source_location="schedule",
        preservation_focus="equipment tags, specifications, and performance data",
        stake_holders="Mechanical engineers and contractors",
        use_case="equipment ordering and installation coordination",
        critical_purpose="proper mechanical system function and energy efficiency"
    )

@register_prompt("Mechanical", "VENTILATION")
def ventilation_prompt():
    """Prompt for ventilation elements."""
    return create_schedule_template(
        schedule_type="ventilation",
        drawing_category="mechanical",
        item_type="ventilation element",
        key_properties="airflow rates, dimensions, and connection types",
        example_structure="""
{
  "MECHANICAL": {
    "VENTILATION": {
      "element_id": "EF-1",
      "type": "EXHAUST FAN",
      "airflow": "1,200 CFM",
      "static_pressure": "0.75 in. w.g.",
      "motor": {
        "horsepower": "1/2 HP",
        "voltage": "120/1/60",
        "fla": "4.8 A"
      },
      "location": "Roof",
      "serving": "Restrooms 101, 102, 103",
      "dimensions": "24\" x 24\"",
      "mounting": "Curb mounted",
      "duct_connection": "16\" diameter",
      "controls": "Controlled by Building Management System",
      "operation": "Continuous during occupied hours"
    },
    "AIR_TERMINAL": {
      "element_id": "VAV-1",
      "type": "VARIABLE AIR VOLUME BOX",
      "size": "8 inch",
      "max_airflow": "450 CFM",
      "min_airflow": "100 CFM",
      "heating_capacity": "5 kW",
      "pressure_drop": "0.25 in. w.g.",
      "location": "Above ceiling in Room 201",
      "controls": "DDC controller with pressure-independent control"
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="airflow rates, equipment specifications, and control requirements",
        stake_holders="Mechanical engineers and contractors",
        use_case="ventilation system design and balancing",
        critical_purpose="proper indoor air quality and comfort"
    )

@register_prompt("Mechanical", "PIPING")
def piping_prompt():
    """Prompt for mechanical piping."""
    return create_schedule_template(
        schedule_type="piping",
        drawing_category="mechanical",
        item_type="piping system",
        key_properties="pipe sizes, materials, and flow rates",
        example_structure="""
{
  "MECHANICAL": {
    "PIPING": {
      "system_type": "CHILLED WATER",
      "pipe_material": "Copper Type L",
      "insulation": {
        "type": "Closed cell foam",
        "thickness": "1 inch",
        "jacket": "All-service vapor barrier jacket"
      },
      "design_pressure": "125 PSI",
      "design_temperature": "40°F supply, 55°F return",
      "flow_rate": "120 GPM",
      "sizes": [
        {
          "size": "2-1/2 inch",
          "location": "Main distribution",
          "flow_rate": "120 GPM",
          "velocity": "4.5 ft/s"
        },
        {
          "size": "2 inch", 
          "location": "Branch to AHU-1",
          "flow_rate": "60 GPM",
          "velocity": "4.2 ft/s"
        }
      ],
      "accessories": [
        {
          "type": "Ball Valve",
          "size": "2 inch",
          "location": "Each branch takeoff",
          "specification": "Bronze body, full port"
        },
        {
          "type": "Flow Meter",
          "size": "2-1/2 inch",
          "location": "Main supply",
          "specification": "Venturi type with pressure ports"
        }
      ],
      "notes": "Provide 3D coordination with all other trades prior to installation"
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="pipe sizes, materials, and system specifications",
        stake_holders="Mechanical engineers and plumbing contractors",
        use_case="piping installation and coordination",
        critical_purpose="proper fluid distribution and system performance"
    )

# Dictionary of all mechanical prompts for backward compatibility
MECHANICAL_PROMPTS = {
    "DEFAULT": default_mechanical_prompt(),
    "EQUIPMENT": equipment_schedule_prompt(),
    "VENTILATION": ventilation_prompt(),
    "PIPING": piping_prompt()
}

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/plumbing.py
```py
"""
Plumbing prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Plumbing")
def default_plumbing_prompt():
    """Default prompt for plumbing drawings."""
    return create_general_template(
        drawing_type="PLUMBING",
        element_type="plumbing",
        instructions="""
Focus on identifying and extracting ALL plumbing elements with a comprehensive structure that includes:

1. Complete fixture schedules with every fixture type (sinks, water closets, urinals, lavatories, drains, cleanouts, etc.)
2. All equipment (water heaters, pumps, mixing valves, shock absorbers, etc.)
3. Pipe materials and connection requirements
4. All general notes, insulation notes, and special requirements
5. Capture each distinct schedule as a separate section with appropriate field structure

Pay special attention to equipment like pumps (CP), mixing valves (TM), and shock absorbers (SA) which must be captured even if located in separate tables or areas of the drawing.
""",
        example="""
{
  "PLUMBING": {
    "metadata": {
      "drawing_number": "P601",
      "title": "PLUMBING SCHEDULES",
      "date": "2023-05-15",
      "revision": "2"
    },
    "FIXTURE": [
      {
        "fixture_id": "S1",
        "description": "SINGLE COMPARTMENT SINK",
        "manufacturer": "McGuire Supplies",
        "model": "N/A",
        "mounting": "Contractor installed",
        "type": "17 gauge brass P-trap",
        "connections": {
          "cold_water": "1/2 inch",
          "waste": "2 inch",
          "vent": "2 inch",
          "hot_water": "1/2 inch"
        },
        "notes": "Contractor installed"
      }
    ],
    "WATER_HEATER": [
      {
        "mark": "WH-1",
        "location": "Mechanical Room",
        "manufacturer": "A.O. Smith",
        "model": "DRE-120-24",
        "specifications": {
          "storage_gallons": "120",
          "operating_water_temp": "140°F",
          "recovery_rate": "99 GPH"
        },
        "mounting": "Floor mounted",
        "notes": ["Provide T&P relief valve"]
      }
    ],
    "PUMP": [
      {
        "mark": "CP",
        "location": "Mechanical Room",
        "serves": "Hot Water Recirculation",
        "type": "IN-LINE",
        "gpm": "10",
        "tdh_ft": "20",
        "hp": "1/2",
        "electrical": "120V/1PH/60HZ",
        "manufacturer": "Bell & Gossett",
        "model": "Series 100"
      }
    ],
    "MIXING_VALVE": [
      {
        "designation": "TM",
        "location": "Mechanical Room",
        "manufacturer": "Powers",
        "model": "LFLM495-1",
        "notes": "Master thermostatic mixing valve"
      }
    ],
    "SHOCK_ABSORBER": [
      {
        "mark": "SA-A",
        "fixture_units": "1-11",
        "manufacturer": "Sioux Chief",
        "model": "660-A"
      }
    ],
    "MATERIAL_LEGEND": {
      "SANITARY SEWER PIPING": "CAST IRON OR SCHEDULE 40 PVC",
      "DOMESTIC WATER PIPING": "TYPE L COPPER"
    },
    "GENERAL_NOTES": [
      "A. All fixtures shall be installed per manufacturer's recommendations."
    ],
    "INSULATION_NOTES": [
      "A. Insulate all domestic hot water piping with 1\" thick fiberglass insulation."
    ]
  }
}
""",
        context="Plumbing engineers and contractors rely on ALL schedules, notes, and specifications for proper system design, coordination, and installation. Missing information can lead to serious installation issues or code violations."
    )

@register_prompt("Plumbing", "FIXTURE")
def fixture_schedule_prompt():
    """Prompt for plumbing fixture schedules."""
    return create_schedule_template(
        schedule_type="fixture",
        drawing_category="plumbing",
        item_type="plumbing fixture and equipment",
        key_properties="identifiers, models, specifications, connections, notes and all related schedules",
        example_structure="""
{
  "PLUMBING": {
    "metadata": {
      "drawing_number": "P601",
      "title": "PLUMBING SCHEDULES",
      "date": "2023-05-15",
      "revision": "2"
    },
    "FIXTURE": [
      {
        "fixture_id": "S1",
        "description": "SINGLE COMPARTMENT SINK",
        "manufacturer": "McGuire Supplies",
        "model": "N/A",
        "mounting": "Contractor installed",
        "type": "17 gauge brass P-trap",
        "connections": {
          "cold_water": "1/2 inch",
          "waste": "2 inch",
          "vent": "2 inch",
          "hot_water": "1/2 inch"
        },
        "location": "Refer to architect for specification",
        "notes": "Contractor installed"
      },
      {
        "fixture_id": "SW-01",
        "description": "WATER CLOSET",
        "manufacturer": "American Standard",
        "model": "2234.001",
        "mounting": "Floor mounted",
        "type": "1.28 GPF, elongated bowl",
        "connections": {
          "cold_water": "1/2 inch",
          "waste": "4 inch",
          "vent": "2 inch"
        },
        "location": "Various",
        "notes": "Provide floor flange and wax ring"
      }
    ],
    "WATER_HEATER": [
      {
        "mark": "WH-1",
        "location": "Mechanical Room",
        "manufacturer": "A.O. Smith",
        "model": "DRE-120-24",
        "specifications": {
          "storage_gallons": "120",
          "operating_water_temp": "140°F",
          "tank_dimensions": "26\" DIA x 71\" H",
          "recovery_rate": "99 GPH at 100°F rise",
          "electric_power": "480V, 3PH, 60HZ",
          "kW_input": "24"
        },
        "mounting": "Floor mounted",
        "notes": [
          "Provide T&P relief valve",
          "Provide seismic restraints per detail P5.1",
          "Provide expansion tank"
        ]
      }
    ],
    "PUMP": [
      {
        "mark": "CP",
        "location": "Mechanical Room",
        "serves": "Hot Water Recirculation",
        "type": "IN-LINE",
        "gpm": "10",
        "tdh_ft": "20",
        "hp": "1/2",
        "rpm": "1750",
        "electrical": "120V/1PH/60HZ",
        "manufacturer": "Bell & Gossett",
        "model": "Series 100",
        "notes": "Provide spring isolation hangers"
      }
    ],
    "MIXING_VALVE": [
      {
        "designation": "TM",
        "location": "Mechanical Room",
        "inlet_temp_F": "140",
        "outlet_temp_F": "120",
        "pressure_drop_psi": "5",
        "manufacturer": "Powers",
        "model": "LFLM495-1",
        "notes": "Master thermostatic mixing valve for domestic hot water system"
      }
    ],
    "SHOCK_ABSORBER": [
      {
        "mark": "SA-A",
        "fixture_units": "1-11",
        "manufacturer": "Sioux Chief",
        "model": "660-A",
        "description": "Water hammer arrestor, size A"
      },
      {
        "mark": "SA-B",
        "fixture_units": "12-32",
        "manufacturer": "Sioux Chief",
        "model": "660-B",
        "description": "Water hammer arrestor, size B"
      }
    ],
    "MATERIAL_LEGEND": {
      "SANITARY SEWER PIPING": "CAST IRON OR SCHEDULE 40 PVC",
      "VENT PIPING": "CAST IRON OR SCHEDULE 40 PVC",
      "DOMESTIC WATER PIPING": "TYPE L COPPER",
      "STORM DRAIN PIPING": "CAST IRON OR SCHEDULE 40 PVC"
    },
    "GENERAL_NOTES": [
      "A. All fixtures and equipment shall be installed per manufacturer's recommendations.",
      "B. Verify all rough-in dimensions with architectural drawings and manufacturer's cut sheets.",
      "C. All hot and cold water piping to be insulated per specifications."
    ],
    "INSULATION_NOTES": [
      "A. Insulate all domestic hot water piping with 1\" thick fiberglass insulation.",
      "B. Insulate all domestic cold water piping with 1/2\" thick fiberglass insulation."
    ]
  }
}
""",
        source_location="schedule",
        preservation_focus="ALL fixture types, equipment, pumps, valves, shock absorbers, materials, and notes",
        stake_holders="Plumbing engineers and contractors",
        use_case="comprehensive fixture and equipment selection and installation coordination",
        critical_purpose="proper system function, water conservation, and code compliance"
    )

@register_prompt("Plumbing", "EQUIPMENT")
def equipment_schedule_prompt():
    """Prompt for plumbing equipment schedules."""
    return create_schedule_template(
        schedule_type="equipment",
        drawing_category="plumbing",
        item_type="plumbing equipment",
        key_properties="capacities, connection sizes, and electrical requirements",
        example_structure="""
{
  "PLUMBING": {
    "EQUIPMENT": {
      "equipment_id": "WH-1",
      "type": "WATER HEATER",
      "manufacturer": "A.O. Smith",
      "model": "DRE-120-24",
      "capacity": "120 gallons",
      "heating_input": "24 kW",
      "recovery_rate": "99 GPH at 100°F rise",
      "electrical": {
        "voltage": "480/3/60",
        "full_load_amps": "28.9 A",
        "minimum_circuit_ampacity": "36.1 A",
        "maximum_overcurrent_protection": "40 A"
      },
      "connections": {
        "cold_water_inlet": "1-1/2 inch",
        "hot_water_outlet": "1-1/2 inch",
        "recirculation": "3/4 inch",
        "relief_valve": "1 inch"
      },
      "dimensions": "26\" DIA x 71\" H",
      "weight": "650 lbs empty, 1,650 lbs full",
      "location": "Mechanical Room 151",
      "notes": "Provide seismic restraints per detail P5.1"
    }
  }
}
""",
        source_location="schedule",
        preservation_focus="equipment specifications and performance requirements",
        stake_holders="Plumbing engineers and contractors",
        use_case="equipment selection and installation coordination",
        critical_purpose="proper hot water system function and energy efficiency"
    )

@register_prompt("Plumbing", "PIPE")
def pipe_schedule_prompt():
    """Prompt for plumbing pipe schedules."""
    return create_schedule_template(
        schedule_type="pipe",
        drawing_category="plumbing",
        item_type="pipe",
        key_properties="materials, sizes, and connection methods",
        example_structure="""
{
  "PLUMBING": {
    "PIPE": {
      "system_type": "DOMESTIC WATER",
      "pipe_material": "Copper Type L",
      "insulation": {
        "type": "Fiberglass",
        "thickness": "1 inch for cold water, 1.5 inch for hot water",
        "jacket": "All-service vapor barrier jacket"
      },
      "joining_method": "Soldered, lead-free",
      "design_pressure": "80 PSI working pressure",
      "testing_pressure": "125 PSI for 4 hours",
      "sizes": [
        {
          "size": "3 inch",
          "location": "Main distribution",
          "flow_rate": "150 GPM",
          "velocity": "6.1 ft/s"
        },
        {
          "size": "1-1/2 inch", 
          "location": "Branch to Restrooms",
          "flow_rate": "45 GPM",
          "velocity": "5.8 ft/s"
        }
      ],
      "accessories": [
        {
          "type": "Ball Valve",
          "size": "3 inch",
          "location": "Main shut-off",
          "specification": "Bronze body, full port, 600 WOG"
        },
        {
          "type": "Pressure Reducing Valve",
          "size": "3 inch",
          "location": "Service entrance",
          "specification": "Watts 223, set at 65 PSI"
        }
      ],
      "notes": "Provide water hammer arrestors at all quick-closing valves"
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="pipe materials, sizes, and installation requirements",
        stake_holders="Plumbing engineers and contractors",
        use_case="piping system installation and coordination",
        critical_purpose="proper water distribution and system performance"
    )

# Dictionary of all plumbing prompts for backward compatibility
PLUMBING_PROMPTS = {
    "DEFAULT": default_plumbing_prompt(),
    "FIXTURE": fixture_schedule_prompt(),
    "EQUIPMENT": equipment_schedule_prompt(),
    "PIPE": pipe_schedule_prompt()
}

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/room_templates.py
```py
import json
import os

def load_template(template_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, f"{template_name}_template.json")
    try:
        with open(template_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Template file not found: {template_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {template_path}")
        return {}

def generate_rooms_data(parsed_data, room_type):
    template = load_template(room_type)
    
    metadata = parsed_data.get('metadata', {})
    
    rooms_data = {
        "metadata": metadata,
        "project_name": metadata.get('project', ''),
        "floor_number": '',
        "rooms": []
    }
    
    parsed_rooms = parsed_data.get('rooms', [])
    
    if not parsed_rooms:
        print(f"No rooms found in parsed data for {room_type}.")
        return rooms_data

    for parsed_room in parsed_rooms:
        room_number = str(parsed_room.get('number', ''))
        room_name = parsed_room.get('name', '')
        
        if not room_number or not room_name:
            print(f"Skipping room with incomplete data: {parsed_room}")
            continue
        
        room_data = template.copy()
        room_data['room_id'] = f"Room_{room_number}"
        room_data['room_name'] = f"{room_name}_{room_number}"
        
        # Copy all fields from parsed_room to room_data
        for key, value in parsed_room.items():
            if key not in ['number', 'name']:
                room_data[key] = value
        
        rooms_data['rooms'].append(room_data)
    
    return rooms_data

def process_architectural_drawing(parsed_data, file_path, output_folder):
    """
    Process architectural drawing data (parsed JSON),
    and generate both e_rooms and a_rooms JSON outputs.
    """
    is_reflected_ceiling = "REFLECTED CEILING PLAN" in file_path.upper()
    
    floor_number = ''  # If floor number is available in the future, extract it here
    
    e_rooms_data = generate_rooms_data(parsed_data, 'e_rooms')
    a_rooms_data = generate_rooms_data(parsed_data, 'a_rooms')
    
    e_rooms_file = os.path.join(output_folder, f'e_rooms_details_floor_{floor_number}.json')
    a_rooms_file = os.path.join(output_folder, f'a_rooms_details_floor_{floor_number}.json')
    
    with open(e_rooms_file, 'w') as f:
        json.dump(e_rooms_data, f, indent=2)
    with open(a_rooms_file, 'w') as f:
        json.dump(a_rooms_data, f, indent=2)
    
    return {
        "e_rooms_file": e_rooms_file,
        "a_rooms_file": a_rooms_file,
        "is_reflected_ceiling": is_reflected_ceiling
    }

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/architectural.py
```py
"""
Architectural prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Architectural")
def default_architectural_prompt():
    """Default prompt for architectural drawings."""
    return create_general_template(
        drawing_type="ARCHITECTURAL",
        element_type="architectural",
        instructions="""
Focus on identifying and extracting ALL architectural elements (rooms, walls, doors, finishes, etc.).
""",
        example="""
{
  "ARCHITECTURAL": {
    "metadata": {
      "drawing_number": "A101",
      "title": "FLOOR PLAN",
      "date": "2023-05-15",
      "revision": "2"
    },
    "elements": {
      "rooms": [],
      "walls": [],
      "doors": [],
      "windows": [],
      "finishes": []
    },
    "notes": []
  }
}
""",
        context="Architects, contractors, and other trades rely on this information for coordination and construction."
    )

@register_prompt("Architectural", "ROOM")
def room_schedule_prompt():
    """Prompt for room schedules."""
    return create_schedule_template(
        schedule_type="room",
        drawing_category="architectural",
        item_type="room",
        key_properties="wall types, finishes, and dimensions",
        example_structure="""
{
  "ARCHITECTURAL": {
    "ROOM": {
      "room_id": "Room_2104",
      "room_name": "CONFERENCE 2104",
      "circuits": {
        "lighting": ["21LP-1"],
        "power": ["21LP-17"]
      },
      "light_fixtures": {
        "fixture_ids": ["F3", "F4"],
        "fixture_count": {
          "F3": 14,
          "F4": 2
        }
      },
      "outlets": {
        "regular_outlets": 3,
        "controlled_outlets": 1
      },
      "data": 4,
      "floor_boxes": 2,
      "mechanical_equipment": [
        {
          "mechanical_id": "fpb-21.03"
        }
      ],
      "switches": {
        "type": "vacancy sensor",
        "model": "WSX-PDT",
        "dimming": "0 to 10V",
        "quantity": 2,
        "mounting_type": "wall-mounted",
        "line_voltage": true
      }
    }
  }
}
""",
        source_location="floor plan",
        preservation_focus="room numbers and names",
        stake_holders="Architects, contractors, and other trades",
        use_case="coordination and building construction",
        critical_purpose="proper space planning and construction"
    )

@register_prompt("Architectural", "DOOR")
def door_schedule_prompt():
    """Prompt for door schedules."""
    return create_schedule_template(
        schedule_type="door",
        drawing_category="architectural",
        item_type="door",
        key_properties="hardware, frame types, and dimensions",
        example_structure="""
{
  "ARCHITECTURAL": {
    "DOOR": {
      "door_id": "2100-01",
      "door_type": "A",
      "door_material": "Solid Core Wood",
      "hardware_type": "Standard Hardware",
      "finish": "PT-4 Paint",
      "louvers": "None",
      "dimensions": {
        "height": "7'-9\\"",
        "width": "3'-0\\"",
        "thickness": "1-3/4\\""
      },
      "frame_type": "Type II (Snap-On Cover)",
      "glass_type": "None",
      "notes": "All private office doors to receive coat hook on interior side at 70\" AFF.",
      "use": "Office"
    },
    "DOOR_HARDWARE": {
      "hardware_type": "Standard Hardware",
      "components": [
        {
          "component": "Push/Pull",
          "model": "CRL 84LPBS",
          "finish": "Brushed Stainless",
          "lever_style": "03 Lever",
          "dimensions": "4-1/2\"",
          "type": "Full Side Closer",
          "note": "Integrated with card reader and motion detector",
          "notes": "Bi-Pass configuration"
        }
      ]
    }
  }
}
""",
        source_location="door schedule",
        preservation_focus="door numbers, types, and hardware groups",
        stake_holders="Architects, contractors, and hardware suppliers",
        use_case="procurement and installation",
        critical_purpose="proper door function and security"
    )

@register_prompt("Architectural", "WALL")
def wall_type_prompt():
    """Prompt for wall types."""
    return create_schedule_template(
        schedule_type="wall type",
        drawing_category="architectural",
        item_type="wall assembly",
        key_properties="assembly details, materials, and dimensions",
        example_structure="""
{
  "ARCHITECTURAL": {
    "WALL_TYPE": {
      "wallTypeId": "Type 1A",
      "description": "1/1A - Full Height Partition",
      "structure": {
        "metalDeflectionTrack": {
          "anchoredTo": "Building Structural Deck",
          "fasteners": "Ballistic Pins"
        },
        "studs": {
          "size": "3 5/8\"",
          "gauge": "20 GA",
          "spacing": "24\" O.C.",
          "fasteners": "1/2\" Type S-12 Pan Head Screws"
        },
        "gypsumBoard": {
          "layers": 1,
          "type": "5/8\" Type 'X'",
          "fastening": "Mechanically fastened to both sides of studs with 1\" Type S-12 screws, stagger joints 24\" O.C.",
          "orientation": "Apply vertically"
        },
        "insulation": {
          "type": "Acoustical Batt",
          "thickness": "3 1/2\"",
          "installation": "Continuous, friction fit"
        },
        "plywood": {
          "type": "Fire retardant treated",
          "thickness": "1/2\"",
          "location": "West side"
        }
      },
      "partition_width": "7 5/8\""
    }
  }
}
""",
        source_location="drawings",
        preservation_focus="materials, dimensions, and assembly details",
        stake_holders="Architects, contractors, and other trades",
        use_case="proper construction",
        critical_purpose="code compliance and building performance"
    )

# Dictionary of all architectural prompts for backward compatibility
ARCHITECTURAL_PROMPTS = {
    "DEFAULT": default_architectural_prompt(),
    "ROOM": room_schedule_prompt(),
    "DOOR": door_schedule_prompt(),
    "WALL": wall_type_prompt()
}

```
</file_contents>

<user_instructions>
HELP ME UND
</user_instructions>
