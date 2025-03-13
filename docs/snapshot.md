<file_map>
/Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template
├── config
│   └── settings.py
├── processing
│   ├── __init__.py
│   ├── batch_processor.py
│   ├── file_processor.py
│   └── job_processor.py
├── templates
│   ├── __init__.py
│   ├── a_rooms_template.json
│   ├── e_rooms_template.json
│   └── room_templates.py
├── tests
│   └── test_pdf_processing.py
├── utils
│   ├── __init__.py
│   ├── api_utils.py
│   ├── constants.py
│   ├── drawing_processor.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   ├── pdf_processor.py
│   └── pdf_utils.py
├── .cursorrules
├── .env.example
├── main.py
├── README.md
└── requirements.txt

</file_map>

<file_contents>
File: config/settings.py
```py
import os
from dotenv import load_dotenv

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

# Template Configuration
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')

```

File: processing/batch_processor.py
```py
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

```

File: processing/file_processor.py
```py
import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

from utils.pdf_processor import extract_text_and_tables_from_pdf
from utils.drawing_processor import process_drawing
from templates.room_templates import process_architectural_drawing

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

async def process_pdf_async(
    pdf_path,
    client,
    output_folder,
    drawing_type,
    templates_created
):
    """
    Process a single PDF asynchronously:
    1) Extract text/tables with PyMuPDF
    2) Use GPT to parse/structure the content
    3) Save JSON output
    """
    file_name = os.path.basename(pdf_path)
    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)  # Start
            raw_content = await extract_text_and_tables_from_pdf(pdf_path)
            pbar.update(20)  # PDF text/tables extracted

            # Standard GPT processing for any drawing
            structured_json = await process_drawing(raw_content, drawing_type, client)
            pbar.update(40)  # GPT processing done
            
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            # Attempt to parse JSON response from GPT
            try:
                parsed_json = json.loads(structured_json)
                output_filename = os.path.splitext(file_name)[0] + '_structured.json'
                output_path = os.path.join(type_folder, output_filename)
                
                with open(output_path, 'w') as f:
                    json.dump(parsed_json, f, indent=2)
                
                pbar.update(20)  # JSON saved
                logging.info(f"Successfully processed and saved: {output_path}")
                
                # If Architectural, generate room templates
                if drawing_type == 'Architectural':
                    result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                    templates_created['floor_plan'] = True
                    logging.info(f"Created room templates: {result}")
                
                pbar.update(10)  # Finishing
                return {"success": True, "file": output_path, "panel_schedule": False}
            
            except json.JSONDecodeError as e:
                pbar.update(100)
                logging.error(f"JSON parsing error for {pdf_path}: {str(e)}")
                logging.info(f"Raw API response: {structured_json}")
                
                raw_output_filename = os.path.splitext(file_name)[0] + '_raw_response.json'
                raw_output_path = os.path.join(type_folder, raw_output_filename)
                
                with open(raw_output_path, 'w') as f:
                    f.write(structured_json)
                
                logging.warning(f"Saved raw API response to {raw_output_path}")
                return {"success": False, "error": "Failed to parse JSON", "file": pdf_path}
        
        except Exception as e:
            pbar.update(100)
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}

```

File: processing/job_processor.py
```py
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

```

File: processing/__init__.py
```py
# Processing package initialization

```

File: templates/a_rooms_template.json
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

File: templates/e_rooms_template.json
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

File: templates/room_templates.py
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

File: templates/__init__.py
```py
# Processing package initialization

```

File: tests/test_pdf_processing.py
```py
import unittest
from utils.pdf_processor import extract_text_and_tables_from_pdf
from utils.pdf_utils import structure_panel_data

class TestPDFProcessing(unittest.IsolatedAsyncioTestCase):
    async def test_panel_schedule_extraction(self):
        test_file = "samples/panel_schedule.pdf"
        content = await extract_text_and_tables_from_pdf(test_file)
        
        self.assertIn("Main Panel", content)
        self.assertRegex(content, r"Circuit\s+\d+")
        
        structured = await structure_panel_data(client, content)
        self.assertIn("circuits", structured)
        self.assertTrue(len(structured["circuits"]) > 5) 
```

File: utils/api_utils.py
```py
import asyncio
import logging
import random

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

async def async_safe_api_call(client, *args, **kwargs):
    """
    Safely call the OpenAI API with retries and backoff.
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

```

File: utils/drawing_processor.py
```py
from openai import AsyncOpenAI

DRAWING_INSTRUCTIONS = {
    "Electrical": "Focus on panel schedules, circuit info, equipment schedules with electrical characteristics, and installation notes.",
    "Mechanical": "Capture equipment schedules, HVAC details (CFM, capacities), and installation instructions.",
    "Plumbing": "Include fixture schedules, pump details, water heater specs, pipe sizing, and system instructions.",
    "Architectural": """
    Extract and structure the following information:
    1. Room details: Create a 'rooms' array with objects for each room, including:
       - 'number': Room number (as a string)
       - 'name': Room name
       - 'finish': Ceiling finish
       - 'height': Ceiling height
    2. Room finish schedules
    3. Door/window details
    4. Wall types
    5. Architectural notes
    Ensure all rooms are captured and properly structured in the JSON output.
    """,
    "General": "Organize all relevant data into logical categories based on content type."
}

async def process_drawing(raw_content: str, drawing_type: str, client: AsyncOpenAI):
    """
    Use GPT to parse PDF text + table data into structured JSON
    based on the drawing type.
    """
    system_message = f"""
    Parse this {drawing_type} drawing/schedule into a structured JSON format. Guidelines:
    1. For text: Extract key information, categorize elements.
    2. For tables: Preserve structure, use nested arrays/objects.
    3. Create a hierarchical structure, use consistent key names.
    4. Include metadata (drawing number, scale, date) if available.
    5. {DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"])}
    6. For all drawing types, if room information is present, always include a 'rooms' array in the JSON output, 
       with each room having at least 'number' and 'name' fields.
    Ensure the entire response is a valid JSON object.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_content}
            ],
            temperature=0.2,
            max_tokens=16000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing {drawing_type} drawing: {str(e)}")
        raise

```

File: utils/file_utils.py
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

File: utils/logging_utils.py
```py
import os
import logging
from datetime import datetime

def setup_logging(output_folder: str) -> None:
    """
    Configure and initialize logging for the application.
    Creates a 'logs' folder in the output directory.
    """
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"process_log_{timestamp}.txt")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print(f"Logging to: {log_file}")

```

File: utils/pdf_processor.py
```py
import pymupdf
import json
import os
import logging
from openai import AsyncOpenAI

# Configure logger
logger = logging.getLogger(__name__)

async def extract_text_and_tables_from_pdf(pdf_path: str) -> str:
    """
    Extract text and tables from a PDF file using PyMuPDF.
    """
    logging.info(f"Extracting text and tables from {pdf_path}")
    doc = pymupdf.open(pdf_path)
    all_content = ""
    
    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        all_content += f"PAGE {page_num+1} TEXT:\n{text}\n\n"
        
        # Extract tables
        tables = page.find_tables()
        for table_num, table in enumerate(tables):
            all_content += f"PAGE {page_num+1} TABLE {table_num+1}:\n"
            markdown = table.to_markdown()
            all_content += markdown + "\n\n"
    
    # Close the document
    doc.close()
    return all_content

async def structure_panel_data(client: AsyncOpenAI, raw_content: str) -> dict:
    """
    Use OpenAI to structure electrical panel data from raw content.
    """
    prompt = f"""
    You are an expert in electrical engineering and panel schedules. 
    Please structure the following content from an electrical panel schedule into a valid JSON format. 
    The content includes both text and tables. Extract key information such as panel name, voltage, amperage, circuits, 
    and any other relevant details.
    Pay special attention to the tabular data, which represents circuit information.
    Ensure your entire response is a valid JSON object.
    Raw content:
    {raw_content}
    """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that structures electrical panel data into JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def process_pdf(pdf_path: str, output_folder: str, client: AsyncOpenAI):
    """
    Process a PDF file: extract content, structure it, and save to JSON.
    """
    logging.info(f"Processing PDF: {pdf_path}")
    os.makedirs(output_folder, exist_ok=True)
    
    raw_content = await extract_text_and_tables_from_pdf(pdf_path)
    
    structured_data = await structure_panel_data(client, raw_content)
    
    panel_name = structured_data.get('panel_name', 'unknown_panel').replace(" ", "_").lower()
    filename = f"{panel_name}_electric_panel.json"
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    logging.info(f"Saved structured panel data: {filepath}")
    return raw_content, structured_data

```

File: utils/__init__.py
```py
# Processing package initialization

```

File: .env.example
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
```

File: main.py
```py
import os
import sys
import asyncio
import logging

from openai import AsyncOpenAI
from config.settings import OPENAI_API_KEY
from utils.logging_utils import setup_logging
from processing.job_processor import process_job_site_async

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_folder> [output_folder]")
        sys.exit(1)
    
    job_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(job_folder, "output")
    
    if not os.path.exists(job_folder):
        print(f"Error: Input folder '{job_folder}' does not exist.")
        sys.exit(1)
    
    # 1) Set up logging
    setup_logging(output_folder)
    logging.info(f"Processing files from: {job_folder}")
    logging.info(f"Output will be saved to: {output_folder}")
    
    # 2) Create OpenAI Client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # 3) Run asynchronous job processing
    asyncio.run(process_job_site_async(job_folder, output_folder, client)) 
```

File: README.md
```md
# Ohmni Oracle

PDF processing pipeline that extracts and structures drawing information using PyMuPDF and GPT-4.

## Features

- Processes multiple drawing types (Architectural, Electrical, etc.)
- Extracts text and tables from PDFs using PyMuPDF
- Structures data using GPT-4
- Handles batch processing with rate limiting
- Generates room templates for architectural drawings
- Comprehensive logging and error handling

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your OpenAI API key

## Usage

```bash
python main.py <input_folder> [output_folder]
```

## Configuration

Environment variables in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `BATCH_SIZE`: PDFs to process in parallel (default: 10)
- `API_RATE_LIMIT`: Max API calls per time window (default: 60)
- `TIME_WINDOW`: Time window in seconds (default: 60)

## Output

Processed files are saved as JSON in the output folder, organized by drawing type. 
```

File: utils/constants.py
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
    'Kitchen': ['K', 'KD']
}

def get_drawing_type(filename: str) -> str:
    """
    Detect the drawing type by examining the first 1-2 letters of the filename.
    """
    prefix = os.path.basename(filename).split('.')[0][:2].upper()
    for dtype, prefixes in DRAWING_TYPES.items():
        if any(prefix.startswith(p.upper()) for p in prefixes):
            return dtype
    return 'General'

```

File: requirements.txt
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
openai~=1.59.8  # For Responses API support
tiktoken~=0.6.0

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
tenacity~=8.2.3  # For enhanced retry patterns
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

File: utils/pdf_utils.py
```py
import pymupdf as fitz
import logging
import aiofiles
from typing import List, Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)

async def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For any other errors during extraction
    """
    logger.info(f"Starting text extraction for {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with fitz.open(file_path) as doc:
            logger.info(f"Successfully opened {file_path}")
            text = ""
            for i, page in enumerate(doc):
                logger.info(f"Processing page {i+1} of {len(doc)}")
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"No text extracted from page {i+1}")
        
        if not text:
            logger.warning(f"No text extracted from {file_path}")
        else:
            logger.info(f"Successfully extracted text from {file_path}")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise

async def extract_images(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract images from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing image information
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For any other errors during extraction
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        images = []
        with fitz.open(file_path) as doc:
            for page_index, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # Get transformation matrix and image rectangle
                    # This gives us more precise positioning information
                    image_rect = None
                    for img_info in page.get_image_info():
                        if img_info["xref"] == xref:
                            image_rect = img_info["bbox"]
                            break
                    
                    bbox = image_rect if image_rect else (0, 0, base_image["width"], base_image["height"])
                    
                    images.append({
                        'page': page_index + 1,
                        'index': img_index,
                        'bbox': bbox,
                        'width': base_image["width"],
                        'height': base_image["height"],
                        'type': base_image["ext"],  # Image extension/type
                        'colorspace': base_image.get("colorspace", ""),
                        'xres': base_image.get("xres", 0),
                        'yres': base_image.get("yres", 0)
                    })
        
        logger.info(f"Extracted {len(images)} images from {file_path}")
        return images
    except Exception as e:
        logger.error(f"Error extracting images from {file_path}: {str(e)}")
        raise

async def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For any other errors during extraction
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with fitz.open(file_path) as doc:
            # Enhanced metadata extraction with more fields
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creationDate": doc.metadata.get("creationDate", ""),
                "modDate": doc.metadata.get("modDate", ""),
                "format": "PDF " + doc.metadata.get("format", ""),
                "pageCount": len(doc),
                "encrypted": doc.is_encrypted,
                "fileSize": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
        logger.info(f"Successfully extracted metadata from {file_path}")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
        raise

async def save_page_as_image(file_path: str, page_num: int, output_path: str, dpi: int = 300) -> str:
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
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with fitz.open(file_path) as doc:
            if page_num < 0 or page_num >= len(doc):
                raise IndexError(f"Page number {page_num} out of range (0-{len(doc)-1})")
                
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            pixmap.save(output_path)
            
            logger.info(f"Saved page {page_num} as image: {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"Error saving page as image: {str(e)}")
        raise

```

File: .cursorrules
```cursorrules
Below is a **step-by-step** guide showing **how to remove the Azure Document Intelligence parts** from your original code, along with the **complete** resulting directory/files. The end result is a template that uses only **PyMuPDF** (and GPT/OpenAI) for PDF processing—**no Azure Document Intelligence**.

---

## 1. Remove Azure Document Intelligence references from the codebase

Your original project references Azure in a few places:

1. **`requirements.txt`**:
   - It includes `azure-ai-documentintelligence==1.0.0` and `azure-core>=1.32.0`.
2. **Environment variables** in `.env.example` (for Azure Document Intelligence).
3. **Imports and usage** in:
   - `panel_schedule_intelligence.py` (direct usage of Azure Document Intelligence)
   - `test_azure_panel.py` (testing that Azure-based flow)
   - `file_processor.py` (the fallback to Azure for panel schedules)
4. **Any references** to `DOCUMENTINTELLIGENCE_ENDPOINT` or `DOCUMENTINTELLIGENCE_API_KEY`.

**We’ll remove them** (or comment them out) so your code only relies on PyMuPDF extraction and GPT for everything.

---

## 2. Keep everything else (OpenAI/GPT usage, PyMuPDF, etc.)

Your original code:
- Uses PyMuPDF (via `pymupdf`) to read PDFs, extract text/tables.
- Uses GPT (via `openai` / `AsyncOpenAI`) to structure the data.
- Has various utility files, logging, templates, etc.

We leave **all** that intact.

---

## 3. Adjust `requirements.txt` (Optional)

Since you said these are your current requirements, you may keep them exactly as is. However, if you truly want to **remove** references to Azure, simply **delete** (or comment out) the following lines:

```
azure-ai-documentintelligence==1.0.0
azure-core>=1.32.0
```

That way, you won’t install the Azure libraries at all. Everything else can remain as-is.

---

## 4. Remove or replace code that references Azure Document Intelligence

- **`panel_schedule_intelligence.py`**: Either empty out the file or remove it entirely.  
- **`test_azure_panel.py`**: Remove Azure usage (the `PanelScheduleProcessor` calls). Possibly keep it as a test for PyMuPDF extraction.  
- **`file_processor.py`**: Remove the logic that checks for `panel_processor` and calls Azure.  
- Any mention of `DOCUMENTINTELLIGENCE_ENDPOINT` or `DOCUMENTINTELLIGENCE_API_KEY` can be removed or commented out.

Below is a **fully updated** directory structure and **all files** with the necessary adjustments. **Take your time** to review and copy what you need.

---

# Final Directory Structure (Minus Document Intelligence)

```
btcelectrician-ohmni-oracle-refined/
├── README.md
├── main.py
├── requirements.txt
├── snapshot.md
├── test_azure_panel.py
├── .cursorrules
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py
├── processing/
│   ├── __init__.py
│   ├── batch_processor.py
│   ├── file_processor.py
│   ├── job_processor.py
│   ├── panel_schedule_intelligence.py
│   └── panel_schedule_intelligence_backup.py
├── templates/
│   ├── __init__.py
│   ├── a_rooms_template.json
│   ├── e_rooms_template.json
│   └── room_templates.py
└── utils/
    ├── __init__.py
    ├── api_utils.py
    ├── constants.py
    ├── drawing_processor.py
    ├── drawing_utils.py
    ├── file_utils.py
    ├── logging_utils.py
    ├── pdf_processor.py
    └── pdf_utils.py
```

Below are the **contents of every file** after removing the Azure Document Intelligence parts.

---

## File: `README.md`

```markdown
# Ohmni Oracle

This project processes various types of drawings (e.g., architectural, electrical, mechanical) by:
1. Extracting text from PDF files (using PyMuPDF / pdfplumber)
2. Converting it into structured JSON via GPT-4

## Installation


3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env`

5. **Run**:
   ```bash
   python main.py <input_folder> [output_folder]
   ```

## Project Structure

```
btcelectrician-ohmni_oracle/
├── config/
│   ├── settings.py
│   └── .gitignore
├── processing/
│   ├── batch_processor.py
│   ├── file_processor.py
│   └── job_processor.py
├── templates/
│   ├── a_rooms_template.json
│   ├── e_rooms_template.json
│   └── room_templates.py
├── utils/
│   ├── api_utils.py
│   ├── constants.py
│   ├── drawing_processor.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   ├── pdf_processor.py
│   └── pdf_utils.py
├── .env
├── main.py
├── README.md
└── requirements.txt
```

## Features

- Processes multiple types of drawings (Architectural, Electrical, etc.)
- Extracts text and tables from PDFs
- Converts unstructured data to structured JSON
- Handles batch processing with rate limiting
- Generates room templates for architectural drawings
- Comprehensive logging and error handling

## Configuration

The following environment variables can be configured in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `BATCH_SIZE`: Number of PDFs to process in parallel (default: 10)
- `API_RATE_LIMIT`: Maximum API calls per time window (default: 60)
- `TIME_WINDOW`: Time window in seconds for rate limiting (default: 60)

## License

[Your chosen license]
```

---

## File: `main.py`
```python
import os
import sys
import asyncio
import logging

from openai import AsyncOpenAI
from config.settings import OPENAI_API_KEY
from utils.logging_utils import setup_logging
from processing.job_processor import process_job_site_async

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_folder> [output_folder]")
        sys.exit(1)
    
    job_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(job_folder, "output")
    
    if not os.path.exists(job_folder):
        print(f"Error: Input folder '{job_folder}' does not exist.")
        sys.exit(1)
    
    # 1) Set up logging
    setup_logging(output_folder)
    logging.info(f"Processing files from: {job_folder}")
    logging.info(f"Output will be saved to: {output_folder}")
    
    # 2) Create OpenAI Client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # 3) Run asynchronous job processing
    asyncio.run(process_job_site_async(job_folder, output_folder, client))
```

---

## File: `requirements.txt`
*(Below is the list you provided. To remove Azure, comment out or remove the lines referencing Azure. Otherwise, you can keep them if you still need them for other reasons. Example:)*

```txt
aiohappyeyeballs==2.4.4
aiohttp==3.11
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.8.0
attrs==24.3.0
# azure-ai-documentintelligence==1.0.0
# azure-core>=1.32.0
certifi==2024.12.14
cffi==1.17.1
charset-normalizer==3.4.1
cryptography==44.0.0
distro==1.9.0
frozenlist==1.4.1
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
idna==3.10
jiter==0.8.2
multidict==6.1.0
openai==1.59.8
pillow==10.4.0
pycparser==2.22
pydantic==2.10.5
pydantic_core==2.27.2
PyMuPDF==1.24.11
pypdfium2==4.30.0
python-dotenv==1.0.1
requests==2.32.3
sniffio==1.3.1
tqdm==4.66.5
typing_extensions==4.12.2
urllib3==2.2.3
Wand==0.6.13
yarl==1.17.0
tiktoken==0.6.0
```

---

## File: `snapshot.md`
*(No Azure references here; unchanged.)*
```md
[Your original snapshot.md content remains here]
```

---

## File: `test_azure_panel.py`
Previously, this file tested Azure Document Intelligence. You can remove it entirely or replace its contents to test only the PyMuPDF logic.  
Below is an example that:

- Detects panel schedules by filename (using `is_panel_schedule`).
- Extracts text with **PyMuPDF**.
- Writes the result to JSON.  

```python
import os
import sys
import json
import logging
import asyncio

from dotenv import load_dotenv
from processing.file_processor import is_panel_schedule
from utils.pdf_processor import extract_text_and_tables_from_pdf

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_azure_panel.py <pdf_file_or_folder>")
        sys.exit(1)

    path_arg = sys.argv[1]
    if not os.path.exists(path_arg):
        print(f"Error: Path '{path_arg}' does not exist.")
        sys.exit(1)

    load_dotenv()

    pdf_files = []
    if os.path.isfile(path_arg) and path_arg.lower().endswith(".pdf"):
        pdf_files.append(path_arg)
    elif os.path.isdir(path_arg):
        for root, _, files in os.walk(path_arg):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))

    output_folder = os.path.join(os.getcwd(), "test_output")
    os.makedirs(output_folder, exist_ok=True)

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)

        if is_panel_schedule(file_name, ""):
            logging.info(f"Detected panel schedule in '{file_name}'.")
            try:
                # Extract with PyMuPDF (async call)
                raw_content = await extract_text_and_tables_from_pdf(pdf_path)
                result_data = {
                    "extracted_content": raw_content,
                    "error": None
                }
                logging.info(f"Successfully processed '{file_name}'.")
            except Exception as e:
                logging.exception(f"Error: {e}")
                result_data = {"extracted_content": "", "error": str(e)}

            out_file = os.path.join(
                output_folder,
                f"{os.path.splitext(file_name)[0]}_test_panel.json"
            )
            with open(out_file, "w") as f:
                json.dump(result_data, f, indent=2)
            logging.info(f"Wrote output to '{out_file}'")

        else:
            logging.info(f"'{file_name}' is NOT flagged as a panel schedule.")

if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
```

*(Or remove this file if you no longer need it.)*

---

## File: `.cursorrules`
*(Unchanged.)*
```text
Commit Message Prefixes:
* "fix:" for bug fixes
* "feat:" for new features
* "perf:" for performance improvements
* "docs:" for documentation changes
* "style:" for formatting changes
* "refactor:" for code refactoring
* "test:" for adding missing tests
* "chore:" for maintenance tasks
Rules:
* Use lowercase for commit messages
* Keep the summary line concise
* Include description for non-obvious changes
* Reference issue numbers when applicable
Documentation

* Maintain clear README with setup instructions
* Document API interactions and data flows
* Keep manifest.json well-documented
* Don't include comments unless it's for complex logic
* Document permission requirements
Development Workflow

* Use proper version control
* Implement proper code review process
* Test in multiple environments
* Follow semantic versioning for releases
* Maintain changelog
```

---

## File: `.env.example`
Remove or comment out the Azure lines:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_key_here

# Optional lines for Azure - removed or commented out
# DOCUMENTINTELLIGENCE_ENDPOINT=<yourEndpoint>
# DOCUMENTINTELLIGENCE_API_KEY=<yourKey>

# Optional Configuration
# LOG_LEVEL=INFO
# BATCH_SIZE=10
# API_RATE_LIMIT=60
# TIME_WINDOW=60
```

---

## File: `config/settings.py`
No changes needed for removing Azure. It only checks `OPENAI_API_KEY`. 
```python
import os
from dotenv import load_dotenv

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

# Template Configuration
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
```

---

## File: `processing/__init__.py`
```python
# Processing package initialization
```

---

## File: `processing/batch_processor.py`
*(Unchanged; no Azure references.)*
```python
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
```

---

## File: `processing/file_processor.py`
**Removed** any reference to Azure or `PanelScheduleProcessor`. We keep `is_panel_schedule` as is.

```python
import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

from utils.pdf_processor import extract_text_and_tables_from_pdf
from utils.drawing_processor import process_drawing
from templates.room_templates import process_architectural_drawing

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

async def process_pdf_async(
    pdf_path,
    client,
    output_folder,
    drawing_type,
    templates_created
):
    """
    Process a single PDF asynchronously:
    1) Extract text/tables with PyMuPDF
    2) Use GPT to parse/structure the content
    3) Save JSON output
    """
    file_name = os.path.basename(pdf_path)
    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)  # Start
            raw_content = await extract_text_and_tables_from_pdf(pdf_path)
            pbar.update(20)  # PDF text/tables extracted

            # Standard GPT processing for any drawing
            structured_json = await process_drawing(raw_content, drawing_type, client)
            pbar.update(40)  # GPT processing done
            
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            # Attempt to parse JSON response from GPT
            try:
                parsed_json = json.loads(structured_json)
                output_filename = os.path.splitext(file_name)[0] + '_structured.json'
                output_path = os.path.join(type_folder, output_filename)
                
                with open(output_path, 'w') as f:
                    json.dump(parsed_json, f, indent=2)
                
                pbar.update(20)  # JSON saved
                logging.info(f"Successfully processed and saved: {output_path}")
                
                # If Architectural, generate room templates
                if drawing_type == 'Architectural':
                    result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                    templates_created['floor_plan'] = True
                    logging.info(f"Created room templates: {result}")
                
                pbar.update(10)  # Finishing
                return {"success": True, "file": output_path, "panel_schedule": False}
            
            except json.JSONDecodeError as e:
                pbar.update(100)
                logging.error(f"JSON parsing error for {pdf_path}: {str(e)}")
                logging.info(f"Raw API response: {structured_json}")
                
                raw_output_filename = os.path.splitext(file_name)[0] + '_raw_response.json'
                raw_output_path = os.path.join(type_folder, raw_output_filename)
                
                with open(raw_output_path, 'w') as f:
                    f.write(structured_json)
                
                logging.warning(f"Saved raw API response to {raw_output_path}")
                return {"success": False, "error": "Failed to parse JSON", "file": pdf_path}
        
        except Exception as e:
            pbar.update(100)
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}
```

---

## File: `processing/job_processor.py`
*(Unchanged; no Azure references.)*
```python
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
```

---

## File: `processing/panel_schedule_intelligence.py`
Since we no longer need Azure-based processing, you can **empty** this file or remove it.  
Example placeholder:

```python
"""
Placeholder file after removing Azure Document Intelligence.
No code needed here anymore.
"""
```

---

## File: `processing/panel_schedule_intelligence_backup.py`
*(Likewise, if it’s strictly Azure code, you can remove or keep as a “backup” file.)*
```python
"""
Legacy backup file. Contains references to Azure Document Intelligence from older versions.
"""
```

---

## File: `templates/__init__.py`
```python
# Templates package initialization
```

---

## File: `templates/a_rooms_template.json`
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

---

## File: `templates/e_rooms_template.json`
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

---

## File: `templates/room_templates.py`
```python
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

---

## File: `utils/__init__.py`
```python
# Utilities package initialization
```

---

## File: `utils/api_utils.py`
*(Unchanged; no Azure references.)*
```python
import asyncio
import logging
import random

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

async def async_safe_api_call(client, *args, **kwargs):
    """
    Safely call the OpenAI API with retries and backoff.
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
```

---

## File: `utils/constants.py`
*(Unchanged; no Azure references.)*
```python
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
    'Kitchen': ['K', 'KD']
}

def get_drawing_type(filename: str) -> str:
    """
    Detect the drawing type by examining the first 1-2 letters of the filename.
    """
    prefix = os.path.basename(filename).split('.')[0][:2].upper()
    for dtype, prefixes in DRAWING_TYPES.items():
        if any(prefix.startswith(p.upper()) for p in prefixes):
            return dtype
    return 'General'
```

---

## File: `utils/drawing_processor.py`
*(Unchanged; the GPT logic for structuring data remains. No Azure references.)*
```python
from openai import AsyncOpenAI

DRAWING_INSTRUCTIONS = {
    "Electrical": "Focus on panel schedules, circuit info, equipment schedules with electrical characteristics, and installation notes.",
    "Mechanical": "Capture equipment schedules, HVAC details (CFM, capacities), and installation instructions.",
    "Plumbing": "Include fixture schedules, pump details, water heater specs, pipe sizing, and system instructions.",
    "Architectural": """
    Extract and structure the following information:
    1. Room details: Create a 'rooms' array with objects for each room, including:
       - 'number': Room number (as a string)
       - 'name': Room name
       - 'finish': Ceiling finish
       - 'height': Ceiling height
    2. Room finish schedules
    3. Door/window details
    4. Wall types
    5. Architectural notes
    Ensure all rooms are captured and properly structured in the JSON output.
    """,
    "General": "Organize all relevant data into logical categories based on content type."
}

async def process_drawing(raw_content: str, drawing_type: str, client: AsyncOpenAI):
    """
    Use GPT to parse PDF text + table data into structured JSON
    based on the drawing type.
    """
    system_message = f"""
    Parse this {drawing_type} drawing/schedule into a structured JSON format. Guidelines:
    1. For text: Extract key information, categorize elements.
    2. For tables: Preserve structure, use nested arrays/objects.
    3. Create a hierarchical structure, use consistent key names.
    4. Include metadata (drawing number, scale, date) if available.
    5. {DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"])}
    6. For all drawing types, if room information is present, always include a 'rooms' array in the JSON output, 
       with each room having at least 'number' and 'name' fields.
    Ensure the entire response is a valid JSON object.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_content}
            ],
            temperature=0.2,
            max_tokens=16000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing {drawing_type} drawing: {str(e)}")
        raise
```

---

## File: `utils/drawing_utils.py`
*(Unchanged; no Azure references.)*
```python
"""
Additional drawing-related helper functions.
"""
```

---

## File: `utils/file_utils.py`
*(Unchanged; no Azure references.)*
```python
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

def cleanup_temporary_files(output_folder: str) -> None:
    """
    Clean up any temporary files created during processing (not currently used).
    """
    pass

def get_project_name(job_folder: str) -> str:
    """
    Extract the project name from the job folder path.
    """
    return os.path.basename(job_folder)
```

---

## File: `utils/logging_utils.py`
*(Unchanged; no Azure references.)*
```python
import os
import logging
from datetime import datetime

def setup_logging(output_folder: str) -> None:
    """
    Configure and initialize logging for the application.
    Creates a 'logs' folder in the output directory.
    """
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"process_log_{timestamp}.txt")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print(f"Logging to: {log_file}")
```

---

## File: `utils/pdf_processor.py`
*(Unchanged; uses PyMuPDF for text/table extraction, no Azure references.)*
```python
import pymupdf
import json
import os
from openai import AsyncOpenAI

async def extract_text_and_tables_from_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    all_content = ""
    for page in doc:
        text = page.get_text()
        all_content += "TEXT:\n" + text + "\n"
        
        tables = page.find_tables()
        for table in tables:
            all_content += "TABLE:\n"
            markdown = table.to_markdown()
            all_content += markdown + "\n"
    
    return all_content

async def structure_panel_data(client: AsyncOpenAI, raw_content: str) -> dict:
    prompt = f"""
    You are an expert in electrical engineering and panel schedules. 
    Please structure the following content from an electrical panel schedule into a valid JSON format. 
    The content includes both text and tables. Extract key information such as panel name, voltage, amperage, circuits, 
    and any other relevant details.
    Pay special attention to the tabular data, which represents circuit information.
    Ensure your entire response is a valid JSON object.
    Raw content:
    {raw_content}
    """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that structures electrical panel data into JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def process_pdf(pdf_path: str, output_folder: str, client: AsyncOpenAI):
    print(f"Processing PDF: {pdf_path}")
    raw_content = await extract_text_and_tables_from_pdf(pdf_path)
    
    structured_data = await structure_panel_data(client, raw_content)
    
    panel_name = structured_data.get('panel_name', 'unknown_panel').replace(" ", "_").lower()
    filename = f"{panel_name}_electric_panel.json"
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    print(f"Saved structured panel data: {filepath}")
    return raw_content, structured_data
```

---

# Done!

This completes the **template** you requested: all original functionality **minus Azure Document Intelligence**. You now have a version that uses **only PyMuPDF** (and GPT/OpenAI) to process PDFs. 

**Usage** remains:

```bash
# (Optional) comment out azure lines in requirements.txt
pip install -r requirements.txt
python main.py <input_folder> [output_folder]
```

You still have **panel-schedule detection** logic via `is_panel_schedule(...)`, but there is **no** Azure intelligence fallback. All PDFs (including panel schedules) will go through PyMuPDF + GPT.

> **Tip:** If you do not need to install Azure packages, remove those lines in `requirements.txt` before installation.

That’s it! You now have a **complete** codebase and steps for removing the Document Intelligence references while retaining everything else.
```
</file_contents>

