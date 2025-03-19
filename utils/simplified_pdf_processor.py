"""
Simplified PDF processor that fixes data loss issues while maintaining performance.
"""
import os
import json
import logging
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from tqdm.asyncio import tqdm

import pymupdf as fitz
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
MAX_RETRIES = 3
API_TIMEOUT = 60  # seconds
API_RATE_LIMIT = 60  # requests per minute
API_TIME_WINDOW = 60  # seconds

# Drawing instructions for different drawing types
DRAWING_INSTRUCTIONS = {
    "Architectural": """
    Extract and structure the following information:
    1. Room details: Create a 'rooms' array with objects for each room, including:
       - 'number': Room number (as a string)
       - 'name': Room name
       - 'finish': Ceiling finish
       - 'height': Ceiling height
       - 'electrical_info': Any electrical specifications 
       - 'architectural_info': Any additional architectural details
    2. Room finish schedules
    3. Door/window details
    4. Wall types
    5. Architectural notes
    Ensure all rooms are captured and properly structured in the JSON output.
    """,
    "Electrical": "Focus on panel schedules, circuit info, equipment schedules with electrical characteristics, and installation notes.",
    "Mechanical": "Capture equipment schedules, HVAC details (CFM, capacities), and installation instructions.",
    "Plumbing": "Include fixture schedules, pump details, water heater specs, pipe sizing, and system instructions.",
    "Specifications": """
    Pay special attention to:
    1. Section numbers and titles
    2. Hierarchical structure (sections, subsections)
    3. Technical requirements
    4. Material specifications
    5. Reference standards
    6. Installation instructions
    Preserve ALL text content without summarization. Maintain document structure.
    """,
    "General": "Organize all relevant data into logical categories based on content type."
}

# Drawing type mapping
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
    Determine the drawing type based on the filename prefix.
    
    Args:
        filename: Path to the PDF file
        
    Returns:
        Drawing type name
    """
    prefix = os.path.basename(filename).split('.')[0][:2].upper()
    
    for dtype, prefixes in DRAWING_TYPES.items():
        if any(prefix.startswith(p.upper()) for p in prefixes):
            return dtype
            
    # Check for specifications
    if "SPEC" in os.path.basename(filename).upper():
        return "Specifications"
        
    return 'General'


async def extract_pdf_content(pdf_path: str) -> str:
    """
    Extract text and tables from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted content as a formatted string
    """
    try:
        logger.info(f"Extracting content from: {pdf_path}")
        
        # Move CPU-intensive work to a separate thread
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _extract_pdf_content_sync, pdf_path)
        
        logger.info(f"Successfully extracted content from {pdf_path}")
        return content
    except Exception as e:
        logger.error(f"Error extracting content from {pdf_path}: {str(e)}")
        raise


def _extract_pdf_content_sync(pdf_path: str) -> str:
    """
    Synchronous helper function to extract PDF content.
    This runs in a separate thread.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted content as a formatted string
    """
    all_content = ""
    
    with fitz.open(pdf_path) as doc:
        # First, add some metadata
        all_content += "METADATA:\n"
        all_content += f"Title: {doc.metadata.get('title', 'Unknown')}\n"
        all_content += f"Author: {doc.metadata.get('author', 'Unknown')}\n"
        all_content += f"Subject: {doc.metadata.get('subject', 'Unknown')}\n"
        all_content += f"Creator: {doc.metadata.get('creator', 'Unknown')}\n"
        all_content += f"Page Count: {len(doc)}\n\n"
        
        # Extract content from each page
        for i, page in enumerate(doc):
            # Add page header
            all_content += f"PAGE {i+1}:\n"
            
            # Extract text
            text = page.get_text()
            all_content += "TEXT:\n" + text + "\n\n"
            
            # Extract tables
            tables = page.find_tables()
            if tables:
                for j, table in enumerate(tables):
                    all_content += f"TABLE {j+1}:\n"
                    markdown = table.to_markdown()
                    all_content += markdown + "\n\n"
    
    return all_content


class RetryableOpenAIError(Exception):
    """Base class for retryable OpenAI errors."""
    pass


class RateLimitError(RetryableOpenAIError):
    """Exception for rate limit errors."""
    pass


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type(RetryableOpenAIError)
)
async def process_with_ai(client: AsyncOpenAI, content: str, drawing_type: str) -> Dict[str, Any]:
    """
    Process content with OpenAI API with retries for rate limits.
    
    Args:
        client: AsyncOpenAI client
        content: Extracted PDF content
        drawing_type: Type of drawing
        
    Returns:
        Structured JSON data
    """
    instruction = DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"])
    
    system_message = f"""
    You are an expert in construction drawings and specifications. Parse this {drawing_type} content into structured JSON.
    
    Instructions:
    1. Extract all relevant information from text and tables
    2. Create a hierarchical JSON structure with consistent key names
    3. Include metadata (drawing number, scale, date, etc.) if available
    4. {instruction}
    
    For all drawing types, if room information is present, always include a 'rooms' array with at least 'number' and 'name' fields for each room.
    
    If this is a specification document:
    - Preserve ALL textual content and organization 
    - Maintain section numbers and hierarchical structure
    - Capture technical requirements completely
    
    Ensure your response is a COMPLETE and VALID JSON object.
    """
    
    # For specifications, ensure more complete extraction
    if drawing_type == "Specifications":
        # Don't force JSON response format for specifications to avoid truncation
        response_format = None
        max_tokens = 32000  # Use more tokens for specifications
    else:
        response_format = {"type": "json_object"}
        max_tokens = 16000
    
    try:
        logger.info(f"Processing {drawing_type} with GPT")
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": content}
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=API_TIMEOUT
        )
        
        # Get the response content
        ai_content = response.choices[0].message.content
        
        # For specifications with no forced JSON format, parse JSON from the response
        if drawing_type == "Specifications" and not response_format:
            # Try to extract JSON from the response, ignoring any explanatory text
            try:
                # Find JSON content by looking for opening brace
                json_start = ai_content.find('{')
                json_end = ai_content.rfind('}')
                
                if json_start != -1 and json_end != -1:
                    json_content = ai_content[json_start:json_end+1]
                    structured_data = json.loads(json_content)
                else:
                    # Try parsing the whole content as JSON
                    structured_data = json.loads(ai_content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from response, returning raw content")
                # Create a structured format for the raw content
                structured_data = {
                    "metadata": {
                        "title": "Specification Document",
                        "processed": "Text format - could not parse as JSON"
                    },
                    "content": ai_content
                }
        else:
            # For non-specifications or when using response_format
            structured_data = json.loads(ai_content)
            
        logger.info(f"Successfully processed {drawing_type} with GPT")
        return structured_data
        
    except Exception as e:
        error_msg = str(e).lower()
        if "rate limit" in error_msg:
            logger.warning(f"Rate limit exceeded, retrying: {error_msg}")
            raise RateLimitError(f"Rate limit: {error_msg}")
        else:
            logger.error(f"Error processing with AI: {error_msg}")
            raise


async def process_pdf(
    pdf_path: str, 
    client: AsyncOpenAI, 
    output_folder: str, 
    drawing_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a single PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        client: AsyncOpenAI client
        output_folder: Output folder for results
        drawing_type: Optional drawing type (auto-detected if not provided)
        
    Returns:
        Processing result
    """
    file_name = os.path.basename(pdf_path)
    
    # Auto-detect drawing type if not provided
    if not drawing_type:
        drawing_type = get_drawing_type(pdf_path)
    
    logger.info(f"Processing {file_name} as {drawing_type}")
    
    try:
        with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
            # Step 1: Extract content
            content = await extract_pdf_content(pdf_path)
            pbar.update(30)
            
            # Step 2: Process with AI
            structured_data = await process_with_ai(client, content, drawing_type)
            pbar.update(50)
            
            # Step 3: Save result
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)
            
            output_filename = os.path.splitext(file_name)[0] + '_structured.json'
            output_path = os.path.join(type_folder, output_filename)
            
            with open(output_path, 'w') as f:
                json.dump(structured_data, indent=2, fp=f)
            
            pbar.update(20)
            logger.info(f"Saved result to {output_path}")
            
            # Step 4: Handle architectural drawings specially
            if drawing_type == "Architectural" and "rooms" in structured_data:
                # Import here to avoid circular imports
                from templates.room_templates import process_architectural_drawing
                
                result = process_architectural_drawing(structured_data, pdf_path, type_folder)
                logger.info(f"Created room templates: {result}")
            
            return {
                "success": True,
                "file": output_path,
                "drawing_type": drawing_type,
                "structured_data": structured_data  # Return the structured data for further processing
            }
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error for {pdf_path}: {str(e)}")
        return {
            "success": False, 
            "error": f"Failed to parse JSON: {str(e)}", 
            "file": pdf_path
        }
    
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "file": pdf_path
        }


async def process_batch(
    batch: List[str],
    client: AsyncOpenAI,
    output_folder: str
) -> List[Dict[str, Any]]:
    """
    Process a batch of PDF files with rate limiting.
    
    Args:
        batch: List of PDF file paths
        client: AsyncOpenAI client
        output_folder: Output folder for results
        
    Returns:
        List of processing results
    """
    results = []
    start_time = time.time()
    
    for i, pdf_path in enumerate(batch):
        # Apply rate limiting
        if i > 0 and i % API_RATE_LIMIT == 0:
            elapsed = time.time() - start_time
            if elapsed < API_TIME_WINDOW:
                wait_time = API_TIME_WINDOW - elapsed
                logger.info(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            start_time = time.time()
        
        # Process the PDF
        result = await process_pdf(pdf_path, client, output_folder)
        results.append(result)
    
    return results


async def process_job_folder(job_folder: str, output_folder: str, client: AsyncOpenAI, batch_size: int = 10) -> Dict[str, Any]:
    """
    Process all PDF files in a job folder.
    
    Args:
        job_folder: Input folder containing PDF files
        output_folder: Output folder for results
        client: AsyncOpenAI client
        batch_size: Number of files to process in each batch
        
    Returns:
        Summary of processing results
    """
    start_time = time.time()
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(job_folder):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    
    if not pdf_files:
        logger.warning("No PDF files found. Please check the input folder.")
        return {
            "success": False,
            "error": "No PDF files found",
            "elapsed_time": 0
        }
    
    # Process in batches
    all_results = []
    total_batches = (len(pdf_files) + batch_size - 1) // batch_size
    
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {total_batches}")
            
            batch_results = await process_batch(batch, client, output_folder)
            all_results.extend(batch_results)
            
            overall_pbar.update(len(batch))
    
    # Calculate processing time
    elapsed_time = time.time() - start_time
    
    # Summarize results
    successes = [r for r in all_results if r.get('success', False)]
    failures = [r for r in all_results if not r.get('success', False)]
    
    logger.info(f"Processing complete. Time elapsed: {elapsed_time:.2f}s")
    logger.info(f"Successes: {len(successes)}, Failures: {len(failures)}")
    
    if failures:
        logger.warning("Failed files:")
        for failure in failures:
            logger.warning(f"  {failure['file']}: {failure.get('error', 'Unknown error')}")
    
    return {
        "success": True,
        "total_files": len(pdf_files),
        "successes": len(successes),
        "failures": len(failures),
        "elapsed_time": elapsed_time,
        "results": all_results
    } 