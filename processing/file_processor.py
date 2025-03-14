"""
File processor module that handles individual file processing.
"""
import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from services.extraction_service import create_extractor
from services.ai_service import DrawingAiService, ModelType
from services.storage_service import FileSystemStorage
from utils.drawing_processor import DRAWING_INSTRUCTIONS, process_drawing
from utils.performance_utils import time_operation
from templates.room_templates import process_architectural_drawing

# Load environment variables
load_dotenv()


def is_panel_schedule(file_name: str, raw_content: str) -> bool:
    """
    Determine if a PDF is likely an electrical panel schedule
    based solely on the file name (no numeric or content checks).
    
    Args:
        file_name: Name of the PDF file
        raw_content: (Unused) Extracted text content from the PDF
        
    Returns:
        True if the file name contains panel-schedule keywords
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
    1) Extract text/tables with appropriate extractor based on drawing type
    2) Use GPT to parse/structure the content
    3) Save JSON output
    
    Args:
        pdf_path: Path to the PDF file
        client: OpenAI client
        output_folder: Output folder for processed files
        drawing_type: Type of drawing
        templates_created: Dictionary tracking created templates
        
    Returns:
        Processing result dictionary
    """
    file_name = os.path.basename(pdf_path)
    logger = logging.getLogger(__name__)
    
    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)  # Start
            
            # Initialize services - use specialized extractor for drawing type
            extractor = create_extractor(drawing_type, logger)
            storage = FileSystemStorage(logger)
            ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS, logger)
            
            # Extract text and tables
            extraction_result = await extractor.extract(pdf_path)
            if not extraction_result.success:
                pbar.update(100)
                logger.error(f"Extraction failed for {pdf_path}: {extraction_result.error}")
                return {"success": False, "error": extraction_result.error, "file": pdf_path}
            
            # Convert to raw_content format expected by process_drawing
            raw_content = ""
            raw_content += extraction_result.raw_text
            for table in extraction_result.tables:
                raw_content += f"\nTABLE:\n{table['content']}\n"
            
            pbar.update(20)  # PDF text/tables extracted
            
            # Process with AI - pass the file_name to help with parameter optimization
            structured_json = await process_drawing(
                raw_content=raw_content,
                drawing_type=drawing_type,
                client=client,
                file_name=file_name
            )
            
            pbar.update(40)  # GPT processing done
            
            # Create output directory
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)
            
            # Handle the structured JSON response
            try:
                # Parse the JSON to validate it
                parsed_json = json.loads(structured_json)
                
                output_filename = os.path.splitext(file_name)[0] + '_structured.json'
                output_path = os.path.join(type_folder, output_filename)
                
                # Save the structured JSON
                await storage.save_json(parsed_json, output_path)
                
                pbar.update(20)  # JSON saved
                logger.info(f"Successfully processed and saved: {output_path}")
                
                # If Architectural, generate room templates
                if drawing_type == 'Architectural':
                    result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                    templates_created['floor_plan'] = True
                    logger.info(f"Created room templates: {result}")
                
                pbar.update(10)  # Finishing
                return {"success": True, "file": output_path, "panel_schedule": False}
                
            except json.JSONDecodeError as e:
                pbar.update(100)
                logger.error(f"JSON parsing error for {pdf_path}: {str(e)}")
                logger.error(f"Raw response: {structured_json[:200]}...")
                
                # Save the raw response for debugging
                raw_output_filename = os.path.splitext(file_name)[0] + '_raw_response.json'
                raw_output_path = os.path.join(type_folder, raw_output_filename)
                
                await storage.save_text(structured_json, raw_output_path)
                
                logger.warning(f"Saved raw API response to {raw_output_path}")
                return {"success": False, "error": f"Failed to parse JSON: {str(e)}", "file": pdf_path}
        
        except Exception as e:
            pbar.update(100)
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}
