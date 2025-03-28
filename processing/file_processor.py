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
                if (drawing_type == 'Architectural' and
                        isinstance(parsed_json, dict) and # Check if parsed_json is a dictionary
                        'ARCHITECTURAL' in parsed_json and # Check if 'ARCHITECTURAL' key exists
                        isinstance(parsed_json.get('ARCHITECTURAL'), dict) and # Check if its value is a dictionary
                        'ROOMS' in parsed_json['ARCHITECTURAL'] and # Check if 'ROOMS' key exists within 'ARCHITECTURAL'
                        isinstance(parsed_json['ARCHITECTURAL'].get('ROOMS'), list)): # Check if its value is a list

                    # Log entry added for confirmation
                    logger.info(f"Found architectural rooms in {file_name}. Calling process_architectural_drawing.")
                    try:
                        from templates.room_templates import process_architectural_drawing
                        # Pass parsed_json (the dictionary) not the string
                        result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                        # Ensure templates_created is defined and accessible in this scope if needed elsewhere
                        templates_created['floor_plan'] = True 
                        logger.info(f"Finished process_architectural_drawing for {file_name}. Result: {result}")
                    except Exception as template_error:
                        logger.error(f"Error during process_architectural_drawing for {file_name}: {template_error}")
                else:
                    # Log if the condition is false for debugging
                    if drawing_type == 'Architectural':
                        logger.warning(f"Architectural file {file_name} processed, but 'ARCHITECTURAL.ROOMS' list not found or invalid in the initial JSON. Skipping template generation.")

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