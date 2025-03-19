import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from typing import Dict, Any

from services.extraction_service import create_extractor
from services.ai_service import DrawingAiService, process_drawing
from services.storage_service import FileSystemStorage
from utils.performance_utils import time_operation
from utils.constants import get_drawing_type

# Load environment variables
load_dotenv()

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
    2) Process with AI using universal prompt
    3) Save structured JSON output
    """
    file_name = os.path.basename(pdf_path)
    logger = logging.getLogger(__name__)

    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)
            extractor = create_extractor(drawing_type, logger)
            storage = FileSystemStorage(logger)
            ai_service = DrawingAiService(client, logger)

            extraction_result = await extractor.extract(pdf_path)
            if not extraction_result.success:
                pbar.update(100)
                logger.error(f"Extraction failed for {pdf_path}: {extraction_result.error}")
                return {"success": False, "error": extraction_result.error, "file": pdf_path}

            # Concatenate all extracted content without any truncation
            raw_content = extraction_result.raw_text
            for table in extraction_result.tables:
                raw_content += f"\nTABLE:\n{table['content']}\n"
                
            # Log content length for specification files
            is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"
            if is_specification:
                logger.info(f"Processing specification document: {file_name} with {len(raw_content)} characters")
            else:
                logger.info(f"Processing {drawing_type} drawing: {file_name} with {len(raw_content)} characters")
                
            pbar.update(20)

            # Send the complete raw content to the AI service using process_drawing
            structured_json = await process_drawing(raw_content, drawing_type, client, file_name)
            pbar.update(40)

            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            try:
                parsed_json = json.loads(structured_json)
            except json.JSONDecodeError as e:
                pbar.update(100)
                logger.error(f"JSON error for {pdf_path}: {str(e)}")
                logger.error(f"Raw API response: {structured_json[:500]}...")  # Log the first 500 chars
                raw_output_path = os.path.join(type_folder, f"{os.path.splitext(file_name)[0]}_raw_response.json")
                await storage.save_text(structured_json, raw_output_path)
                return {"success": False, "error": f"JSON parse failed: {str(e)}", "file": pdf_path}

            output_path = os.path.join(type_folder, f"{os.path.splitext(file_name)[0]}_structured.json")
            await storage.save_json(parsed_json, output_path)
            pbar.update(20)
            logger.info(f"Saved: {output_path}")

            if drawing_type == 'Architectural' and 'rooms' in parsed_json:
                from templates.room_templates import process_architectural_drawing
                result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                templates_created['floor_plan'] = True
                logger.info(f"Created templates: {result}")

            pbar.update(10)
            return {"success": True, "file": output_path}
        except Exception as e:
            pbar.update(100)
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}