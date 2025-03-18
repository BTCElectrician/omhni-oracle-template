import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from typing import Dict, Any

from services.extraction_service import create_extractor
from services.ai_service import DrawingAiService, ModelType
from services.storage_service import FileSystemStorage
from utils.performance_utils import time_operation
from backup.prompts import PROMPTS
from utils.constants import get_drawing_type, get_drawing_subtype
from templates.room_templates import process_architectural_drawing

# Load environment variables
load_dotenv()

def is_panel_schedule(file_name: str, raw_content: str) -> bool:
    """
    Check if the drawing is a panel schedule based on filename.
    """
    keywords = ["panel schedule", "electrical schedule"]
    return any(keyword in file_name.lower() for keyword in keywords)

def standardize_field_names(data: Any, drawing_subtype: str) -> Any:
    """
    Standardize field names in the JSON output to handle variations across projects.
    Args:
        data: The JSON data produced by GPT.
        drawing_subtype: The subtype of the drawing (e.g., 'electrical_panel_schedule').
    Returns:
        The standardized JSON data.
    """
    if drawing_subtype == "electrical_panel_schedule":
        # Handle multiple panels if the data is an array
        if isinstance(data, list):
            for panel in data:
                # Standardize fields in 'marks'
                marks = panel.get("marks", {})
                for variant in ["amps", "amperage", "current_rating"]:
                    if variant in marks and "amps" not in marks:
                        marks["amps"] = marks.pop(variant)
                for variant in ["feed", "incoming_feed", "feed_type"]:
                    if variant in marks and "feed" not in marks:
                        marks["feed"] = marks.pop(variant)

                # Standardize fields in 'panel' circuits
                panel_info = panel.get("panel", {})
                for circuit in panel_info.get("circuits", []):
                    # Map load variations to "load_name"
                    for variant in ["load", "description", "loadType", "loadDesc", "load_info"]:
                        if variant in circuit and "load_name" not in circuit:
                            circuit["load_name"] = circuit.pop(variant)
                    # Map trip variations to "trip"
                    for variant in ["ocp", "amperage", "breaker_size", "amp", "trip_rating"]:
                        if variant in circuit and "trip" not in circuit:
                            circuit["trip"] = circuit.pop(variant)

    elif drawing_subtype == "mechanical_schedule":
        # Standardize fields in 'equipment'
        for equipment in data.get("equipment", []):
            for variant in ["volt_ph", "voltage_phase", "volts"]:
                if variant in equipment and "volt_ph" not in equipment:
                    equipment["volt_ph"] = equipment.pop(variant)

    elif drawing_subtype == "plumbing_schedule":
        # Standardize water heater fields
        for heater in data.get("electric_water_heater_schedule", []):
            for variant in ["elec_power_per_unit", "power", "voltage"]:
                if variant in heater and "elec_power_per_unit" not in heater:
                    heater["elec_power_per_unit"] = heater.pop(variant)

    elif drawing_subtype == "architectural_schedule":
        # Standardize wall type fields
        for wall in data.get("wall_types", []):
            details = wall.get("details", {})
            for variant in ["stud_width", "stud_size", "stud_thickness"]:
                if variant in details and "stud_width" not in details:
                    details["stud_width"] = details.pop(variant)

    return data

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
    1) If file name suggests a panel schedule, route to panel_schedule_processor
    2) Otherwise, use drawing subtype to select prompt and process with AI
    """
    file_name = os.path.basename(pdf_path)
    logger = logging.getLogger(__name__)

    # Get drawing subtype
    drawing_subtype = get_drawing_subtype(file_name)
    prompt = PROMPTS.get(drawing_subtype, PROMPTS["default"])

    # Check for panel schedule (existing logic preserved)
    raw_content_placeholder = ""
    if is_panel_schedule(file_name, raw_content_placeholder):
        logger.info(f"Detected panel schedule for {file_name}. Routing to specialized processor.")
        from processing.panel_schedule_processor import process_panel_schedule_pdf_async
        return await process_panel_schedule_pdf_async(pdf_path, client, output_folder, drawing_type)

    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)  # Start

            # Initialize services
            extractor = create_extractor(drawing_type, logger)
            storage = FileSystemStorage(logger)
            ai_service = DrawingAiService(client, {}, logger)  # Empty instructions, using prompt

            # Extract text and tables
            extraction_result = await extractor.extract(pdf_path)
            if not extraction_result.success:
                pbar.update(100)
                logger.error(f"Extraction failed for {pdf_path}: {extraction_result.error}")
                return {"success": False, "error": extraction_result.error, "file": pdf_path}

            # Prepare raw content
            raw_content = extraction_result.raw_text
            for table in extraction_result.tables:
                raw_content += f"\nTABLE:\n{table['content']}\n"

            pbar.update(20)  # Extraction done

            # Process with AI using the prompt
            structured_json = await ai_service.process_with_prompt(
                raw_content=raw_content,
                prompt=prompt,
                model_type=ModelType.GPT_4O_MINI
            )

            pbar.update(40)  # AI processing done

            # Create output directory
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            # Validate JSON
            try:
                parsed_json = json.loads(structured_json)
                # Standardize field names for consistency
                parsed_json = standardize_field_names(parsed_json, drawing_subtype)
            except json.JSONDecodeError as e:
                pbar.update(100)
                logger.error(f"JSON parsing error for {pdf_path}: {str(e)}")
                logger.error(f"Raw response: {structured_json[:200]}...")

                # Save raw for debugging
                raw_output_filename = os.path.splitext(file_name)[0] + '_raw_response.json'
                raw_output_path = os.path.join(type_folder, raw_output_filename)
                await storage.save_text(structured_json, raw_output_path)

                return {"success": False, "error": f"Failed to parse JSON: {str(e)}", "file": pdf_path}

            # Save the structured JSON
            output_filename = os.path.splitext(file_name)[0] + '_structured.json'
            output_path = os.path.join(type_folder, output_filename)
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

        except Exception as e:
            pbar.update(100)
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}
