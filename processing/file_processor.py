import os
import json
import logging
import asyncio
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

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
from utils.json_utils import parse_json_safely

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__) # Define logger if not already defined at module level

def normalize_panel_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize field names in parsed panel schedule JSON data for consistency.
    Handles the expected nested structure from the prompt ('ELECTRICAL.PANEL_SCHEDULE.panel').

    Args:
        data: Dictionary containing the parsed JSON data, expected to have the nested structure.

    Returns:
        Dictionary representing the main panel object with normalized field names,
        or the original data if the expected structure isn't found.
    """
    if not isinstance(data, dict):
        logger.warning("normalize_panel_fields received non-dict data, returning as is.")
        return data

    panel_object = None
    panel_name_for_log = "Unknown Panel"

    # Navigate the expected structure: ELECTRICAL -> PANEL_SCHEDULE -> panel
    try:
        if "ELECTRICAL" in data and isinstance(data["ELECTRICAL"], dict) and \
           "PANEL_SCHEDULE" in data["ELECTRICAL"] and isinstance(data["ELECTRICAL"]["PANEL_SCHEDULE"], dict) and \
           "panel" in data["ELECTRICAL"]["PANEL_SCHEDULE"] and isinstance(data["ELECTRICAL"]["PANEL_SCHEDULE"]["panel"], dict):
            panel_object = data["ELECTRICAL"]["PANEL_SCHEDULE"]["panel"]
            panel_name_for_log = panel_object.get("name", "Unknown Panel")
            logger.debug(f"Found panel object '{panel_name_for_log}' within ELECTRICAL.PANEL_SCHEDULE.panel")
        elif "panel" in data and isinstance(data["panel"], dict): # Fallback for flatter structure
             panel_object = data["panel"]
             panel_name_for_log = panel_object.get("name", "Unknown Panel")
             logger.debug(f"Found panel object '{panel_name_for_log}' directly under root (fallback).")
        else:
             logger.warning("Could not find 'panel' object in expected structure for normalization.")
             # Attempt normalization on root if it looks like a panel schedule
             if "circuits" in data and isinstance(data["circuits"], list):
                 panel_object = data
                 panel_name_for_log = panel_object.get("name", "Unknown Root Panel")
                 logger.debug("Attempting normalization on root object as fallback.")
             else:
                  logger.error("Normalization failed: Unexpected data structure, returning original.")
                  return data # Return original if structure is truly unexpected
    except KeyError as e:
         logger.error(f"Normalization failed: Missing expected key '{e}' in data structure.")
         return data
    except Exception as e:
         logger.error(f"Normalization failed: Unexpected error accessing panel data: {e}")
         return data


    if panel_object is None:
         logger.error("Normalization failed: panel_object is None after checks.")
         return data # Should not happen, but safety check

    # --- Normalize circuit fields within the identified panel_object ---
    circuits = panel_object.get("circuits", [])
    if not isinstance(circuits, list):
        logger.warning(f"Panel '{panel_name_for_log}' circuits field is not a list. Skipping circuit normalization.")
        # Still return panel_object as metadata might be useful
        return panel_object

    normalized_circuits = []
    for idx, circuit in enumerate(circuits):
        if not isinstance(circuit, dict):
            logger.warning(f"Skipping non-dict item at index {idx} in circuits list for panel '{panel_name_for_log}': {circuit}")
            normalized_circuits.append(circuit) # Keep non-dict items as is
            continue

        normalized_circuit = circuit.copy() # Work on a copy

        # Normalize load_name synonyms -> 'load_name'
        load_name_synonyms = ["description", "loadType", "load type", "item", "equipment", "load description"]
        for syn in load_name_synonyms:
            if syn in normalized_circuit and "load_name" not in normalized_circuit:
                normalized_circuit["load_name"] = normalized_circuit.pop(syn)
                break

        # Normalize trip synonyms -> 'trip'
        trip_synonyms = ["ocp", "breaker_size", "amperage", "amp", "amps", "size", "rating", "breaker"]
        for syn in trip_synonyms:
            if syn in normalized_circuit and "trip" not in normalized_circuit:
                normalized_circuit["trip"] = normalized_circuit.pop(syn)
                break

        # Normalize circuit synonyms -> 'circuit'
        circuit_synonyms = ["circuit_no", "circuit_number", "ckt", "circuit no", "#", "no"]
        for syn in circuit_synonyms:
             if syn == "no" and ("circuit" in normalized_circuit or len(str(normalized_circuit.get(syn, ''))) > 3):
                 continue # Avoid short 'no' if circuit exists or value is long
             if syn in normalized_circuit and "circuit" not in normalized_circuit:
                normalized_circuit["circuit"] = str(normalized_circuit.pop(syn)) # Ensure circuit is string
                break
             elif syn == "circuit" and syn in normalized_circuit: # Ensure existing circuit is string
                 normalized_circuit["circuit"] = str(normalized_circuit["circuit"])


        # Normalize poles synonyms -> 'poles'
        poles_synonyms = ["pole", "p"]
        for syn in poles_synonyms:
            if syn in normalized_circuit and "poles" not in normalized_circuit:
                normalized_circuit["poles"] = normalized_circuit.pop(syn)
                break

        normalized_circuits.append(normalized_circuit)

    panel_object["circuits"] = normalized_circuits
    logger.info(f"Field normalization applied to panel: '{panel_name_for_log}'")

    # IMPORTANT: Return the potentially modified panel_object itself.
    # The calling function will decide how to save this (e.g., save just this object).
    return panel_object

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
            structured_json_str = None # To store the raw string response from AI

            try:
                # Step 1: Always call process_drawing, passing the detected subtype
                logger.info(f"Calling AI service for file '{file_name}' with subtype '{subtype}'...")
                structured_json_str = await process_drawing(
                    raw_content=raw_content,
                    drawing_type=subtype, # Use the potentially specific subtype for prompt selection
                    client=client,
                    file_name=file_name
                )
                pbar.update(40) # Mark AI processing step complete

                # Step 2: Parse the JSON string response safely
                is_panel_schedule_subtype = "PANEL_SCHEDULE" in subtype
                if structured_json_str:
                    logger.info(f"Attempting to parse JSON response (length: {len(structured_json_str)} chars)...")
                    parsed_json = parse_json_safely(structured_json_str, repair=is_panel_schedule_subtype) # Enable repair for panels
                else:
                     logger.warning(f"AI service returned empty response for {file_name}.")

                # Step 3: Handle parsing failure
                if parsed_json is None and structured_json_str: # Check structured_json_str to ensure it wasn't just an empty response
                    error_msg = "Failed to parse AI response as JSON" + (" (panel schedule, repair attempted)" if is_panel_schedule_subtype else "")
                    logger.error(f"{error_msg} for {file_name}.")
                    # Save the raw response for debugging JSON errors
                    return handle_json_error(structured_json_str, pdf_path, drawing_type, output_folder, file_name, storage, pbar)
                elif parsed_json is None and not structured_json_str:
                    logger.error(f"AI processing failed or returned empty for {file_name}")
                    return {"success": False, "error": "AI processing returned no content", "file": pdf_path}
                else:
                     logger.info(f"Successfully parsed JSON for {file_name}.")


                # Step 4: Apply normalization *specifically* for panel schedules AFTER successful parsing
                if is_panel_schedule_subtype:
                    logger.info(f"Applying field normalization for panel schedule: {file_name}")
                    # Pass the successfully parsed dictionary to the normalizer
                    # Replace parsed_json with the result (which is the core panel object)
                    parsed_json = normalize_panel_fields(parsed_json)
                    logger.info(f"Panel schedule normalization complete for: {file_name}")

            except ValueError as ve: # Catch context length errors etc. from process_drawing
                 logger.error(f"Processing error for {file_name}: {ve}")
                 return {"success": False, "error": str(ve), "file": pdf_path}
            except Exception as e: # Catch any other unexpected errors during AI call/parsing/normalization
                 logger.error(f"Error during AI processing/parsing/normalization for {file_name}: {str(e)}", exc_info=True)
                 # Optionally save raw response on generic error too
                 handle_json_error(structured_json_str, pdf_path, drawing_type, output_folder, file_name, storage, pbar)
                 return {"success": False, "error": f"Processing error: {str(e)}", "file": pdf_path}

            # Step 5: Unified Saving Logic
            if parsed_json: # Check if we have successfully parsed (and potentially normalized) data
                # Use the original main drawing_type for the output subfolder
                type_folder = os.path.join(output_folder, drawing_type)
                os.makedirs(type_folder, exist_ok=True)

                base_name = os.path.splitext(file_name)[0]
                # Determine suffix based on subtype for clarity
                if "PANEL_SCHEDULE" in subtype:
                     suffix = "_panel_schedule_structured.json" # Specific suffix for normalized panel data
                elif "SPECIFICATION" in subtype.upper():
                     suffix = "_specification_structured.json"
                else:
                     suffix = "_structured.json" # General suffix

                output_filename = f"{base_name}{suffix}"
                output_path = os.path.join(type_folder, output_filename)

                # Save the final parsed_json object (which might be the normalized panel data)
                save_success = await storage.save_json(parsed_json, output_path)
                pbar.update(20) # Mark saving step

                if not save_success:
                     logger.error(f"Failed to save output JSON to {output_path}")
                     return {"success": False, "error": f"Failed to save output file", "file": pdf_path}

                logger.info(f"Saved structured data to: {output_path}")

                # Step 6: Architectural Room Template Logic (should function correctly if parsed_json structure is right)
                if drawing_type == 'Architectural' and isinstance(parsed_json, dict):
                    # Check within the parsed_json structure expected from architectural prompts
                    arch_data = parsed_json.get('ARCHITECTURAL', {})
                    rooms_list = arch_data.get('ROOMS', []) if isinstance(arch_data, dict) else []

                    if isinstance(rooms_list, list) and rooms_list: # Check if ROOMS is a non-empty list
                        logger.info(f"Architectural file {file_name} contains rooms. Calling process_architectural_drawing.")
                        try:
                            from templates.room_templates import process_architectural_drawing
                            # Pass the entire parsed_json dictionary
                            result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                            templates_created['floor_plan'] = True # Assuming this flag is managed correctly scope-wise
                            logger.info(f"Finished process_architectural_drawing for {file_name}. Result: {result}")
                        except ImportError:
                            logger.error("Could not import process_architectural_drawing. Skipping room template generation.")
                        except Exception as template_error:
                            logger.error(f"Error during process_architectural_drawing for {file_name}: {template_error}", exc_info=True)
                    else:
                        # Log if architectural but no rooms found in expected structure
                        logger.warning(f"Architectural file {file_name} processed, but 'ARCHITECTURAL.ROOMS' list not found or empty in the JSON. Skipping template generation.")
                else:
                    # Log if the condition is false for debugging
                    if drawing_type == 'Architectural':
                        logger.warning(f"Architectural file {file_name} processed, but 'ARCHITECTURAL.ROOMS' list not found or invalid in the initial JSON. Skipping template generation.")

                pbar.update(10)
                return {"success": True, "file": output_path}
            else:
                # This case should now be caught earlier by parsing failure checks
                logger.error(f"Processing completed but no valid JSON was available for saving for {file_name}")
                return {"success": False, "error": "No valid JSON available post-processing", "file": pdf_path}
                
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