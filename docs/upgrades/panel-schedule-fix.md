mplementation Plan

Phase 1: Add New Utilities and Prompts

Create JSON Utilities File (utils/json_utils.py)

Action: Create a new file.
File Path: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/json_utils.py
Content: Add the following Python code to the new file:
Python

# /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/json_utils.py
import re
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def repair_panel_json(json_str: str) -> str:
    """
    Attempt to repair common JSON syntax errors often found in AI-generated panel schedule responses.

    Args:
        json_str: String containing potentially malformed JSON

    Returns:
        Repaired JSON string, or original if repair heuristics don't apply or fail validation.
    """
    if not isinstance(json_str, str):
        logger.warning("repair_panel_json received non-string input, returning as is.")
        return json_str

    fixed = json_str
    # Attempt to fix missing commas between objects in an array (common issue)
    fixed = re.sub(r'}\s*{', '}, {', fixed)

    # Attempt to fix trailing commas before closing brackets/braces (strict JSON invalid)
    fixed = re.sub(r',\s*}', '}', fixed)
    fixed = re.sub(r',\s*\]', ']', fixed)

    # Attempt to fix missing quotes around keys (heuristic, might be imperfect)
    try:
        fixed = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
    except Exception as e:
        logger.warning(f"Regex error during key quoting repair: {e}")

    # Attempt to fix truncated JSON by adding missing closing brackets/braces
    open_braces = fixed.count('{')
    close_braces = fixed.count('}')
    open_brackets = fixed.count('[')
    close_brackets = fixed.count(']')

    if open_braces > close_braces:
        fixed += '}' * (open_braces - close_braces)
        logger.debug(f"Added {open_braces - close_braces} closing braces.")
    if open_brackets > close_brackets:
        fixed += ']' * (open_brackets - close_brackets)
        logger.debug(f"Added {open_brackets - close_brackets} closing brackets.")

    # Final validation check
    try:
        json.loads(fixed)
        if fixed != json_str:
            logger.info("JSON repair applied successfully.")
        return fixed
    except json.JSONDecodeError:
        logger.warning("JSON repair attempt failed validation. Returning original string.")
        return json_str

def parse_json_safely(json_str: str, repair: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse JSON string with an optional fallback repair attempt.

    Args:
        json_str: JSON string to parse.
        repair: If True, attempt to repair the JSON string if initial parsing fails.

    Returns:
        Parsed JSON object (Dict) or None if parsing failed even after repair.
    """
    if not isinstance(json_str, str):
        logger.error("parse_json_safely received non-string input.")
        return None
    try:
        # Try standard parsing first
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parsing failed: {e}. Raw start: {json_str[:100]}...")
        if repair:
            logger.info("Attempting JSON repair...")
            repaired_str = repair_panel_json(json_str)
            try:
                # Try parsing the repaired string
                parsed_obj = json.loads(repaired_str)
                logger.info("Successfully parsed repaired JSON.")
                return parsed_obj
            except json.JSONDecodeError as e2:
                # Still failed after repair attempt
                logger.error(f"JSON parsing failed even after repair attempt: {e2}")
                logger.error(f"Repaired string snippet: {repaired_str[:500]}...")
                return None
        else:
            # No repair requested, return None
            return None
    except Exception as ex:
        logger.error(f"Unexpected error during JSON parsing: {ex}")
        return None
Verification: Ensure the file /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/json_utils.py is created with the exact content above.
Update Panel Schedule Prompt (templates/prompts/electrical.py)

Action: Modify the existing file.
File Path: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/electrical.py
Instructions:
Ensure register_prompt and create_schedule_template are imported.
Replace the entire panel_schedule_prompt function definition with the code block below. Make sure the @register_prompt("Electrical", "PANEL_SCHEDULE") decorator is present.
Ensure the ELECTRICAL_PROMPTS dictionary at the bottom of the file correctly references the updated panel_schedule_prompt function.
Code to Add/Replace:
Python

# Add/Modify in /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/prompts/electrical.py

# Ensure these imports are present at the top
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

# ... (other prompts like default_electrical_prompt) ...

@register_prompt("Electrical", "PANEL_SCHEDULE")
def panel_schedule_prompt():
    """Prompt for electrical panel schedules."""
    base_prompt = create_schedule_template(
        schedule_type="panel schedule",
        drawing_category="electrical",
        item_type="circuit",
        # Instructions integrated into key_properties for clarity
        key_properties="CRITICAL: First, create a 'DRAWING_METADATA' object containing drawing metadata like drawing_number, title, revision, date, job_number, project_name found in the title block. Second, create a main 'ELECTRICAL' object. Inside 'ELECTRICAL', create a 'PANEL_SCHEDULE' object. Inside 'PANEL_SCHEDULE', create a 'panel' object holding ALL panel metadata found (e.g., name, location, voltage, phases, wires, main_breaker/MLO, rating, aic_rating, mounting, enclosure, fed_from). This 'panel' object MUST also contain a 'circuits' list. For EACH circuit listed in the schedule, create a JSON object in the 'circuits' list containing AT LEAST 'circuit' number, 'trip' size (breaker amps), 'poles', and 'load_name'. Include ANY other details provided per circuit (VA, room, notes, GFCI status, etc.). Extract any general notes related to panels into an 'ELECTRICAL.general_notes' list.",
        example_structure="""
{
  "DRAWING_METADATA": {
    "drawing_number": "E4.01",
    "title": "PANEL SCHEDULES",
    "revision": "1",
    "date": "2024-01-15",
    "job_number": "P12345",
    "project_name": "Sample Project"
  },
  "ELECTRICAL": {
    "PANEL_SCHEDULE": {
      "panel": {
        "name": "K1",
        "location": "Kitchen 118",
        "voltage": "120/208 Wye",
        "phases": 3,
        "wires": 4,
        "main_breaker": "30 A Main Breaker",
        "rating": "225 A",
        "aic_rating": "14K",
        "mounting": "Surface",
        "enclosure": "NEMA 1",
        "fed_from": "MDP",
        "circuits": [
          {
            "circuit": "1",
            "load_name": "Kitchen Equipment - Refrigerator",
            "trip": "20 A",
            "poles": 1,
            "va_per_pole": 1200,
            "room_id": ["Kitchen 118"],
            "notes": "GFCI Breaker"
          },
          {
            "circuit": "3,5",
            "load_name": "Oven",
            "trip": "50 A",
            "poles": 2,
            "va_per_pole": 4800,
            "room_id": ["Kitchen 118"]
          }
        ],
        "panel_totals": {
           "total_connected_load_va": 25600,
           "total_demand_load_va": 21800,
           "total_connected_amps": 71.1,
           "total_demand_amps": 60.5
        }
      }
    },
    "general_notes": [
        "Verify all panel locations with architectural drawings.",
        "All breakers to be bolt-on type."
    ]
  }
}
""",
        source_location="panel schedule drawings or tables",
        preservation_focus="panel metadata, ALL circuit numbers, trip sizes, poles, and load descriptions",
        stake_holders="Electrical engineers, estimators, and installers",
        use_case="critical electrical system design, load calculation, and installation",
        critical_purpose="preventing safety hazards, ensuring code compliance, and proper circuit protection"
    )

    # Append critical formatting requirements
    formatting_reqs = """

CRITICAL FORMATTING REQUIREMENTS (Strict JSON):
1. Ensure ALL property names (keys) are enclosed in double quotes (e.g., "name": "K1").
2. ALL string values must be enclosed in double quotes (e.g., "voltage": "120/208 Wye"). Numeric values should NOT be quoted (e.g., "poles": 1). Boolean values (true/false) should NOT be quoted.
3. Ensure ALL items in an array (like 'circuits' or 'room_id') are separated by commas.
4. There must be NO trailing comma after the last item in an array or the last key-value pair in an object.
5. Objects must start with '{' and end with '}'. Arrays must start with '[' and end with ']'.
6. Each circuit in the 'circuits' array MUST be a complete JSON object '{...}'.

Example of correct circuit array formatting:
"circuits": [
  { "circuit": "1", "load_name": "Equipment", "trip": "20 A", "poles": 1 },
  { "circuit": "2", "load_name": "Lighting", "trip": "15 A", "poles": 1 }
]
(NO comma after the last circuit object '}')
"""
    return base_prompt + formatting_reqs

# ... (other prompts like lighting_fixture_prompt, etc.) ...

# Ensure this dictionary at the end of the file is updated
ELECTRICAL_PROMPTS = {
    "DEFAULT": default_electrical_prompt(),
    "PANEL_SCHEDULE": panel_schedule_prompt(), # Make sure this references the function above
    "LIGHTING": lighting_fixture_prompt(),
    "POWER": power_connection_prompt(),
    "SPEC": electrical_spec_prompt()
}
Verification: Check the file content, decorator, function definition, and the update to the ELECTRICAL_PROMPTS dictionary.
Phase 2: Modify Core Processing Logic

Enhance Subtype Detection (services/ai_service.py)

Action: Modify the existing file.
File Path: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/ai_service.py
Instructions:
Add import re and import os at the top of the file if they are missing.
Locate the detect_drawing_subtype function.
Inside this function, within the if drawing_type == DrawingCategory.ELECTRICAL.value: block, replace the existing logic used to detect panel schedules with the following code block which uses more robust regular expressions.
Code to Replace Existing Panel Detection Logic With:
Python

# This block replaces the existing panel detection inside the 'if drawing_type == DrawingCategory.ELECTRICAL.value:' section
# --- Start Replacement ---
# Look for stronger panel schedule indicators using regex
panel_indicators_regex = [
    r"panel", r"schedule", r"panelboard", r"circuit",
    r"breaker", r"distribution", r"single line", r"riser", r"one line", # Added riser/one-line
    # Panel naming patterns (simple examples, case-insensitive matching)
    r"^[a-z][0-9]+[a-z]?$",                  # Matches names like h1, k1s, l1a at start of filename
    r"[a-z][0-9]+[a-z]?(-| panel| schedule)", # Matches names like h1-panel, k1s schedule anywhere
    r"^[0-9]{1,2}[a-z]{1,2}(p|h)?-[0-9]+$",   # Matches names like 21lp-1, 20h-1 at start
    # Add more specific project patterns here if known, e.g. r"^lp-.*"
]
# Match against the base name without extension, case-insensitive
base_name_lower = os.path.splitext(os.path.basename(file_name))[0].lower()
is_panel_schedule = False
for pattern in panel_indicators_regex:
    # Try searching anywhere in the full filename first
    if re.search(pattern, file_name.lower()):
         is_panel_schedule = True
         logger.debug(f"Panel indicator '{pattern}' found in filename '{file_name}'")
         break
    # Try matching specific patterns (like ^...$) against the base name
    # Adjust regex if they shouldn't be start/end anchored (remove ^ and $)
    if pattern.startswith('^') and re.match(pattern, base_name_lower):
         is_panel_schedule = True
         logger.debug(f"Panel pattern '{pattern}' matched base name '{base_name_lower}'")
         break

if is_panel_schedule:
    logger.info(f"Detected PANEL_SCHEDULE subtype for '{file_name}'")
    return f"{drawing_type}_{ElectricalSubtype.PANEL_SCHEDULE.value}"
# --- End Replacement ---

# The existing 'elif' conditions for other electrical subtypes (LIGHTING, POWER, etc.) should follow here
elif any(term in file_name_lower for term in ["light", "lighting", "fixture", "lamp", "luminaire", "rcp", "ceiling"]):
     logger.info(f"Detected LIGHTING subtype for '{file_name}'")
     return f"{drawing_type}_{ElectricalSubtype.LIGHTING.value}"
# ... rest of existing electrical subtype checks ...
Verification: Ensure import re and import os are present. Confirm the panel schedule detection logic within the Electrical block uses the new regex code.
Implement Normalization Function (processing/file_processor.py)

Action: Modify the existing file.
File Path: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/file_processor.py
Instructions: Add the entire normalize_panel_fields function definition within this file, outside of any other function (e.g., near the top after imports, or at the bottom).
Code to Add:
Python

# Add this function definition inside /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/file_processor.py

import logging # Ensure logging is imported at the top
from typing import Dict, Any, List, Optional # Ensure typing is imported at the top

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
Verification: Ensure the function normalize_panel_fields is added correctly to the file.
Modify Main File Processing Logic (processing/file_processor.py)

Action: Modify the process_pdf_async function within the existing file.
File Path: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/file_processor.py
Instructions:
Add the import from utils.json_utils import parse_json_safely at the top of the file.
Ensure import json and import os are present.
Locate the main try...except block within the process_pdf_async function after the extraction_result is obtained.
Replace the entire section that handles the conditional processing (like the old elif "PanelSchedule" in subtype... and the else: block calling process_drawing) up to the point where parsed_json is saved, with the new unified logic block provided below.
Code to Replace Existing Processing/Parsing/Normalization Section With:
Python

# This block replaces the conditional processing/parsing logic inside process_pdf_async

# --- Start Replacement ---
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
    # End Architectural Room Logic

    pbar.update(10) # Final progress update for the file
    return {"success": True, "file": output_path} # Return success with the path to the saved file
else:
     # This case should now be caught earlier by parsing failure checks
     logger.error(f"Processing completed but no valid JSON was available for saving for {file_name}")
     return {"success": False, "error": "No valid JSON available post-processing", "file": pdf_path}

# --- End Replacement ---
Verification: Read through the process_pdf_async function. Confirm the old conditional logic is gone. Verify the new sequence: process_drawing call -> parse_json_safely call (with repair=True for panels) -> handle parsing failure -> normalize_panel_fields call (only for panels) -> save final parsed_json. Ensure the architectural room logic remains and accesses parsed_json correctly.
Phase 3: Cleanup and Removal

Remove Old Panel Processor Code References (processing/file_processor.py)

Action: Modify the existing file.
File Path: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/file_processor.py
Instructions:
Delete the line from processing.panel_schedule_processor import process_panel_schedule_content_async from the imports section (if it still exists).
Delete the entire function definition for is_panel_schedule(file_name: str, raw_content: str) -> bool.
Verification: Ensure the file no longer contains the specified import or function definition. Search the file to ensure no references remain.
Delete Old Panel Processor File

Action: Delete the file from the file system.
File Path: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/panel_schedule_processor.py
Verification: Confirm the file no longer exists in the processing directory.