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
