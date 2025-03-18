import os
import json
import logging
from typing import Dict, Any, List
from tqdm.asyncio import tqdm

from services.extraction_service import PyMuPdfExtractor
from services.storage_service import FileSystemStorage
from services.ai_service import DrawingAiService, AiRequest, ModelType

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
    Unify synonyms in GPT's JSON:
      - 'description', 'loadType' => 'load_name'
      - 'ocp', 'amperage', 'breaker_size' => 'trip'
      - 'circuit_no', 'circuit_number' => 'circuit'
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

        # trip synonyms
        if "ocp" in c and "trip" not in c:
            c["trip"] = c.pop("ocp")
        if "breaker_size" in c and "trip" not in c:
            c["trip"] = c.pop("breaker_size")
        if "amperage" in c and "trip" not in c:
            c["trip"] = c.pop("amperage")
        if "amp" in c and "trip" not in c:
            c["trip"] = c.pop("amp")

        # circuit synonyms
        if "circuit_no" in c and "circuit" not in c:
            c["circuit"] = c.pop("circuit_no")
        if "circuit_number" in c and "circuit" not in c:
            c["circuit"] = c.pop("circuit_number")

        new_circuits.append(c)

    panel_data["circuits"] = new_circuits
    return panel_data

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

            tables = extraction_result.tables
            raw_text = extraction_result.raw_text

            if tables:
                logger.info(f"Found {len(tables)} table(s) in {file_name}. Using table-based parsing.")
                panels_data = await _parse_tables_chunked(tables, client, logger)
            else:
                logger.warning(f"No tables found in {file_name}—fallback to raw text approach.")
                panels_data = await _fallback_raw_text(raw_text, client, logger)

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

async def _parse_tables_chunked(tables: List[Dict[str, Any]], client, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Process each table's markdown as a complete unit - no chunking.
    Returns a list of panel objects (one per table).
    """
    from services.ai_service import AiRequest

    ai_service = DrawingAiService(client, drawing_instructions={}, logger=logger)
    all_panels = []

    system_prompt = """
You are an advanced electrical-engineering assistant. I'm giving you a table from a panel schedule in Markdown form.
Synonyms:
  - 'Load', 'Load Type', 'Description' => 'load_name'
  - 'Trip', 'OCP', 'Breaker Size', 'Amperage', 'Amp' => 'trip'
  - 'Circuit No', 'Circuit Number' => 'circuit'
Return valid JSON like:
{
  "panel_name": "...",
  "panel_metadata": {},
  "circuits": [
    { "circuit": "...", "load_name": "...", "trip": "...", "poles": "...", ... }
  ]
}
Do not skip any circuit rows. If missing data, leave blank.
    """.strip()

    for i, tbl_info in enumerate(tables):
        table_md = tbl_info["content"]
        if not table_md.strip():
            logger.debug(f"Skipping empty table {i}.")
            continue

        # Process the entire table as a single unit - NO CHUNKING
        user_text = f"FULL TABLE:\n{table_md}"

        request = AiRequest(
            content=user_text,
            model_type=ModelType.GPT_4O_MINI,
            temperature=0.2,
            max_tokens=3000,
            system_message=system_prompt
        )

        response = await ai_service.process(request)
        if not response.success or not response.content:
            logger.warning(f"GPT parse error on table {i}: {response.error}")
            continue

        try:
            panel_json = json.loads(response.content)
            # Normalize synonyms
            panel_json = normalize_panel_data_fields(panel_json)
            # If we found circuits or a panel name, add it
            if panel_json.get("panel_name") or panel_json.get("circuits"):
                all_panels.append(panel_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error (table {i}): {str(e)}")

    return all_panels

async def _fallback_raw_text(raw_text: str, client, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    If no tables found, process the entire raw_text as a single unit.
    Return a list of one or more panels if discovered.
    """
    from services.ai_service import AiRequest

    ai_service = DrawingAiService(client, drawing_instructions={}, logger=logger)

    fallback_prompt = """
You are an electrical-engineering assistant. I have raw text from a panel schedule PDF.
Columns might be unclear. Return JSON like:
{
  "panel_name": "...",
  "panel_metadata": {},
  "circuits": [
    { "circuit": "...", "load_name": "...", "trip": "...", "poles": "...", ... }
  ]
}
Do not skip lines that look like circuit data.
    """.strip()

    # Process the entire content as a single unit - NO CHUNKING
    user_text = f"FULL RAW TEXT:\n{raw_text}"

    request = AiRequest(
        content=user_text,
        model_type=ModelType.GPT_4O_MINI,
        temperature=0.2,
        max_tokens=3000,
        system_message=fallback_prompt
    )

    response = await ai_service.process(request)
    if not response.success or not response.content:
        logger.warning(f"GPT parse error in fallback processing: {response.error}")
        return []

    try:
        fallback_data = json.loads(response.content)
        # Normalize synonyms
        fallback_data = normalize_panel_data_fields(fallback_data)

        # If no circuits or panel name found, return empty list
        if not fallback_data.get("panel_name") and not fallback_data.get("circuits"):
            return []

        return [fallback_data]
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in fallback processing: {str(e)}")
        return []
