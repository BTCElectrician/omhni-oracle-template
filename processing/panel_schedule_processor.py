import os
import json
import logging
from typing import Dict, Any, List
from tqdm.asyncio import tqdm

from services.extraction_service import PyMuPdfExtractor
from services.storage_service import FileSystemStorage
from services.ai_service import DrawingAiService, AiRequest, ModelType
from utils.performance_utils import time_operation

# How many lines we feed to GPT in each chunk
DEFAULT_CHUNK_SIZE = 30

def split_text_into_chunks(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """
    Splits a text string into multiple chunks of N lines each.
    This prevents passing huge blocks to GPT at once.
    """
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), chunk_size):
        chunk_slice = lines[i : i + chunk_size]
        chunk_str = "\n".join(chunk_slice)
        chunks.append(chunk_str)
    return chunks

def normalize_panel_data_fields(panel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optionally unify synonyms in GPT's JSON:
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

@time_operation("total_processing")
async def process_panel_schedule_pdf_async(pdf_path: str, client, output_folder: str) -> Dict[str, Any]:
    """
    Specialized function for panel schedules:
    1) Extract with PyMuPdf
    2) If any tables found, parse them chunk-by-chunk
    3) If no tables, fallback to raw text chunking
    4) Merge partial results & synonyms
    5) Save to _panel_schedules.json
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

            # Write final JSON
            base_name = os.path.splitext(file_name)[0]
            output_filename = f"{base_name}_panel_schedules.json"
            output_path = os.path.join(output_folder, output_filename)

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
    For each table's markdown, chunk it and parse with GPT.
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

        # Break table Markdown into chunks
        table_chunks = split_text_into_chunks(table_md, DEFAULT_CHUNK_SIZE)
        merged_panel = {
            "panel_name": None,
            "panel_metadata": {},
            "circuits": []
        }

        for cidx, chunk_text in enumerate(table_chunks):
            user_text = f"TABLE CHUNK {cidx+1}/{len(table_chunks)}:\n{chunk_text}"

            # Build AiRequest, not a dict
            request = AiRequest(
                content=user_text,
                model_type=ModelType.GPT_4O_MINI,
                temperature=0.2,
                max_tokens=3000,
                system_message=system_prompt
            )

            response = await ai_service.process(request)
            if not response.success or not response.content:
                logger.warning(f"GPT parse error on table {i}, chunk {cidx}: {response.error}")
                continue

            try:
                partial_json = json.loads(response.content)
                # Merge partial
                if "panel_name" in partial_json and not merged_panel["panel_name"]:
                    merged_panel["panel_name"] = partial_json["panel_name"]

                if "panel_metadata" in partial_json and isinstance(partial_json["panel_metadata"], dict):
                    merged_panel["panel_metadata"].update(partial_json["panel_metadata"])

                if "circuits" in partial_json and isinstance(partial_json["circuits"], list):
                    merged_panel["circuits"].extend(partial_json["circuits"])

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error (table {i}, chunk {cidx}): {str(e)}")

        # Normalize synonyms
        merged_panel = normalize_panel_data_fields(merged_panel)
        # If we found circuits or a panel name, add it
        if merged_panel["panel_name"] or merged_panel["circuits"]:
            all_panels.append(merged_panel)

    return all_panels

async def _fallback_raw_text(raw_text: str, client, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    If no tables found, we chunk the entire raw_text and let GPT parse lines as circuits.
    Return a list of one or more panels if discovered.
    """
    from services.ai_service import AiRequest

    ai_service = DrawingAiService(client, drawing_instructions={}, logger=logger)

    text_chunks = split_text_into_chunks(raw_text, DEFAULT_CHUNK_SIZE)

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

    fallback_data = {
        "panel_name": "",
        "panel_metadata": {},
        "circuits": []
    }

    for idx, chunk_text in enumerate(text_chunks):
        user_text = f"RAW TEXT CHUNK {idx+1}/{len(text_chunks)}:\n{chunk_text}"

        # Create AiRequest
        request = AiRequest(
            content=user_text,
            model_type=ModelType.GPT_4O_MINI,
            temperature=0.2,
            max_tokens=3000,
            system_message=fallback_prompt
        )

        response = await ai_service.process(request)
        if not response.success or not response.content:
            logger.warning(f"GPT parse error in fallback chunk {idx}: {response.error}")
            continue

        try:
            partial_json = json.loads(response.content)
            # Merge partial
            if "panel_name" in partial_json and not fallback_data["panel_name"]:
                fallback_data["panel_name"] = partial_json["panel_name"]

            if "panel_metadata" in partial_json and isinstance(partial_json["panel_metadata"], dict):
                fallback_data["panel_metadata"].update(partial_json["panel_metadata"])

            if "circuits" in partial_json and isinstance(partial_json["circuits"], list):
                fallback_data["circuits"].extend(partial_json["circuits"])

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in fallback chunk {idx}: {str(e)}")

    # unify synonyms
    fallback_data = normalize_panel_data_fields(fallback_data)

    # If no circuits or panel name found, return empty list
    if not fallback_data["panel_name"] and not fallback_data["circuits"]:
        return []

    return [fallback_data]
