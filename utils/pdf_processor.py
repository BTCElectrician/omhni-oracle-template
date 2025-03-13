import pymupdf
import json
import os
import logging
from openai import AsyncOpenAI

# Configure logger
logger = logging.getLogger(__name__)

async def extract_text_and_tables_from_pdf(pdf_path: str) -> str:
    """
    Extract text and tables from a PDF file using PyMuPDF.
    """
    logging.info(f"Extracting text and tables from {pdf_path}")
    doc = pymupdf.open(pdf_path)
    all_content = ""
    
    for page_num, page in enumerate(doc):
        # Extract text
        text = page.get_text()
        all_content += f"PAGE {page_num+1} TEXT:\n{text}\n\n"
        
        # Extract tables
        tables = page.find_tables()
        for table_num, table in enumerate(tables):
            all_content += f"PAGE {page_num+1} TABLE {table_num+1}:\n"
            markdown = table.to_markdown()
            all_content += markdown + "\n\n"
    
    # Close the document
    doc.close()
    return all_content

async def structure_panel_data(client: AsyncOpenAI, raw_content: str) -> dict:
    """
    Use OpenAI to structure electrical panel data from raw content.
    """
    prompt = f"""
    You are an expert in electrical engineering and panel schedules. 
    Please structure the following content from an electrical panel schedule into a valid JSON format. 
    The content includes both text and tables. Extract key information such as panel name, voltage, amperage, circuits, 
    and any other relevant details.
    Pay special attention to the tabular data, which represents circuit information.
    Ensure your entire response is a valid JSON object.
    Raw content:
    {raw_content}
    """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that structures electrical panel data into JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def process_pdf(pdf_path: str, output_folder: str, client: AsyncOpenAI):
    """
    Process a PDF file: extract content, structure it, and save to JSON.
    """
    logging.info(f"Processing PDF: {pdf_path}")
    os.makedirs(output_folder, exist_ok=True)
    
    raw_content = await extract_text_and_tables_from_pdf(pdf_path)
    
    structured_data = await structure_panel_data(client, raw_content)
    
    panel_name = structured_data.get('panel_name', 'unknown_panel').replace(" ", "_").lower()
    filename = f"{panel_name}_electric_panel.json"
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    logging.info(f"Saved structured panel data: {filepath}")
    return raw_content, structured_data
