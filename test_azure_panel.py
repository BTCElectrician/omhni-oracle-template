import os
import sys
import json
import logging
import asyncio

from dotenv import load_dotenv
from processing.file_processor import is_panel_schedule
from utils.pdf_processor import extract_text_and_tables_from_pdf

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_azure_panel.py <pdf_file_or_folder>")
        sys.exit(1)

    path_arg = sys.argv[1]
    if not os.path.exists(path_arg):
        print(f"Error: Path '{path_arg}' does not exist.")
        sys.exit(1)

    load_dotenv()

    pdf_files = []
    if os.path.isfile(path_arg) and path_arg.lower().endswith(".pdf"):
        pdf_files.append(path_arg)
    elif os.path.isdir(path_arg):
        for root, _, files in os.walk(path_arg):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))

    output_folder = os.path.join(os.getcwd(), "test_output")
    os.makedirs(output_folder, exist_ok=True)

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)

        if is_panel_schedule(file_name, ""):
            logging.info(f"Detected panel schedule in '{file_name}'.")
            try:
                # Extract with PyMuPDF (async call)
                raw_content = await extract_text_and_tables_from_pdf(pdf_path)
                result_data = {
                    "extracted_content": raw_content,
                    "error": None
                }
                logging.info(f"Successfully processed '{file_name}'.")
            except Exception as e:
                logging.exception(f"Error: {e}")
                result_data = {"extracted_content": "", "error": str(e)}

            out_file = os.path.join(
                output_folder,
                f"{os.path.splitext(file_name)[0]}_test_panel.json"
            )
            with open(out_file, "w") as f:
                json.dump(result_data, f, indent=2)
            logging.info(f"Wrote output to '{out_file}'")

        else:
            logging.info(f"'{file_name}' is NOT flagged as a panel schedule.")

if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
