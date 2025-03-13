# Ohmni Oracle

PDF processing pipeline that extracts and structures drawing information using PyMuPDF and GPT-4.

## Features

- Processes multiple drawing types (Architectural, Electrical, etc.)
- Extracts text and tables from PDFs using PyMuPDF
- Structures data using GPT-4o-mini
- Handles batch processing with rate limiting
- Generates room templates for architectural drawings
- Comprehensive logging and error handling

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your OpenAI API key

## Usage

```bash
python main.py <input_folder> [output_folder]
```

## Configuration

Environment variables in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `BATCH_SIZE`: PDFs to process in parallel (default: 10)
- `API_RATE_LIMIT`: Max API calls per time window (default: 60)
- `TIME_WINDOW`: Time window in seconds (default: 60)

## Output

Processed files are saved as JSON in the output folder, organized by drawing type. 