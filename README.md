# Ohmni Oracle

PDF processing pipeline that extracts and structures drawing information using PyMuPDF and GPT-4.

## Features

- Processes multiple drawing types (Architectural, Electrical, etc.)
- Automatic detection of drawing subtypes for specialized processing
- Extracts text and tables from PDFs using PyMuPDF
- Structures data using GPT-4o-mini
- Handles batch processing with rate limiting
- Generates room templates for architectural drawings
- Comprehensive logging and error handling

## Drawing Type Detection

The system automatically detects the following drawing subtypes based on filename:

### Electrical Subtypes:
- **Panel Schedules** (Electrical_PanelSchedule): Detailed processing of electrical panels, circuits, and loads
- **Lighting** (Electrical_Lighting): Specialized extraction of lighting fixtures, controls and circuits
- **Power** (Electrical_Power): Focused on power outlets, equipment connections, and circuiting
- **Fire Alarm** (Electrical_FireAlarm): Extraction of fire alarm devices, panels, and connections
- **Technology** (Electrical_Technology): Processing of low voltage systems (data, security, AV)

### Architectural Subtypes:
- **Floor Plans** (Architectural_FloorPlan): Focus on room layouts, dimensions, and relationships
- **Reflected Ceiling Plans** (Architectural_ReflectedCeiling): Processing of ceiling heights, materials, and mounted elements
- **Partition Types** (Architectural_Partition): Extraction of wall types, construction details, and ratings

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

Processed files are saved as JSON in the output folder, organized by drawing type. Each drawing is processed using specialized instructions based on its detected subtype.

## Extending the System

To add support for a new drawing subtype:

1. Add a new entry to the `DRAWING_INSTRUCTIONS` dictionary in `services/ai_service.py`:
   ```python
   "NewType_NewSubtype": """
   Detailed instructions for processing this subtype...
   """
   ```

2. Update the `detect_drawing_subtype` function in `services/ai_service.py`:
   ```python
   if drawing_type == "NewType":
       # Add detection for the new subtype
       if any(term in file_name_lower for term in ["keyword1", "keyword2"]):
           return "NewType_NewSubtype"
   ```

3. Run the tests to verify the detection works correctly. 