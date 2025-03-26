# Ohmni Oracle

PDF processing pipeline that extracts and structures drawing information using PyMuPDF and GPT-4.

## Features

- Processes multiple drawing types (Architectural, Electrical, Mechanical, Plumbing)
- Intelligent drawing subtype detection for specialized processing
- Modular prompt template system with specialized prompts for each drawing type
- Extracts text and tables from PDFs using PyMuPDF
- Structures data using GPT-4o-mini or GPT-4o with intelligent model selection
- Handles batch processing with rate limiting
- Generates room templates for architectural drawings
- Comprehensive logging and error handling

## Drawing Type Detection

The system automatically detects the following drawing types and subtypes based on filename:

### Electrical Subtypes:
- **Panel Schedules** (Electrical_PANEL_SCHEDULE): Detailed processing of electrical panels, circuits, and loads
- **Lighting** (Electrical_LIGHTING): Specialized extraction of lighting fixtures, controls and circuits
- **Power** (Electrical_POWER): Focused on power outlets, equipment connections, and circuiting
- **Fire Alarm** (Electrical_FIREALARM): Extraction of fire alarm devices, panels, and connections
- **Technology** (Electrical_TECHNOLOGY): Processing of low voltage systems (data, security, AV)
- **Specifications** (Electrical_SPEC): Electrical specifications and requirements

### Architectural Subtypes:
- **Room Plans** (Architectural_ROOM): Focus on room layouts, dimensions, and relationships
- **Ceiling Plans** (Architectural_CEILING): Processing of ceiling heights, materials, and mounted elements
- **Wall Types** (Architectural_WALL): Extraction of wall types, construction details, and ratings
- **Door Details** (Architectural_DOOR): Door schedules, hardware, and specifications
- **Details** (Architectural_DETAIL): Architectural details and sections

### Mechanical Subtypes:
- **Equipment** (Mechanical_EQUIPMENT): HVAC and mechanical equipment specifications
- **Ventilation** (Mechanical_VENTILATION): Air distribution, diffusers, and ductwork
- **Piping** (Mechanical_PIPING): Mechanical piping systems and details

### Plumbing Subtypes:
- **Fixtures** (Plumbing_FIXTURE): Plumbing fixtures and connections
- **Equipment** (Plumbing_EQUIPMENT): Water heaters, pumps, and other plumbing equipment
- **Pipe Systems** (Plumbing_PIPE): Domestic water, waste, and vent piping systems

## Prompt Template System

The system uses a modular prompt template system that provides:

- **Base templates**: Reusable templates for different drawing types
- **Specialized prompts**: Customized instructions for each drawing subtype
- **Few-shot examples**: Example structures to guide the AI's output format
- **Fallback hierarchy**: Default prompts when specific subtypes aren't available

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
- `FORCE_MINI_MODEL`: Set to "true" to force using GPT-4o-mini for all documents (default: false)

### Model Selection

By default, the system intelligently selects between GPT-4o-mini and GPT-4o based on:
- Document type (specifications always use GPT-4o)
- Content length (longer documents use GPT-4o)
- Drawing complexity

Setting `FORCE_MINI_MODEL=true` bypasses this logic and always uses GPT-4o-mini, which:
- Processes documents faster
- Costs less per document
- May reduce quality for complex documents

To test both models and compare results:
```bash
# Process with normal model selection
python main.py /path/to/drawings /path/to/output-normal

# Process with forced mini model
FORCE_MINI_MODEL=true python main.py /path/to/drawings /path/to/output-mini
```

## Output

Processed files are saved as JSON in the output folder, organized by drawing type. 

For architectural drawings, the system also generates template files:
- `e_rooms_details_floor_*.json`: Room templates with electrical information
- `a_rooms_details_floor_*.json`: Room templates with architectural information

Each room in the drawing gets entries in both templates.

## Extending the System

The modular prompt template system makes it easy to add support for new drawing types or subtypes.

### Adding a New Drawing Subtype

To add a new subtype to an existing drawing type:

1. **Add the subtype to the appropriate enum** in `templates/prompt_types.py`:
   ```python
   class ElectricalSubtype(Enum):
       # Existing subtypes...
       NEW_SUBTYPE = "NEW_SUBTYPE"
   ```

2. **Create a prompt function** in the appropriate file under `templates/prompts/`:
   ```python
   # In templates/prompts/electrical.py
   @register_prompt("Electrical", "NEW_SUBTYPE")
   def new_subtype_prompt():
       """Prompt for the new electrical subtype."""
       return create_schedule_template(
           schedule_type="new subtype",
           drawing_category="electrical",
           item_type="item",
           key_properties="key properties to extract",
           example_structure="""
           {
             "ELECTRICAL": {
               "NEW_SUBTYPE": {
                 "field1": "value1",
                 "field2": "value2"
               }
             }
           }
           """,
           source_location="drawing",
           preservation_focus="important elements to preserve",
           stake_holders="who needs this information",
           use_case="how the information is used",
           critical_purpose="why accuracy is important"
       )
   ```

3. **Add detection logic** to `detect_drawing_subtype` in `services/ai_service.py`:
   ```python
   # For electrical subtypes
   if drawing_type == DrawingCategory.ELECTRICAL.value:
       # Existing detection logic...
       elif any(term in file_name_lower for term in ["keyword1", "keyword2"]):
           return f"{drawing_type}_{ElectricalSubtype.NEW_SUBTYPE.value}"
   ```

4. **Update the prompt dictionary** for backward compatibility:
   ```python
   # In templates/prompts/electrical.py
   ELECTRICAL_PROMPTS = {
       # Existing entries...
       "NEW_SUBTYPE": new_subtype_prompt()
   }
   ```

### Adding a Completely New Drawing Type

To add an entirely new drawing type:

1. **Add the new category to the DrawingCategory enum** in `templates/prompt_types.py`:
   ```python
   class DrawingCategory(Enum):
       # Existing categories...
       NEW_CATEGORY = "NewCategory"
   ```

2. **Create a new subtype enum** for the new drawing type:
   ```python
   class NewCategorySubtype(Enum):
       """New drawing category subtypes."""
       TYPE_ONE = "TYPE_ONE"
       TYPE_TWO = "TYPE_TWO"
       DEFAULT = "DEFAULT"
   ```

3. **Create a new prompts file** in `templates/prompts/new_category.py`:
   ```python
   """
   New category prompt templates for construction drawing processing.
   """
   from templates.prompt_registry import register_prompt
   from templates.base_templates import create_general_template, create_schedule_template

   @register_prompt("NewCategory")
   def default_new_category_prompt():
       """Default prompt for new category drawings."""
       return create_general_template(
           # Parameters...
       )

   @register_prompt("NewCategory", "TYPE_ONE")
   def type_one_prompt():
       """Prompt for type one."""
       return create_schedule_template(
           # Parameters...
       )

   # Dictionary of all prompts for backward compatibility
   NEW_CATEGORY_PROMPTS = {
       "DEFAULT": default_new_category_prompt(),
       "TYPE_ONE": type_one_prompt()
   }
   ```

4. **Update the import and prompt categories** in `templates/prompt_templates.py`:
   ```python
   # Import prompt dictionaries
   from templates.prompts.new_category import NEW_CATEGORY_PROMPTS

   # Update mapping
   PROMPT_CATEGORIES = {
       # Existing categories...
       "NewCategory": NEW_CATEGORY_PROMPTS
   }
   ```

5. **Add detection logic** to `detect_drawing_subtype` in `services/ai_service.py`:
   ```python
   # New drawing category subtypes
   elif drawing_type == DrawingCategory.NEW_CATEGORY.value:
       if any(term in file_name_lower for term in ["keyword1", "keyword2"]):
           return f"{drawing_type}_{NewCategorySubtype.TYPE_ONE.value}"
       # Other subtype detection...
   ```

6. **Add any special processing** needed for the new drawing type in `processing/file_processor.py` if required.

7. **Write tests** for the new drawing type detection and prompt retrieval.

### Example JSON Structure

Always include a comprehensive example structure in your prompts:

```json
{
  "NEW_CATEGORY": {
    "metadata": {
      "drawing_number": "NC101",
      "title": "NEW CATEGORY DRAWING",
      "date": "2023-05-15",
      "revision": "2"
    },
    "TYPE_ONE": {
      "field1": "value1",
      "field2": {
        "subfield1": "value2",
        "subfield2": "value3"
      },
      "items": [
        {
          "id": "ITEM-1",
          "specification": "Detailed specifications"
        }
      ]
    },
    "notes": ["Note 1", "Note 2"]
  }
}
```

This structure guides the AI in properly formatting its output, ensuring consistency across drawing types.

## Testing

Run the automated tests to verify system functionality:

```bash
python -m unittest discover -s tests
```

To test a specific component:

```bash
python -m unittest tests/test_prompt_templates.py
``` 