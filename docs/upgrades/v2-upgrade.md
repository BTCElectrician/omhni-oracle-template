'''
Construction Document Processor
-------------------------------
Integrating specialized extraction prompts for different construction documents
'''

# Folder Structure (Proposed Implementation):
# 
# document_processor/
# ├── __init__.py
# ├── main.py                    # Main entry point
# ├── config.py                  # Configuration settings
# ├── extractors/
# │   ├── __init__.py
# │   ├── base_extractor.py      # Base class for all extractors
# │   ├── electrical_extractor.py
# │   ├── mechanical_extractor.py
# │   ├── plumbing_extractor.py
# │   ├── architectural_extractor.py
# │   └── civil_extractor.py
# ├── prompts/
# │   ├── __init__.py
# │   ├── prompt_registry.py     # Maps file patterns to prompts
# │   ├── electrical_prompts.py  # Contains all electrical-related prompts
# │   ├── mechanical_prompts.py
# │   ├── plumbing_prompts.py
# │   ├── architectural_prompts.py
# │   └── civil_prompts.py
# ├── processors/
# │   ├── __init__.py
# │   └── file_processor.py      # Processes individual files
# ├── utils/
# │   ├── __init__.py
# │   ├── file_utils.py          # File handling utilities
# │   ├── schema_validator.py    # JSON schema validation
# │   └── llm_client.py          # Client for LLM API
# └── schemas/
#     ├── __init__.py
#     ├── electrical_schemas.py  # JSON schemas for validation
#     ├── mechanical_schemas.py
#     ├── plumbing_schemas.py
#     └── architectural_schemas.py

# Let's implement key components of this system

# ======== prompts/prompt_registry.py ========

import re
import importlib

class PromptRegistry:
    """Maps document patterns to their appropriate prompts"""
    
    # File pattern to module mapping
    PATTERN_TO_MODULE = {
        r'(?i)panel|LP-|H\d+\.json|K\d+\.json': 'electrical_prompts',
        r'(?i)fixture|lighting|Light': 'electrical_prompts',
        r'(?i)roomdetails|room': 'electrical_prompts',
        r'(?i)mech|mechanical|HVAC': 'mechanical_prompts',
        r'(?i)plumb|plumbing': 'plumbing_prompts',
        r'(?i)door|doorhardware|doorspecs': 'architectural_prompts',
        r'(?i)wall|walltype': 'architectural_prompts',
        r'(?i)site|civil|grading': 'civil_prompts'
    }
    
    # File pattern to prompt function mapping
    PATTERN_TO_PROMPT = {
        r'(?i)panel|LP-|H\d+\.json|K\d+\.json': 'get_panel_schedule_prompt',
        r'(?i)fixture|lighting|Light': 'get_lighting_fixture_prompt',
        r'(?i)roomdetails|room': 'get_room_electrical_prompt',
        r'(?i)mech|mechanical|HVAC': 'get_mechanical_prompt',
        r'(?i)plumb|plumbing': 'get_plumbing_prompt',
        r'(?i)door|doorhardware|doorspecs': 'get_door_schedule_prompt',
        r'(?i)wall|walltype': 'get_wall_type_prompt',
        r'(?i)site|civil|grading': 'get_civil_prompt'
    }
    
    @classmethod
    def get_prompt_for_file(cls, filename):
        """
        Determine the appropriate prompt for a given filename
        
        Args:
            filename: Name of the file to process
            
        Returns:
            tuple: (prompt text, prompt type)
        """
        for pattern, prompt_func in cls.PATTERN_TO_PROMPT.items():
            if re.search(pattern, filename):
                for module_pattern, module_name in cls.PATTERN_TO_MODULE.items():
                    if re.search(module_pattern, filename):
                        module = importlib.import_module(f'document_processor.prompts.{module_name}')
                        prompt_function = getattr(module, prompt_func)
                        return prompt_function(), prompt_func
        
        # Default prompt if no match is found
        return None, None

# ======== prompts/electrical_prompts.py ========

def get_panel_schedule_prompt():
    """Returns the prompt for extracting electrical panel schedules"""
    return """
Extract the complete electrical panel schedule data into this JSON structure:

{
  "panelboard_schedule": {
    "designation": "[panel name/number]",
    "main_type": "[MLO, MCB, etc.]",
    "mounting": "[Surface, Flush, etc.]",
    "branch_ocp_type": "[circuit breaker type]",
    "voltage": "[full voltage spec]",
    "phases": [number of phases],
    "aic_rating": "[if provided]",
    "circuits": [
      {
        "circuit_no": "[exact circuit number/range as shown]",
        "description": "[load description]",
        "ocp": "[breaker trip rating with units]",
        "poles": [integer or string],
        "room_id": ["array of rooms served, if provided"],
        "equipment_ref": "[referenced equipment ID]"
      }
    ],
    "panel_totals": {
      "total_connected_load": "[if provided]",
      "total_estimated_demand": "[if provided]",
      "total_connected_amps": "[if provided]"
    }
  }
}

Important details:
1. Capture multi-pole circuits (e.g., "2/4", "13-15-17")
2. Preserve exact circuit numbering format
3. Include room references using standard room number format
4. Cross-reference equipment where indicated
5. Note any panels that feed other panels or transformers
"""

def get_lighting_fixture_prompt():
    """Returns the prompt for extracting lighting fixture schedules"""
    return """
Extract the complete lighting fixture schedule data into this JSON structure:

{
  "lighting_schedule": {
    "fixtures": [
      {
        "type": "[fixture type/ID]",
        "description": "[fixture description]",
        "watts": "[wattage with units]",
        "lamp_type": "[LED, etc.]",
        "voltage": "[voltage with units]",
        "manufacturer": "[manufacturer name]",
        "catalog_series": "[catalog number/model]",
        "ballast_type": "[ballast/driver type]",
        "mounting": "[ceiling, wall, etc.]",
        "dimensions": "[if provided]",
        "dimmable": "[yes/no if specified]",
        "location": "[typical locations]",
        "remarks": "[any additional notes]"
      }
    ],
    "notes": ["Array of general notes about fixtures"],
    "sensors": [
      {
        "sensor_id": "[sensor type ID]",
        "type": "[full sensor description]",
        "model": "[model number]",
        "description": "[functionality details]"
      }
    ]
  }
}

Important details:
1. Capture all technical specifications for each fixture type
2. Include sensor types and controls when provided
3. Preserve manufacturer catalog numbers exactly as written
4. Note any special mounting requirements or locations
"""

def get_room_electrical_prompt():
    """Returns the prompt for extracting room electrical details"""
    return """
Extract the complete room electrical details into this JSON structure:

{
  "rooms": [
    {
      "room_id": "[room identifier]",
      "room_name": "[room name/function]",
      "circuits": {
        "lighting": ["array of lighting circuit IDs"],
        "power": ["array of power circuit IDs"]
      },
      "light_fixtures": {
        "fixture_ids": ["array of fixture types used"],
        "fixture_count": {
          "[fixture_type]": [quantity]
        }
      },
      "outlets": {
        "regular_outlets": [quantity],
        "controlled_outlets": [quantity],
        "special_outlets": [if applicable]
      },
      "data": [number of data outlets],
      "floor_boxes": [quantity],
      "mechanical_equipment": ["array of equipment IDs"],
      "switches": {
        "type": "[switch/sensor type]",
        "model": "[model number]",
        "dimming": "[dimming protocol if applicable]",
        "quantity": [number of devices],
        "mounting_type": "[wall, ceiling, etc.]",
        "line_voltage": [boolean]
      },
      "additional_equipment": {
        "[equipment_type]": [quantity]
      }
    }
  ]
}

Important details:
1. Capture relationships between rooms and electrical circuits
2. Document quantities of each fixture type per room
3. Include control systems and switches
4. Note special equipment like floor boxes or furniture feeds
5. Preserve mechanical equipment cross-references
"""

# ======== processors/file_processor.py ========

import json
import os
from document_processor.prompts.prompt_registry import PromptRegistry
from document_processor.utils.llm_client import LLMClient
from document_processor.utils.schema_validator import SchemaValidator

class FileProcessor:
    """Processes construction document files using appropriate prompts"""
    
    def __init__(self, llm_client=None, schema_validator=None):
        """
        Initialize the file processor
        
        Args:
            llm_client: Client for LLM API calls
            schema_validator: Validator for schema compliance
        """
        self.llm_client = llm_client or LLMClient()
        self.schema_validator = schema_validator or SchemaValidator()
    
    def process_file(self, file_path):
        """
        Process a single construction document file
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            dict: Extracted structured data
        """
        # Get the filename
        filename = os.path.basename(file_path)
        
        # Read the file content
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        # Get the appropriate prompt
        prompt, prompt_type = PromptRegistry.get_prompt_for_file(filename)
        
        if not prompt:
            raise ValueError(f"No appropriate prompt found for file: {filename}")
        
        # Call the LLM with the prompt and file content
        llm_response = self.llm_client.extract_data(file_content, prompt)
        
        # Parse the response as JSON
        try:
            extracted_data = json.loads(llm_response)
        except json.JSONDecodeError:
            # Extract JSON from text response if needed
            extracted_data = self._extract_json_from_text(llm_response)
        
        # Validate the extracted data against schema
        if prompt_type:
            self.schema_validator.validate(extracted_data, prompt_type)
        
        return extracted_data
    
    def _extract_json_from_text(self, text):
        """Extract JSON from a text response that might contain other content"""
        # Simple extraction - find the first { and last }
        start = text.find('{')
        end = text.rfind('}')
        
        if start >= 0 and end > start:
            json_str = text[start:end+1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        raise ValueError("Could not extract valid JSON from LLM response")

# ======== utils/llm_client.py ========

import os
import requests
import json

class LLMClient:
    """Client for interacting with the LLM API"""
    
    def __init__(self, api_key=None, api_url=None):
        """
        Initialize the LLM client
        
        Args:
            api_key: API key for the LLM service
            api_url: URL for the LLM API
        """
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        self.api_url = api_url or os.environ.get("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
    
    def extract_data(self, content, prompt):
        """
        Extract structured data from content using a prompt
        
        Args:
            content: The content to extract data from
            prompt: The prompt to use for extraction
            
        Returns:
            str: The LLM response
        """
        messages = [
            {"role": "system", "content": "You are a precise JSON extractor for construction documents."},
            {"role": "user", "content": f"{prompt}\n\nHere is the document to extract from:\n\n{content}"}
        ]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4o-mini",  # Or your preferred model
            "messages": messages,
            "temperature": 0.1,  # Lower temperature for more deterministic responses
            "response_format": {"type": "json_object"}  # Request JSON response
        }
        
        response = requests.post(self.api_url, headers=headers, data=json.dumps(data))
        
        if response.status_code != 200:
            raise Exception(f"LLM API error: {response.status_code} - {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]

# ======== main.py ========

import os
import argparse
import json
from document_processor.processors.file_processor import FileProcessor

def main():
    """Main entry point for the document processor"""
    parser = argparse.ArgumentParser(description='Process construction documents')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--output', '-o', help='Output directory', default='outputs')
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize the processor
    processor = FileProcessor()
    
    # Process a single file or all files in a directory
    if os.path.isfile(args.input):
        process_single_file(args.input, args.output, processor)
    else:
        process_directory(args.input, args.output, processor)

def process_single_file(file_path, output_dir, processor):
    """Process a single file"""
    try:
        print(f"Processing {file_path}...")
        result = processor.process_file(file_path)
        
        # Create the output path
        base_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_processed.json")
        
        # Write the result
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Processed {file_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def process_directory(dir_path, output_dir, processor):
    """Process all JSON files in a directory"""
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, dir_path)
                file_output_dir = os.path.join(output_dir, relative_path)
                
                # Create the output directory
                os.makedirs(file_output_dir, exist_ok=True)
                
                process_single_file(file_path, file_output_dir, processor)

if __name__ == "__main__":
    main()


   Construction Drawing Extraction System Integration
I've designed an implementation that integrates the specialized prompts into a typical document processing pipeline. This approach:

Uses a pattern-matching system to route different file types to appropriate specialized prompts
Maintains prompts as separate modules for easy maintenance
Processes files consistently with validation
Follows good software engineering practices

Key Components of the Implementation
1. Prompt Registry System
The system uses filename patterns to automatically select the correct specialized prompt:
pythonCopy# Example from the implementation
PATTERN_TO_PROMPT = {
    r'(?i)panel|LP-|H\d+\.json|K\d+\.json': 'get_panel_schedule_prompt',
    r'(?i)fixture|lighting|Light': 'get_lighting_fixture_prompt',
    r'(?i)roomdetails|room': 'get_room_electrical_prompt',
    r'(?i)mech|mechanical|HVAC': 'get_mechanical_prompt',
    # ... other patterns
}
2. Modular Prompt Organization
Each trade has its own prompt module, making it easy to update prompts independently:
Copyprompts/
├── __init__.py
├── prompt_registry.py     # Maps file patterns to prompts
├── electrical_prompts.py  # Contains all electrical-related prompts
├── mechanical_prompts.py
├── plumbing_prompts.py
└── architectural_prompts.py
3. Processing Pipeline
The main processor handles the workflow from file input to structured JSON output:
pythonCopydef process_file(self, file_path):
    # Get the filename
    filename = os.path.basename(file_path)
    
    # Read the file content
    with open(file_path, 'r') as f:
        file_content = f.read()
    
    # Get the appropriate prompt
    prompt, prompt_type = PromptRegistry.get_prompt_for_file(filename)
    
    # Call the LLM with the prompt and file content
    llm_response = self.llm_client.extract_data(file_content, prompt)
    
    # Parse and validate the response
    extracted_data = json.loads(llm_response)
    self.schema_validator.validate(extracted_data, prompt_type)
    
    return extracted_data
Running the System
The main entry point provides a simple command-line interface:
bashCopy# Process a single file
python -m document_processor.main path/to/panel.json --output processed_data

# Process an entire directory
python -m document_processor.main path/to/drawing_directory --output processed_data
Next Steps

Add Validation Schemas: Create JSON schemas for each document type to validate extraction results
Improve Error Handling: Add more robust error handling for malformed documents
Add Logging: Implement detailed logging to track processing status
Consider Parallel Processing: For large document sets, add parallel processing capability

This implementation gives you a clean, maintainable way to integrate the specialized prompts, with each prompt focused on a specific document type while maintaining a consistent processing workflow. 