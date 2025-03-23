Here's a prompt you can use in that other project:

I need you to help me update my AI service code based on the JSON output files I've already processed from previous jobs. These files in the project knowledge represent successful outputs that I want future processing to be consistent with.
Here's the simplified code for my ai_service.py that I want to update:
pythonCopy# This is from the "Simplified Drawing Processor Implementation Plan"
#import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, TypeVar, Generic
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.performance_utils import time_operation

# Drawing type-specific instructions with main types and subtypes
DRAWING_INSTRUCTIONS = {
    # Main drawing types
    "Electrical": "Focus on panel schedules, circuit info, equipment schedules with electrical characteristics, and installation notes.",
    
    "Mechanical": "Capture equipment schedules, HVAC details (CFM, capacities), and installation instructions.",
    
    "Plumbing": "Include fixture schedules, pump details, water heater specs, pipe sizing, and system instructions.",
    
    "Architectural": """
    Extract and structure the following information:
    1. Room details: Create a 'rooms' array with objects for each room, including:
       - 'number': Room number (as a string)
       - 'name': Room name
       - 'finish': Ceiling finish
       - 'height': Ceiling height
       - 'electrical_info': Any electrical specifications
       - 'architectural_info': Any additional architectural details
    2. Room finish schedules
    3. Door/window details
    4. Wall types
    5. Architectural notes
    Ensure all rooms are captured and properly structured in the JSON output.
    """,
    
    "Specifications": """
    IMPORTANT: Preserve the FULL CONTENT of each specification section, not just the headers.
    For each specification section:
    1. Create objects in the 'specifications' array with:
       - 'section_title': The section number and title (e.g., "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS")
       - 'content': The COMPLETE text content of the section, including all parts, subsections, and items
    2. Maintain the hierarchical structure (SECTION > PART > SUBSECTION)
    3. Preserve all numbered and lettered items
    4. Include all paragraphs, tables, and detailed requirements
    Do not summarize or truncate the content - include the entire text of each section.
    """,
    
    "General": "Organize all relevant data into logical categories based on content type.",
    
    # Electrical subtypes
    "Electrical_PanelSchedule": """
    You are an expert electrical engineer analyzing panel schedules.
    Extract and structure the following information:
    
    1. Panel metadata:
       - Panel name/ID
       - Voltage (e.g., 120/208V, 277/480V)
       - Phases (1Φ or 3Φ)
       - Amperage rating
       - Main breaker size
       - Fed from (source panel)
       
    2. Circuit information (create a 'circuits' array):
       - Circuit number
       - Trip/breaker size (amperage)
       - Number of poles (1P, 2P, 3P)
       - Load name/description
       - Amp load (if available)
       - Connected rooms/areas
       
    Structure everything into a clear, valid JSON format with proper nesting.
    Be precise and complete, including all circuits and their details.
    """,
    
    "Electrical_Lighting": """
    Focus on lighting fixtures, controls, and circuits:
    
    1. Create a 'fixtures' array with:
       - Fixture type/ID
       - Manufacturer and model (if available)
       - Lamp/bulb specifications
       - Mounting type
       
    2. Identify lighting control systems:
       - Switch locations and types
       - Dimming systems
       - Occupancy/vacancy sensors
       - Daylight sensors
       
    3. Document circuit assignments:
       - Panel and circuit numbers
       - Load calculations
       
    4. Map fixtures to rooms where possible
    
    Ensure complete coverage of all lighting elements in the drawing.
    """,
    
    "Electrical_Power": """
    Focus on power distribution and equipment connections:
    
    1. Document all outlets and receptacles:
       - Type (standard, GFCI, special purpose)
       - Mounting height
       - Circuit assignment
       
    2. Identify equipment connections:
       - Equipment type
       - Power requirements
       - Connection method
       
    3. Detail any special power systems:
       - UPS or emergency power
       - Dedicated circuits
       - Special voltage requirements
       
    4. Map power elements to rooms where possible
    
    Be thorough in identifying all power elements in the drawing.
    """,
    
    "Electrical_FireAlarm": """
    Extract all fire alarm system components and connections:
    
    1. Document all devices with:
       - Device type (smoke detector, pull station, etc.)
       - Location and mounting details
       - Zone/circuit assignment
       
    2. Identify control equipment:
       - Fire alarm control panel details
       - Remote annunciators
       - Power supplies
       
    3. Capture wiring and circuit information:
       - Circuit types (initiating, notification, etc.)
       - Wiring specifications
       
    4. Note any special requirements or instructions
    
    Ensure comprehensive documentation of the fire alarm system.
    """,
    
    "Electrical_Technology": """
    Focus on low voltage systems including data, security, and AV:
    
    1. Document data/telecom infrastructure:
       - Outlet locations and types
       - Cable pathways
       - Equipment rooms and racks
       
    2. Identify security systems:
       - Camera locations
       - Card readers and access control
       - Intrusion detection
       
    3. Capture audiovisual systems:
       - Equipment locations
       - Connection details
       - Control systems
       
    4. Map technology elements to rooms where possible
    
    Be detailed in documenting all low voltage/technology systems.
    """,
    
    # Architectural subtypes
    "Architectural_FloorPlan": """
    Focus on room layout and relationships:
    
    1. Create a comprehensive 'rooms' array with:
       - Room number and name
       - Dimensions and area
       - Adjacent rooms and connections
       - Door and window placements
       
    2. Identify circulation paths:
       - Corridors and hallways
       - Stairs and elevators
       - Entry/exit points
       
    3. Document wall types and partitions
    
    4. Note any area designations or zones
    
    Ensure all spaces are accounted for and properly related to each other.
    """,
    
    "Architectural_ReflectedCeiling": """
    Focus on ceiling elements and systems:
    
    1. Create a 'rooms' array with ceiling-specific details:
       - Room number and name
       - Ceiling height(s)
       - Ceiling material/type
       - Special ceiling features
       
    2. Document ceiling-mounted elements:
       - Lighting fixture locations (coordinate with electrical)
       - HVAC diffusers and grilles
       - Sprinklers and fire alarm devices
       
    3. Identify ceiling transitions:
       - Soffits and bulkheads
       - Height changes
       
    Be thorough in documenting all ceiling conditions.
    """,
    
    "Architectural_Partition": """
    Focus on wall types and partition details:
    
    1. Create a 'wall_types' array with:
       - Type identifier (e.g., "A", "B", etc.)
       - Construction details (materials, thickness)
       - Fire rating (if applicable)
       - Acoustic rating (if applicable)
       - Height specifications
       
    2. Document special wall conditions:
       - Soffits and bulkheads
       - Partial-height partitions
       - Expansion joints
       
    3. Note finish information where available
    
    Be precise in documenting all wall and partition elements.
    """,
}

def detect_drawing_subtype(drawing_type: str, file_name: str) -> str:
    """
    Detect more specific drawing subtype based on drawing type and filename.
    
    Args:
        drawing_type: Main drawing type (Electrical, Architectural, etc.)
        file_name: Name of the file being processed
        
    Returns:
        More specific subtype or the original drawing type if no subtype detected
    """
    if not file_name or not drawing_type:
        return drawing_type
    
    file_name_lower = file_name.lower()
    
    # Electrical subtypes
    if drawing_type == "Electrical":
        # Order is important here - check more specific patterns first
        if any(term in file_name_lower for term in ["light", "lighting", "fixture"]):
            return "Electrical_Lighting"
        elif any(term in file_name_lower for term in ["panel", "schedule", "panelboard"]):
            return "Electrical_PanelSchedule"
        elif any(term in file_name_lower for term in ["power", "outlet", "receptacle"]):
            return "Electrical_Power"
        elif any(term in file_name_lower for term in ["fire", "alarm", "fa", "detection"]):
            return "Electrical_FireAlarm"
        elif any(term in file_name_lower for term in ["tech", "data", "comm", "security", "av", "low voltage"]):
            return "Electrical_Technology"
    
    # Architectural subtypes
    elif drawing_type == "Architectural":
        if any(term in file_name_lower for term in ["rcp", "ceiling", "reflected"]):
            return "Architectural_ReflectedCeiling"
        elif any(term in file_name_lower for term in ["partition", "wall type", "wall-type"]):
            return "Architectural_Partition"
        elif any(term in file_name_lower for term in ["floor", "plan", "layout"]):
            return "Architectural_FloorPlan"
    
    # Add more drawing type detections as needed
    
    # If no subtype detected, return the main type
    return drawing_type

def optimize_model_parameters(
    drawing_type: str,
    raw_content: str,
    file_name: str
) -> Dict[str, Any]:
    """
    Determine optimal model parameters based on drawing type and content.
    
    Args:
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        raw_content: Raw content from the drawing
        file_name: Name of the file being processed
        
    Returns:
        Dictionary of optimized parameters
    """
    content_length = len(raw_content)
    
    # Default parameters using a model with a large context window
    params = {
        "model_type": ModelType.GPT_4O_MINI,
        "temperature": 0.2,
        "max_tokens": 16000,
    }
    
    # Adjust based on drawing type
    if "Electrical" in drawing_type:
        # Electrical drawings need more precision
        params["temperature"] = 0.1
        
        # Panel schedules need even lower temperature
        if "PanelSchedule" in drawing_type:
            params["temperature"] = 0.05
    
    elif "Architectural" in drawing_type:
        if "ReflectedCeiling" in drawing_type:
            params["temperature"] = 0.15
    
    # For specifications, use lower temperature for more deterministic results
    if "SPECIFICATION" in file_name.upper() or "Specifications" in drawing_type:
        logging.info(f"Processing specification document ({content_length} chars)")
        params["temperature"] = 0.1
        # Increase max_tokens for specifications to ensure full content preservation
        params["max_tokens"] = 32000
    
    logging.info(f"Using model {params['model_type'].value} with temperature {params['temperature']} and max_tokens {params['max_tokens']}")
    
    return params

class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"

T = TypeVar('T')

class AiRequest:
    """
    Class to hold AI API request parameters.
    """
    def __init__(
        self,
        content: str,
        model_type: 'ModelType' = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
        system_message: str = ""
    ):
        """
        Initialize an AiRequest.
        
        Args:
            content: Content to send to the API
            model_type: Model type to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            system_message: System message to use
        """
        self.content = content
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message

class AiResponse(Generic[T]):
    """
    Generic class to hold AI API response data or errors.
    """
    def __init__(self, success: bool = True, content: str = "", parsed_content: Optional[T] = None, error: str = ""):
        """
        Initialize an AiResponse.
        
        Args:
            success: Whether the API call was successful
            content: Raw content from the API
            parsed_content: Optional parsed content (of generic type T)
            error: Error message if the call failed
        """
        self.success = success
        self.content = content
        self.parsed_content = parsed_content
        self.error = error

class DrawingAiService:
    """
    Specialized AI service for processing construction drawings.
    """
    def __init__(self, client: AsyncOpenAI, drawing_instructions: Dict[str, str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the DrawingAiService.

        Args:
            client: AsyncOpenAI client instance
            drawing_instructions: Optional dictionary of drawing type-specific instructions
            logger: Optional logger instance
        """
        self.client = client
        self.drawing_instructions = drawing_instructions or DRAWING_INSTRUCTIONS
        self.logger = logger or logging.getLogger(__name__)

    def _get_default_system_message(self, drawing_type: str) -> str:
        """
        Get the default system message for the given drawing type.
        
        Args:
            drawing_type: Type of drawing (Architectural, Electrical, etc.) or subtype
            
        Returns:
            System message string
        """
        drawing_instruction = ""
        
        # Try to get instructions for the specific subtype first
        if hasattr(self, 'drawing_instructions') and drawing_type in self.drawing_instructions:
            drawing_instruction = self.drawing_instructions[drawing_type]
        # If subtype not found and it contains an underscore, try the main type
        elif '_' in drawing_type and drawing_type.split('_')[0] in self.drawing_instructions:
            main_type = drawing_type.split('_')[0]
            drawing_instruction = self.drawing_instructions[main_type]
        # Fall back to general instructions
        else:
            drawing_instruction = self.drawing_instructions.get("General", "")
        
        return f"""
        You are processing a construction drawing. Extract all relevant information and organize it into a JSON object with the following sections:
        - 'metadata': Include drawing number, title, date, etc.
        - 'schedules': Array of schedules with type and data.
        - 'notes': Array of notes.
        - 'specifications': Array of specification sections.
        - 'rooms': For architectural drawings, include an array of rooms with 'number', 'name', 'electrical_info', and 'architectural_info'.
        
        {drawing_instruction}
        
        IMPORTANT: For architectural drawings, ALWAYS include a 'rooms' array, even if you have to infer room information from context.
        Ensure the output is valid JSON.
        """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process(self, request: AiRequest) -> AiResponse[Dict[str, Any]]:
        """
        Process an AI request.
        
        Args:
            request: AiRequest object containing parameters
            
        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing content of length {len(request.content)}")
            
            response = await self.client.chat.completions.create(
                model=request.model_type.value,
                messages=[
                    {"role": "system", "content": request.system_message},
                    {"role": "user", "content": request.content}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...")  # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error processing drawing: {str(e)}")
            return AiResponse(success=False, error=str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process_drawing_with_responses(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI,
        system_message: Optional[str] = None
    ) -> AiResponse:
        """
        Process a drawing using the OpenAI API.
        
        Args:
            raw_content: Complete raw content from the drawing - NO TRUNCATION
            drawing_type: Type of drawing
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            model_type: Model type to use
            system_message: Optional system message
            
        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing {drawing_type} drawing with {len(raw_content)} characters")
            
            response = await self.client.chat.completions.create(
                model=model_type.value,
                messages=[
                    {"role": "system", "content": system_message or self._get_default_system_message(drawing_type)},
                    {"role": "user", "content": raw_content}  # Send the FULL content
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...")  # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error processing drawing: {str(e)}")
            return AiResponse(success=False, error=str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process_with_prompt(
        self,
        raw_content: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI,
        system_message: Optional[str] = None
    ) -> str:
        """
        Process a drawing using a specific prompt, ensuring full content is sent to the API.
        
        Args:
            raw_content: Raw content from the drawing
            temperature: Temperature parameter for the AI model
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            system_message: Optional custom system message to override default
        
        Returns:
            Processed content as a JSON string

        Raises:
            JSONDecodeError: If the response is not valid JSON
            ValueError: If the JSON structure is invalid
            Exception: For other processing errors
        """
        default_system_message = """
        You are an AI assistant tasked with processing construction drawings. Your job is to extract all relevant information from the provided content and organize it into a structured JSON object with the following sections:
        
        - "metadata": An object containing drawing metadata such as "drawing_number", "title", "date", and "revision". Include any available information; if a field is missing, omit it.
        
        - "schedules": An array of schedule objects. Each schedule should have a "type" (e.g., "electrical_panel", "mechanical") and a "data" array containing objects with standardized field names. For example, use "load_name" for fields like "Load" or "Load Type", and "trip" for "Breaker Size" or "Amperage". If a field doesn't match a standard name, include it as-is.
        
        - "notes": An array of strings containing any notes or instructions found in the drawing.
        
        - "specifications": An array of objects, each with a "section_title" and "content" for specification sections.
        
        - "rooms": For architectural drawings, include an array of rooms with 'number', 'name', 'electrical_info', and 'architectural_info'.
        
        IMPORTANT: For architectural drawings, ALWAYS include a 'rooms' array, even if you have to infer room information from context.
        
        Ensure that the entire response is a single, valid JSON object. Do not include any additional text or explanations outside of the JSON.
        """

        # Use the provided system message or fall back to default
        final_system_message = system_message if system_message else default_system_message
        
        self.logger.info(f"Processing content of length {len(raw_content)} with model {model_type.value}")

        try:
            response = await self.client.chat.completions.create(
                model=model_type.value,
                messages=[
                    {"role": "system", "content": final_system_message},
                    {"role": "user", "content": raw_content}  # Send FULL content
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            content = response.choices[0].message.content
            try:
                parsed_content = json.loads(content)
                if not self.validate_json(parsed_content):
                    raise ValueError("Invalid JSON structure: missing required keys")
                return content
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...")  # Log the first 500 chars for debugging
                raise
        except Exception as e:
            self.logger.error(f"Error processing drawing: {str(e)}")
            raise

    def validate_json(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate the JSON structure.

        Args:
            json_data: Parsed JSON data

        Returns:
            True if the JSON has all required keys, False otherwise
        """
        required_keys = ["metadata", "schedules", "notes", "specifications"]
        
        # For specifications, ensure specifications is an array of objects with section_title and content
        if "specifications" in json_data:
            specs = json_data["specifications"]
            # Check if specifications is an array of strings (old format) or objects (new format)
            if isinstance(specs, list) and len(specs) > 0:
                if isinstance(specs[0], str):
                    # Converting old format to new format
                    self.logger.warning("Converting specifications from string array to object array")
                    json_data["specifications"] = [{"section_title": spec, "content": ""} for spec in specs]
        
        return all(key in json_data for key in required_keys)

@time_operation("ai_processing")
async def process_drawing(raw_content: str, drawing_type: str, client, file_name: str = "") -> str:
    """
    Use GPT to parse PDF text and table data into structured JSON based on the drawing type.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        client: OpenAI client
        file_name: Optional name of the file being processed
        
    Returns:
        Structured JSON as a string
    """
    try:
        # Detect more specific drawing subtype
        subtype = detect_drawing_subtype(drawing_type, file_name)
        
        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
        
        # Get optimized parameters for this drawing
        params = optimize_model_parameters(subtype, raw_content, file_name)
        
        logging.info(f"Processing {subtype} drawing with {len(raw_content)} characters")
        
        # Check if this is a specification document
        is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"
        
        # Enhanced system message with different emphasis based on document type
        if is_specification:
            system_message = f"""
            You are processing a SPECIFICATION document. Extract all relevant information and organize it into a JSON object.
            
            CRITICAL INSTRUCTIONS:
            - In the 'specifications' array, create objects with 'section_title' and 'content' fields
            - The 'section_title' should contain the section number and name (e.g., "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS")
            - The 'content' field MUST contain the COMPLETE TEXT of each section, including all parts, subsections, and detailed requirements
            - Preserve the hierarchical structure (SECTION > PART > SUBSECTION)
            - Include all numbered and lettered items, paragraphs, tables, and detailed requirements
            - Do not summarize or truncate - include the ENTIRE text of each section
            
            {DRAWING_INSTRUCTIONS.get("Specifications", DRAWING_INSTRUCTIONS["General"])}
            
            Ensure the output is valid JSON.
            """
        else:
            # Get the appropriate system message - fall back to main type if subtype not found
            type_instruction = DRAWING_INSTRUCTIONS.get(subtype, DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"]))
            
            system_message = f"""
            You are processing a construction drawing. Extract all relevant information and organize it into a JSON object with the following sections:
            - 'metadata': Include drawing number, title, date, etc.
            - 'schedules': Array of schedules with type and data.
            - 'notes': Array of notes.
            - 'specifications': Array of specification sections.
            - 'rooms': For architectural drawings, include an array of rooms with 'number', 'name', 'electrical_info', and 'architectural_info'.
            
            {type_instruction}
            
            IMPORTANT: For architectural drawings, ALWAYS include a 'rooms' array, even if you have to infer room information from context.
            Ensure the output is valid JSON.
            """
        
        # Process the drawing using the Responses API (falls back to standard if needed)
        response: AiResponse[Dict[str, Any]] = await ai_service.process_drawing_with_responses(
            raw_content=raw_content,  # Send the complete content without modifications
            drawing_type=subtype,  # Use the detected subtype
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            model_type=params["model_type"],
            system_message=system_message  # Pass the enhanced system message
        )
        
        if response.success:
            return response.content
        else:
            logging.error(f"Error processing {subtype} drawing: {response.error}")
            raise Exception(f"Error processing {subtype} drawing: {response.error}")
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing: {str(e)}")
        raise

@time_operation("ai_processing")
async def process_drawing_simple(raw_content: str, drawing_type: str, client, file_name: str = "") -> str:
    """
    Process a drawing using a simple, universal prompt - similar to pasting into ChatGPT UI.
    Requires fewer specific instructions but relies on the model's understanding of construction drawings.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (for logging and minimal customization)
        client: OpenAI client
        file_name: Optional name of the file being processed
        
    Returns:
        Structured JSON as a string
    """
    try:
        # Detect more specific drawing subtype
        subtype = detect_drawing_subtype(drawing_type, file_name)
        
        # Log processing attempt
        logging.info(f"Processing {subtype} drawing with {len(raw_content)} characters using simplified approach")
        
        # Create minimal customization based on drawing subtype
        drawing_hint = ""
        
        if "Architectural" in subtype:
            drawing_hint = " Include room information where available."
            if "ReflectedCeiling" in subtype:
                drawing_hint = " Pay special attention to ceiling heights, materials, and fixtures mounted on the ceiling."
            elif "FloorPlan" in subtype:
                drawing_hint = " Capture all rooms, their dimensions, and relationships to adjacent spaces."
            elif "Partition" in subtype:
                drawing_hint = " Detail all wall types, their construction, and ratings."
                
        elif "Electrical" in subtype:
            if "PanelSchedule" in subtype:
                drawing_hint = " Extract all panel schedule data including circuits, breaker sizes, loads, and phases."
            elif "Lighting" in subtype:
                drawing_hint = " Identify lighting fixtures, controls, and circuit assignments."
            elif "Power" in subtype:
                drawing_hint = " Document all power outlets, equipment connections, and circuit assignments."
            elif "FireAlarm" in subtype:
                drawing_hint = " Catalog all fire alarm devices, their locations, and wiring."
            elif "Technology" in subtype:
                drawing_hint = " Detail all low voltage systems including data, security, and AV."
            else:
                drawing_hint = " Pay attention to panel schedules and circuit information."
                
        elif "Specifications" in subtype:
            drawing_hint = " Preserve ALL specification text content completely."
        
        # Get optimized parameters based on the subtype
        params = optimize_model_parameters(subtype, raw_content, file_name)
        
        # Single, universal prompt
        system_message = f"""
        Structure this construction drawing content into well-organized JSON.{drawing_hint}
        Include all relevant information from the document and preserve the relationships between elements.
        Ensure your response is ONLY valid JSON with no additional text.
        """
        
        # Make the API call
        response = await client.chat.completions.create(
            model=params["model_type"].value,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_content}
            ],
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error in simplified processing of {drawing_type} drawing: {str(e)}")
        raise 


Please:

Review the JSON files in the project knowledge that represent previous processed outputs
Identify patterns in these files, especially for electrical and architectural drawings
Update the DRAWING_INSTRUCTIONS dictionary to better match these patterns
Add any additional drawing types or subtypes you see in the existing files
Keep the code structure simple and maintain the approach of the original code
Pay special attention to field naming variations in panel schedules and other drawing types
Update the detect_drawing_subtype function based on the filenames in the project knowledge

Important: Keep the implementation simple, relying more on the AI model's intelligence through clear instructions rather than complex code structures. I want to handle field naming variations (like "load" vs "load type" vs "equipment type") through smart prompting rather than complex mapping logic.