import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, TypeVar, Generic, List
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.performance_utils import time_operation

# Drawing type-specific instructions with main types and subtypes
DRAWING_INSTRUCTIONS = {
    # Main drawing types
    "Electrical": """
    You are an electrical drawing expert extracting structured information. Focus on:
    
    1. All panel schedules - capture complete information about:
       - Panel metadata (name, voltage, phases, rating, location)
       - All circuits with numbers, trip sizes, poles, load descriptions
       - Any panel notes or specifications
    
    2. All equipment schedules with:
       - Complete electrical characteristics (voltage, phase, current ratings)
       - Connection types and mounting specifications
       - Part numbers and manufacturers when available
    
    3. Installation details:
       - Circuit assignments and home run information
       - Mounting heights and special requirements
       - Keyed notes relevant to electrical items
       
    Structure all schedule information into consistent field names (e.g., use 'load_name' for descriptions, 
    'circuit' for circuit numbers, 'trip' for breaker sizes).
    
    IMPORTANT: Ensure ALL circuits, equipment items, and notes are captured in your output. Missing information 
    can cause installation errors.
    """,
    
    "Mechanical": """
    Capture ALL mechanical equipment information with extreme detail and precision:
    
    1. Complete equipment schedules with:
       - Model numbers, manufacturers, and dimensions
       - Capacities (CFM, BTU, tonnage)
       - Electrical requirements (voltage, phase, FLA)
       - Mounting details and clearance requirements
    
    2. HVAC specifications:
       - Airflow volumes and static pressures
       - Control requirements and sequence of operations
       - Ductwork specifications and sizing
    
    3. Installation instructions:
       - Mounting heights and clearances
       - Connection requirements (electrical, plumbing, controls)
       - Testing and balancing requirements
       
    IMPORTANT: ALL mechanical information must be captured with exact values - don't round or approximate.
    Engineers rely on these precise specifications for proper system operation.
    """,
    
    "Plumbing": """
    You are extracting detailed plumbing information. Capture:
    
    1. Complete fixture schedules with:
       - Manufacturer, model, and connection types
       - Flow rates and pressure requirements
       - Mounting details and rough-in dimensions
    
    2. Equipment specifications:
       - Water heaters (capacity, recovery rate, electrical)
       - Pumps (GPM, head pressure, electrical requirements)
       - Special systems (medical gas, vacuum, grease interceptors)
    
    3. System details:
       - Pipe sizing and materials
       - Flow requirements and fixture counts
       - Installation notes and special requirements
       
    IMPORTANT: Capture ALL plumbing elements with their complete specifications. 
    Missing or incomplete information can lead to system failures.
    """,
    
    "Architectural": """
    Extract and structure the following information with PRECISE detail:
    
    1. Room information:
       Create a comprehensive 'rooms' array with objects for EACH room, including:
       - 'number': Room number as string (EXACTLY as shown)
       - 'name': Complete room name
       - 'finish': All ceiling finishes
       - 'height': Ceiling height (with units)
       - 'electrical_info': Any electrical specifications
       - 'architectural_info': Additional architectural details
       - 'wall_types': Wall construction for each wall (north/south/east/west)
    
    2. Complete door and window schedules:
       - Door/window numbers, types, sizes, and materials
       - Hardware specifications and fire ratings
       - Frame types and special requirements
    
    3. Wall type details:
       - Create a 'wall_types' array with complete construction details
       - Include ALL layers, thicknesses, and special requirements
       - Document fire and sound ratings
    
    4. Architectural notes:
       - Capture ALL general notes and keyed notes
       - Include ALL finish schedule information
       
    CRITICAL: EVERY room must be captured. Missing rooms can cause major coordination issues.
    For rooms with minimal information, include what's available and note any missing details.
    """,
    
    "Specifications": """
    PRESERVE THE COMPLETE CONTENT of each specification section with its EXACT structure and details.
    
    For each specification section:
    
    1. Create objects in the 'specifications' array with:
       - 'section_title': The complete section number and title (e.g., "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS")
       - 'content': The FULL TEXT of the entire section, including ALL parts, subsections, and items
    
    2. Maintain the exact hierarchical structure:
       - Preserve all SECTION > PART > SUBSECTION relationships
       - Keep all numbering and lettering of items (1., 1.1, A., etc.)
       - Include all indentation and formatting
    
    3. Include EVERY paragraph, table, list, and requirement:
       - Do not summarize or truncate any content
       - Maintain all technical details, values, and specifications
       - Preserve all notes, warnings, and special instructions
    
    CRITICAL: The entire text must be preserved exactly as written. Even minor omissions or changes 
    can alter contract requirements and cause legal issues.
    """,
    
    "General": """
    Extract ALL relevant content and organize into a comprehensive, structured JSON:
    
    1. Identify the document type and organize data accordingly:
       - For schedules: Create arrays of consistently structured objects
       - For specifications: Preserve the complete text with hierarchical structure
       - For drawings: Document all annotations, dimensions, and references
    
    2. Capture EVERY piece of information:
       - Include ALL notes, annotations, and references
       - Document ALL equipment, fixtures, and components
       - Preserve ALL technical specifications and requirements
    
    3. Maintain relationships between elements:
       - Link components to their locations (rooms, areas)
       - Connect items to their technical specifications
       - Reference related notes and details
    
    Structure everything into a clear, consistent JSON format that preserves ALL the original information.
    """,
    
    # Electrical subtypes
    "Electrical_PanelSchedule": """
    You are an expert electrical engineer analyzing panel schedules. Your goal is to produce a precisely structured JSON representation with COMPLETE information.
    
    Extract and structure the following information with PERFECT accuracy:
    
    1. Panel metadata (create a 'panel' object):
       - 'name': Panel name/ID (EXACTLY as written)
       - 'location': Physical location of panel
       - 'voltage': Full voltage specification (e.g., "120/208V Wye", "277/480V")
       - 'phases': Number of phases (1 or 3) and wires (e.g., "3 Phase 4 Wire")
       - 'amperage': Main amperage rating
       - 'main_breaker': Main breaker size if present
       - 'aic_rating': AIC/interrupting rating
       - 'feed': Source information (fed from)
       - Any additional metadata present (enclosure type, mounting, etc.)
       
    2. Circuit information (create a 'circuits' array with objects for EACH circuit):
       - 'circuit': Circuit number or range EXACTLY as shown (e.g., "1", "2-4-6", "3-5")
       - 'trip': Breaker/trip size with units (e.g., "20A", "70 A")
       - 'poles': Number of poles (1, 2, or 3)
       - 'load_name': Complete description of the connected load
       - 'equipment_ref': Reference to equipment ID if available
       - 'room_id': Connected room(s) if specified
       - Any additional circuit information present
       
    3. Additional information:
       - 'panel_totals': Connected load, demand factors, and calculated loads
       - 'notes': Any notes specific to the panel
       
    CRITICAL: EVERY circuit must be documented EXACTLY as shown on the schedule. Missing circuits, incorrect numbering, or incomplete information can cause dangerous installation errors.
    """,
    
    "Electrical_Lighting": """
    You are extracting complete lighting fixture and control information.
    
    1. Create a comprehensive 'lighting_fixtures' array with details for EACH fixture:
       - 'type_mark': Fixture type designation (EXACTLY as shown)
       - 'description': Complete fixture description
       - 'manufacturer': Manufacturer name
       - 'product_number': Model or catalog number
       - 'lamp_type': Complete lamp specification (e.g., "LED, 35W, 3500K")
       - 'mounting': Mounting type and height
       - 'voltage': Operating voltage
       - 'wattage': Power consumption
       - 'dimensions': Complete fixture dimensions
       - 'count': Quantity of fixtures when specified
       - 'location': Installation locations
       - 'dimmable': Dimming capability and type
       - 'remarks': Any special notes or requirements
       
    2. Document all lighting controls with specific details:
       - Switch types and functions
       - Sensors (occupancy, vacancy, daylight)
       - Dimming systems and protocols
       - Control zones and relationships
       
    3. Document circuit assignments:
       - Panel and circuit numbers
       - Connected areas and zones
       - Load calculations
       
    IMPORTANT: Capture EVERY fixture type and ALL specifications. Missing or incorrect information
    can lead to incompatible installations and lighting failure.
    """,
    
    "Electrical_Power": """
    Extract ALL power distribution and equipment connection information with complete detail:
    
    1. Document all outlets and receptacles in an organized array:
       - Type (standard, GFCI, special purpose, isolated ground)
       - Voltage and amperage ratings
       - Mounting height and orientation
       - Circuit assignment (panel and circuit number)
       - Room location and mounting surface
       - NEMA configuration
       
    2. Create a structured array of equipment connections:
       - Equipment type and designation
       - Power requirements (voltage, phase, amperage)
       - Connection method (hardwired, cord-and-plug)
       - Circuit assignment
       - Disconnecting means
       
    3. Detail specialized power systems:
       - UPS connections and specifications
       - Emergency or standby power
       - Isolated power systems
       - Specialty voltage requirements
       
    4. Document all keyed notes related to power:
       - Special installation requirements
       - Code compliance notes
       - Coordination requirements
       
    IMPORTANT: ALL power elements must be captured with their EXACT specifications.
    Electrical inspectors will verify these details during installation.
    """,
    
    "Electrical_FireAlarm": """
    Extract complete fire alarm system information with precise detail:
    
    1. Document ALL devices in a structured array:
       - Device type (smoke detector, heat detector, pull station, etc.)
       - Model number and manufacturer
       - Mounting height and location
       - Zone/circuit assignment
       - Addressable or conventional designation
       
    2. Identify all control equipment:
       - Fire alarm control panel specifications
       - Power supplies and battery calculations
       - Remote annunciators
       - Auxiliary control functions
       
    3. Capture ALL wiring specifications:
       - Circuit types (initiating, notification, signaling)
       - Wire types, sizes, and ratings
       - Survivability requirements
       
    4. Document interface requirements:
       - Sprinkler system monitoring
       - Elevator recall functions
       - HVAC shutdown
       - Door holder/closer release
       
    CRITICAL: Fire alarm systems are life-safety systems subject to strict code enforcement.
    ALL components and functions must be documented exactly as specified to ensure proper operation.
    """,
    
    "Electrical_Technology": """
    Extract ALL low voltage systems information with complete technical detail:
    
    1. Document data/telecom infrastructure in structured arrays:
       - Outlet types and locations
       - Cable specifications (category, shielding)
       - Mounting heights and orientations
       - Pathway types and sizes
       - Equipment rooms, racks, and cabinets
       
    2. Identify security systems with specific details:
       - Camera types, models, and coverage areas
       - Access control devices and door hardware
       - Intrusion detection sensors and zones
       - Control equipment and monitoring requirements
       
    3. Document audiovisual systems:
       - Display types, sizes, and mounting details
       - Audio equipment and speaker layout
       - Control systems and interfaces
       - Signal routing and processing
       
    4. Capture specialty systems:
       - Nurse call or emergency communication
       - Distributed antenna systems (DAS)
       - Paging and intercom
       - Radio and wireless systems
       
    IMPORTANT: Technology systems require precise documentation to ensure proper integration.
    ALL components, connections, and configurations must be captured as specified.
    """,
    
    # Architectural subtypes
    "Architectural_FloorPlan": """
    Extract COMPLETE floor plan information with precise room-by-room detail:
    
    1. Create a comprehensive 'rooms' array with objects for EACH room, capturing:
       - 'number': Room number EXACTLY as shown (including prefixes/suffixes)
       - 'name': Complete room name
       - 'dimensions': Length, width, and area when available
       - 'adjacent_rooms': List of connecting room numbers
       - 'wall_types': Wall construction for each room boundary (north/south/east/west)
       - 'door_numbers': Door numbers providing access to the room
       - 'window_numbers': Window numbers in the room
       
    2. Document circulation paths with specific details:
       - Corridor widths and clearances
       - Stair dimensions and configurations
       - Elevator locations and sizes
       - Exit paths and egress requirements
       
    3. Identify area designations and zoning:
       - Fire-rated separations and occupancy boundaries
       - Smoke compartments
       - Security zones
       - Department or functional areas
       
    CRITICAL: EVERY room must be documented with ALL available information.
    Missing rooms or incomplete details can cause serious coordination issues across all disciplines.
    When room information is unclear or incomplete, note this in the output.
    """,
    
    "Architectural_ReflectedCeiling": """
    Extract ALL ceiling information with complete room-by-room detail:
    
    1. Create a comprehensive 'rooms' array with ceiling-specific objects for EACH room:
       - 'number': Room number EXACTLY as shown
       - 'name': Complete room name
       - 'ceiling_type': Material and system (e.g., "2x2 ACT", "GWB")
       - 'ceiling_height': Height above finished floor (with units)
       - 'soffit_heights': Heights of any soffits or bulkheads
       - 'slope': Ceiling slope information if applicable
       - 'fixtures': Array of ceiling-mounted elements (lights, diffusers, sprinklers)
       
    2. Document ceiling transitions with specific details:
       - Height changes between areas
       - Bulkhead and soffit dimensions
       - Special ceiling features (clouds, islands)
       
    3. Identify ceiling-mounted elements:
       - Lighting fixtures (coordinated with electrical)
       - HVAC diffusers and registers
       - Sprinkler heads and fire alarm devices
       - Specialty items (projector mounts, speakers)
       
    IMPORTANT: Ceiling coordination is critical for clash detection.
    EVERY room must have complete ceiling information to prevent conflicts with mechanical, 
    electrical, and plumbing systems during installation.
    """,
    
    "Architectural_Partition": """
    Extract ALL wall and partition information with precise construction details:
    
    1. Create a comprehensive 'wall_types' array with objects for EACH type:
       - 'type': Wall type designation EXACTLY as shown
       - 'description': Complete description of the wall assembly
       - 'details': Object containing:
         - 'stud_type': Material and thickness (steel, wood)
         - 'stud_width': Dimension with units
         - 'stud_spacing': Spacing with units
         - 'layers': Complete description of all layers from exterior to interior
         - 'insulation': Type and R-value
         - 'total_thickness': Overall dimension with units
       - 'fire_rating': Fire resistance rating with duration
       - 'sound_rating': STC or other acoustic rating
       - 'height': Height designation ("to deck", "above ceiling")
       
    2. Document room-to-wall type relationships:
       - For each room, identify wall types used on each boundary (north/south/east/west)
       - Note any special conditions or variations
       
    3. Identify special wall conditions:
       - Seismic considerations
       - Expansion/control joints
       - Bracing requirements
       - Wall transitions
       
    CRITICAL: Wall type details impact all disciplines (architectural, structural, mechanical, electrical).
    EVERY wall type must be fully documented with ALL construction details to ensure proper installation.
    """,
    
    "Architectural_Details": """
    Extract ALL architectural details with complete construction information:
    
    1. Document each architectural detail:
       - Detail number and reference
       - Complete description of the assembly
       - Materials and dimensions
       - Connection methods and fastening
       - Finish requirements
       
    2. Capture specific assembly information:
       - Waterproofing and flashing details
       - Thermal and moisture protection
       - Acoustic treatments
       - Fire and smoke barriers
       
    3. Document all annotations and notes:
       - Construction requirements
       - Installation sequence
       - Quality standards
       - Reference standards
       
    IMPORTANT: Architectural details provide critical information for proper construction.
    ALL detail information must be captured exactly as specified to ensure code compliance
    and proper installation.
    """,
    
    "Architectural_Schedules": """
    Extract ALL architectural schedules with complete information for each element:
    
    1. Door schedules with comprehensive detail:
       - Door number EXACTLY as shown
       - Type, size (width, height, thickness)
       - Material and finish
       - Fire rating and label requirements
       - Frame type and material
       - Hardware sets and special requirements
       
    2. Window schedules with specific details:
       - Window number and type
       - Dimensions and configuration
       - Glazing type and performance ratings
       - Frame material and finish
       - Operating requirements
       
    3. Room finish schedules:
       - Room number and name
       - Floor, base, wall, and ceiling finishes
       - Special treatments or requirements
       - Finish transitions
       
    4. Accessory and equipment schedules:
       - Item designations and types
       - Mounting heights and locations
       - Material and finish specifications
       - Quantities and installation notes
       
    CRITICAL: Schedule information is used by multiple trades and disciplines.
    EVERY scheduled item must be completely documented with ALL specifications to ensure
    proper procurement and installation.
    """
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
        # Panel schedules - check for these first as they're most specific
        if any(term in file_name_lower for term in ["panel", "schedule", "panelboard", "circuit", "h1", "l1", "k1", "k1s", "21lp-1", "20h-1"]):
            return "Electrical_PanelSchedule"
        # Lighting fixtures and controls
        elif any(term in file_name_lower for term in ["light", "lighting", "fixture", "lamp", "luminaire", "rcp", "ceiling"]):
            return "Electrical_Lighting"
        # Power distribution
        elif any(term in file_name_lower for term in ["power", "outlet", "receptacle", "equipment", "connect", "riser", "metering"]):
            return "Electrical_Power"
        # Fire alarm systems
        elif any(term in file_name_lower for term in ["fire", "alarm", "fa", "detection", "smoke", "emergency", "evacuation"]):
            return "Electrical_FireAlarm"
        # Low voltage systems
        elif any(term in file_name_lower for term in ["tech", "data", "comm", "security", "av", "low voltage", "telecom", "network"]):
            return "Electrical_Technology"
    
    # Architectural subtypes
    elif drawing_type == "Architectural":
        # Reflected ceiling plans
        if any(term in file_name_lower for term in ["rcp", "ceiling", "reflected"]):
            return "Architectural_ReflectedCeiling"
        # Wall types and partitions
        elif any(term in file_name_lower for term in ["partition", "wall type", "wall-type", "wall", "room wall"]):
            return "Architectural_Partition"
        # Floor plans
        elif any(term in file_name_lower for term in ["floor", "plan", "layout", "room"]):
            return "Architectural_FloorPlan"
        # Door and window schedules
        elif any(term in file_name_lower for term in ["door", "window", "hardware", "schedule"]):
            return "Architectural_Schedules"
        # Architectural details
        elif any(term in file_name_lower for term in ["detail", "section", "elevation", "assembly"]):
            return "Architectural_Details"
    
    # Specification documents
    elif "specification" in drawing_type.lower() or "spec" in file_name_lower:
        return "Specifications"
    
    # If no subtype detected, return the main type
    return drawing_type

class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini"  # Updated to remove date-specific version
    GPT_4O = "gpt-4o"  # Updated to remove date-specific version

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
    
    # Default parameters
    params = {
        "model_type": ModelType.GPT_4O_MINI,
        "temperature": 0.1,  # Reduced default temperature for more consistent output
        "max_tokens": 16000,
    }
    
    # For very long content or complex documents, use more powerful model
    if content_length > 50000 or "specification" in drawing_type.lower():
        params["model_type"] = ModelType.GPT_4O
        # Calculate max_tokens dynamically based on content length
        # Estimate token count as roughly chars/4 for English text
        estimated_input_tokens = min(128000, len(raw_content) // 4)  # Cap at 128k tokens maximum
        # Reserve at least 8000 tokens for output, but don't exceed model context limits
        params["max_tokens"] = max(8000, min(14000, 32000 - estimated_input_tokens))
    
    # Adjust based on drawing type
    if "Electrical" in drawing_type:
        # Panel schedules need lowest temperature for precision
        if "PanelSchedule" in drawing_type:
            params["temperature"] = 0.05
            # Use more capable model for complex panel schedules
            if content_length > 15000:
                params["model_type"] = ModelType.GPT_4O
        # Other electrical drawings need precision too
        else:
            params["temperature"] = 0.1
    
    elif "Architectural" in drawing_type:
        # Room information needs precision but some inference
        if "FloorPlan" in drawing_type or "ReflectedCeiling" in drawing_type:
            params["temperature"] = 0.1
            if content_length > 20000:
                params["model_type"] = ModelType.GPT_4O
    
    # For specifications, use more powerful model and lower temperature
    if "SPECIFICATION" in file_name.upper() or "Specifications" in drawing_type:
        logging.info(f"Processing specification document ({content_length} chars)")
        params["temperature"] = 0.05
        params["model_type"] = ModelType.GPT_4O
        # Use dynamic calculation for specifications as well
        estimated_input_tokens = min(128000, len(raw_content) // 4)
        params["max_tokens"] = max(8000, min(14000, 32000 - estimated_input_tokens))
    
    # Safety check: ensure max_tokens is within reasonable limits
    params["max_tokens"] = min(params["max_tokens"], 16000)
    
    logging.info(f"Using model {params['model_type'].value} with temperature {params['temperature']} and max_tokens {params['max_tokens']}")
    
    return params

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
        # Try to get instructions for the specific subtype first
        drawing_instruction = self.drawing_instructions.get(
            drawing_type,
            # If subtype not found and it contains an underscore, try the main type
            self.drawing_instructions.get(
                drawing_type.split('_')[0] if '_' in drawing_type else "",
                # Fall back to general instructions
                self.drawing_instructions.get("General", "")
            )
        )
        
        return f"""
        You are a construction drawing expert tasked with extracting complete, detailed information from the provided content.
        Your job is to structure this information into a comprehensive, well-organized JSON object with these sections:
        
        - 'metadata': Include drawing number, title, date, revision, and any other identifying information.
        - 'schedules': Array of schedules with type and data. Use consistent field names for similar data across different schedules.
        - 'notes': Array of notes and annotations found in the drawing.
        - 'specifications': Array of specification sections with 'section_title' and 'content' fields.
        - 'rooms': For architectural drawings, include an array of rooms with detailed information.
        - Additional sections as appropriate for the specific drawing type.
        
        {drawing_instruction}
        
        CRITICAL REQUIREMENTS:
        1. Extract ALL information from the drawing - nothing should be omitted
        2. Structure data consistently with descriptive field names
        3. For unclear or ambiguous information, include it with a note about the uncertainty
        4. Ensure the output is valid JSON - no malformed structures or syntax errors
        5. For architectural drawings, ALWAYS include a 'rooms' array when room information is present
        
        Accurate and complete extraction is essential - construction decisions will be based on this data.
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
        system_message: Optional[str] = None,
        example_output: Optional[str] = None
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
            example_output: Optional example output for few-shot learning
            
        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing {drawing_type} drawing with {len(raw_content)} characters")
            
            messages = [
                {"role": "system", "content": system_message or self._get_default_system_message(drawing_type)}
            ]
            
            # Add example output if provided (few-shot learning)
            if example_output:
                messages.append({"role": "user", "content": "Please process this drawing content and convert it to structured JSON:"})
                messages.append({"role": "assistant", "content": example_output})
                messages.append({"role": "user", "content": "Now process this new drawing content in the same format:"})
            
            # Add the actual content to process
            messages.append({"role": "user", "content": raw_content})
            
            response = await self.client.chat.completions.create(
                model=model_type.value,
                messages=messages,
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
        system_message: Optional[str] = None,
        example_output: Optional[str] = None
    ) -> str:
        """
        Process a drawing using a specific prompt, ensuring full content is sent to the API.
        
        Args:
            raw_content: Raw content from the drawing
            temperature: Temperature parameter for the AI model
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            system_message: Optional custom system message to override default
            example_output: Optional example output for few-shot learning
        
        Returns:
            Processed content as a JSON string

        Raises:
            JSONDecodeError: If the response is not valid JSON
            ValueError: If the JSON structure is invalid
            Exception: For other processing errors
        """
        default_system_message = """
        You are an AI assistant specialized in construction drawings. Extract all relevant information from the provided content and organize it into a structured JSON object with these sections:
        
        - "metadata": An object containing drawing metadata such as "drawing_number", "title", "date", and "revision".
        Include any available information; if a field is missing, omit it.
        
        - "schedules": An array of schedule objects. Each schedule should have a "type" (e.g., "electrical_panel",
        "mechanical") and a "data" array containing objects with standardized field names. For panel schedules,
        use consistent field names like "circuit" for circuit numbers, "trip" for breaker sizes, 
        "load_name" for equipment descriptions, and "poles" for the number of poles.
        
        - "notes": An array of strings containing any notes or instructions found in the drawing.
        
        - "specifications": An array of objects, each with a "section_title" and "content" for specification sections.
        
        - "rooms": For architectural drawings, include an array of rooms with 'number', 'name', 'finish', 'height',
        'electrical_info', and 'architectural_info'.
        
        CRITICAL REQUIREMENTS:
        1. The JSON output MUST include ALL information from the drawing - nothing should be omitted
        2. Structure data consistently with descriptive field names
        3. Panel schedules MUST include EVERY circuit, with correct circuit numbers, trip sizes, and descriptions
        4. For architectural drawings, ALWAYS include a 'rooms' array with ALL rooms
        5. For specifications, preserve the COMPLETE text in the 'content' field
        6. Ensure the output is valid JSON with no syntax errors
        
        Construction decisions will be based on this data, so accuracy and completeness are essential.
        """

        # Use the provided system message or fall back to default
        final_system_message = system_message if system_message else default_system_message
        
        content_length = len(raw_content)
        self.logger.info(f"Processing content of length {content_length} with model {model_type.value}")

        # Check if content is too large and log a warning
        if content_length > 250000 and model_type == ModelType.GPT_4O_MINI:
            self.logger.warning(f"Content length ({content_length} chars) may exceed GPT-4o-mini context window. Switching to GPT-4o.")
            model_type = ModelType.GPT_4O
        
        if content_length > 500000:
            self.logger.warning(f"Content length ({content_length} chars) exceeds GPT-4o context window. Processing may be incomplete.")

        try:
            messages = [
                {"role": "system", "content": final_system_message}
            ]
            
            # Add example output if provided (few-shot learning)
            if example_output:
                messages.append({"role": "user", "content": "Please process this drawing content and convert it to structured JSON:"})
                messages.append({"role": "assistant", "content": example_output})
                messages.append({"role": "user", "content": "Now process this new drawing content in the same format:"})
            
            # Add the actual content to process
            messages.append({"role": "user", "content": raw_content})
            
            # Calculate rough token estimate for logging
            estimated_tokens = content_length // 4
            self.logger.info(f"Estimated input tokens: ~{estimated_tokens}")
            
            try:
                response = await self.client.chat.completions.create(
                    model=model_type.value,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}  # Ensure JSON response
                )
                content = response.choices[0].message.content
                
                # Process usage information if available
                if hasattr(response, 'usage') and response.usage:
                    self.logger.info(f"Token usage - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
                
                try:
                    # Validate JSON parsing
                    parsed_content = json.loads(content)
                    if not self.validate_json(parsed_content):
                        self.logger.warning("JSON validation failed - missing required keys")
                        # Still return the content, as it might be usable even with missing keys
                    
                    return content
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decoding error: {str(e)}")
                    self.logger.error(f"Raw content received: {content[:500]}...")  # Log the first 500 chars for debugging
                    raise
            except Exception as e:
                if "maximum context length" in str(e).lower() or "token limit" in str(e).lower():
                    self.logger.error(f"Context length exceeded: {str(e)}")
                    raise ValueError(f"Content too large for model context window: {str(e)}")
                else:
                    self.logger.error(f"API error: {str(e)}")
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
        # Basic validation - check for required top-level keys
        required_keys = ["metadata", "schedules", "notes"]
        
        # Specifications validation - check structure and convert if needed
        if "specifications" in json_data:
            specs = json_data["specifications"]
            if isinstance(specs, list) and specs:
                # Convert string arrays to object arrays if needed
                if isinstance(specs[0], str):
                    self.logger.warning("Converting specifications from string array to object array")
                    json_data["specifications"] = [{"section_title": spec, "content": ""} for spec in specs]
        else:
            # Add empty specifications array if missing
            json_data["specifications"] = []
            
        # For architectural drawings, check for rooms array
        if "metadata" in json_data and "drawing_type" in json_data["metadata"]:
            if "architectural" in json_data["metadata"]["drawing_type"].lower() and "rooms" not in json_data:
                self.logger.warning("Architectural drawing missing 'rooms' array")
                return False
                
        return all(key in json_data for key in required_keys)

    async def get_example_output(self, drawing_type: str) -> Optional[str]:
        """
        Retrieve an example output for the given drawing type from a library of examples.
        This enables few-shot learning for better consistency.
        
        Args:
            drawing_type: Type of drawing
            
        Returns:
            Example output as JSON string, or None if no example is available
        """
        # This would typically load from a database or file system
        # For now, we'll return None as a placeholder
        return None

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
        
    Raises:
        ValueError: If the content is too large for processing
        JSONDecodeError: If the response is not valid JSON
        Exception: For other processing errors
    """
    if not raw_content:
        logging.warning(f"Empty content received for {file_name}. Cannot process.")
        raise ValueError("Cannot process empty content")
        
    # Log details about processing task
    content_length = len(raw_content)
    drawing_type = drawing_type or "Unknown"
    file_name = file_name or "Unknown"
    
    logging.info(f"Starting drawing processing: Type={drawing_type}, File={file_name}, Content length={content_length}")
    
    try:
        # Detect more specific drawing subtype
        subtype = detect_drawing_subtype(drawing_type, file_name)
        logging.info(f"Detected drawing subtype: {subtype}")
        
        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
        
        # Get optimized parameters for this drawing
        params = optimize_model_parameters(subtype, raw_content, file_name)
        
        # Try to get an example output for few-shot learning
        example_output = await ai_service.get_example_output(subtype)
        
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
            
            {DRAWING_INSTRUCTIONS.get("Specifications")}
            
            Ensure the output is valid JSON.
            """
        else:
            # Get the appropriate system message based on detected subtype
            type_instruction = DRAWING_INSTRUCTIONS.get(subtype, DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"]))
            
            system_message = f"""
            You are processing a construction drawing of type: {subtype}
            
            Extract ALL relevant information and organize it into a comprehensive JSON object with the following sections:
            - 'metadata': Object containing drawing number, title, date, and any other identifying information
            - 'schedules': Array of schedules with type and data using consistent field names
            - 'notes': Array of all notes and annotations
            - 'specifications': Array of specification sections with 'section_title' and 'content'
            - Other sections as appropriate for this drawing type
            
            {type_instruction}
            
            CRITICAL: Your output MUST include ALL information from the drawing - nothing should be omitted.
            Use consistent field names for similar data (e.g., "circuit", "trip", "load_name" for panel schedules).
            For architectural drawings, ALWAYS include a 'rooms' array with ALL rooms.
            
            Engineers, contractors, and installers will rely on this data for construction decisions.
            Accuracy and completeness are essential to prevent costly mistakes and safety issues.
            """
        
        # Process the drawing using the most appropriate method
        try:
            response = await ai_service.process_with_prompt(
                raw_content=raw_content,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                model_type=params["model_type"],
                system_message=system_message,
                example_output=example_output
            )
            
            # Validate JSON structure
            try:
                parsed = json.loads(response)
                logging.info(f"Successfully processed {subtype} drawing ({len(response)} chars output)")
                return response
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response from AI service for {file_name}")
                raise
                
        except ValueError as e:
            if "content too large" in str(e).lower():
                logging.error(f"Content too large for {file_name}: {str(e)}")
                raise ValueError(f"Drawing content exceeds model context limits: {str(e)}")
            else:
                logging.error(f"Value error processing {file_name}: {str(e)}")
                raise
                
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing '{file_name}': {str(e)}")
        raise

@time_operation("ai_processing")
async def process_drawing_with_examples(raw_content: str, drawing_type: str, client, file_name: str = "", example_outputs: List[str] = None) -> str:
    """
    Process a drawing using few-shot learning with example outputs.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing
        client: OpenAI client
        file_name: Optional name of the file being processed
        example_outputs: List of example JSON outputs for few-shot learning
        
    Returns:
        Structured JSON as a string
        
    Raises:
        ValueError: If the content is too large for processing
        JSONDecodeError: If the response is not valid JSON
        Exception: For other processing errors
    """
    if not raw_content:
        logging.warning(f"Empty content received for {file_name}. Cannot process.")
        raise ValueError("Cannot process empty content")
    
    # Log details about the processing task
    content_length = len(raw_content)
    drawing_type = drawing_type or "Unknown"
    file_name = file_name or "Unknown"
    
    logging.info(f"Starting drawing processing with examples: Type={drawing_type}, File={file_name}, Content length={content_length}")
    logging.info(f"Number of example outputs provided: {len(example_outputs) if example_outputs else 0}")
    
    try:
        # Detect more specific drawing subtype
        subtype = detect_drawing_subtype(drawing_type, file_name)
        logging.info(f"Detected drawing subtype: {subtype}")
        
        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
        
        # Get optimized parameters for this drawing
        params = optimize_model_parameters(subtype, raw_content, file_name)
        
        # Check if this is a specification document
        is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"
        
        # Base system message with type-specific instructions
        type_instruction = DRAWING_INSTRUCTIONS.get(subtype, DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"]))
        
        system_message = f"""
        You are a construction drawing expert tasked with extracting ALL information from {subtype} drawings.
        
        Your job is to convert raw drawing content into structured JSON following the exact format shown in the examples.
        Pay close attention to field names and structure used in the examples - your output should match this format.
        
        {type_instruction}
        
        CRITICAL REQUIREMENTS:
        1. Extract EVERY piece of information from the drawing - nothing should be omitted
        2. Use the EXACT same field names and structure as shown in the examples
        3. Include ALL circuits in panel schedules, ALL rooms in architectural drawings, etc.
        4. When in doubt about a field name, check the examples first
        5. Ensure your output is valid JSON with no syntax errors
        
        Construction decisions will be based on this data, so accuracy and completeness are essential.
        """
        
        # Process using the examples-based approach
        try:
            if example_outputs and len(example_outputs) > 0:
                # Use first example only as some APIs have token limits
                example = example_outputs[0]
                logging.info("Processing with example-based learning")
                response = await ai_service.process_with_prompt(
                    raw_content=raw_content,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    model_type=params["model_type"],
                    system_message=system_message,
                    example_output=example
                )
            else:
                # No examples provided, fall back to standard processing
                logging.info("No examples provided, falling back to standard processing")
                response = await process_drawing(raw_content, drawing_type, client, file_name)
            
            # Validate JSON structure
            try:
                parsed = json.loads(response)
                logging.info(f"Successfully processed {subtype} drawing with examples ({len(response)} chars output)")
                return response
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response from AI service for {file_name}")
                raise
        
        except ValueError as e:
            if "content too large" in str(e).lower():
                logging.error(f"Content too large for {file_name}: {str(e)}")
                raise ValueError(f"Drawing content exceeds model context limits: {str(e)}")
            else:
                logging.error(f"Value error processing {file_name}: {str(e)}")
                raise
        
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing '{file_name}' with examples: {str(e)}")
        raise