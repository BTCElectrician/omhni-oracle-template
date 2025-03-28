# File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/ai_service.py
import json
import logging
import re
import os
from enum import Enum
from typing import Dict, Any, Optional, TypeVar, Generic, List
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.performance_utils import time_operation
from dotenv import load_dotenv
from templates.prompt_types import (
    DrawingCategory,
    ArchitecturalSubtype,
    ElectricalSubtype,
    MechanicalSubtype,
    PlumbingSubtype
)
from templates.prompt_templates import get_prompt_template

# Initialize logger at module level
logger = logging.getLogger(__name__)

# Drawing type-specific instructions with main types and subtypes
# (DRAWING_INSTRUCTIONS dictionary remains the same - keeping it for brevity)
DRAWING_INSTRUCTIONS = {
    # ... (Keep the existing dictionary content) ...
    "Electrical": """
    You are an electrical drawing expert extracting structured information. Focus on:
    
    1. CRITICAL: Extract all metadata from the drawing's title block including:
       - drawing_number: The drawing number identifier 
       - title: The drawing title/description
       - revision: Revision number or letter
       - date: Drawing date
       - job_number: Project/job number
       - project_name: Full project name
    
    2. All panel schedules - capture complete information about:
       - Panel metadata (name, voltage, phases, rating, location)
       - All circuits with numbers, trip sizes, poles, load descriptions
       - Any panel notes or specifications
    
    3. All equipment schedules with:
       - Complete electrical characteristics (voltage, phase, current ratings)
       - Connection types and mounting specifications
       - Part numbers and manufacturers when available
    
    4. Installation details:
       - Circuit assignments and home run information
       - Mounting heights and special requirements
       - Keyed notes relevant to electrical items
       
    Structure all schedule information into consistent field names (e.g., use 'load_name' for descriptions, 
    'circuit' for circuit numbers, 'trip' for breaker sizes).
    
    IMPORTANT: Ensure ALL circuits, equipment items, and notes are captured in your output. Missing information 
    can cause installation errors.
    """,
    
    "Electrical_LIGHTING": """
    You are an expert in electrical lighting analyzing a lighting drawing or fixture schedule.
    
    CRITICAL: Extract all metadata from the drawing's title block, including:
    - drawing_number (e.g., "E1.00")
    - title (e.g., "LIGHTING - FLOOR LEVEL")
    - revision (e.g., "3")
    - date (e.g., "08/15/2024")
    - job_number (e.g., "30J7925")
    - project_name (e.g., "ELECTRIC SHUFFLE")
    
    Capture ALL lighting fixtures with these details:
    - type_mark: The fixture type identifier
    - count: Quantity of this fixture type
    - manufacturer: Fixture manufacturer name
    - product_number: Product/model number
    - description: Complete fixture description
    - finish: Material finish
    - lamp_type: Lamp specification with wattage and color temp
    - mounting: Mounting method
    - dimensions: Physical dimensions with units
    - location: Installation location
    - wattage: Power consumption
    - ballast_type: Driver/ballast type
    - dimmable: Whether fixture is dimmable
    - remarks: Any special notes
    - catalog_series: Full catalog reference
    
    Also document all lighting zones and controls:
    - zone: Zone identifier
    - area: Area served
    - circuit: Circuit number
    - fixture_type: Type of fixture
    - dimming_control: Control type
    - notes: Special conditions
    - quantities_or_linear_footage: Installation quantity
    
    Structure into a clear, consistent JSON format with metadata at the top level:
    {
      "ELECTRICAL": {
        "metadata": {
          "drawing_number": "E1.00",
          "title": "LIGHTING - FLOOR LEVEL",
          "revision": "3",
          "date": "08/15/2024", 
          "job_number": "30J7925",
          "project_name": "ELECTRIC SHUFFLE"
        },
        "LIGHTING_FIXTURE": [...],
        "LIGHTING_ZONE": [...]
      }
    }
    
    Lighting design coordination requires COMPLETE accuracy in fixture specifications.
    Missing or incorrect information can cause ordering errors and installation conflicts.
    """,
    
    "Mechanical": """
    Extract ALL mechanical information with a simplified, comprehensive structure.

1. Create a straightforward JSON structure with these main categories:
   - "equipment": Object containing arrays of ALL mechanical equipment grouped by type
   - "systems": Information about ductwork, piping, and distribution systems
   - "notes": ALL notes, specifications, and requirements
   - "remarks": ALL remarks and numbered references

2. For ANY type of equipment (air handlers, fans, VAVs, pumps, etc.):
   - Group by equipment type using descriptive keys (airHandlers, exhaustFans, chillers, etc.)
   - Include EVERY specification field with its EXACT value - never round or approximate
   - Use camelCase field names based on original headers
   - Always include identification (tag/ID), manufacturer, model, and capacity information
   - Capture ALL performance data (CFM, tonnage, BTU, static pressure, etc.)
   - Include ALL electrical characteristics (voltage, phase, FLA, MCA, etc.)

3. For ALL mechanical information:
   - Preserve EXACT values - never round or approximate
   - Include units of measurement
   - Keep the structure flat and simple
   - Don't skip ANY information shown on the drawing

Example simplified structure:
{
  "equipment": {
    "airHandlingUnits": [
      {
        "id": "AHU-1",
        "manufacturer": "Trane",
        "model": "M-Series",
        "cfm": "10,000",
        // ALL other fields exactly as shown
      }
    ],
    "exhaustFans": [
      // ALL fan data with EVERY field
    ]
  },
  "notes": [
    // ALL notes and specifications
  ],
  "remarks": [
    // ALL remarks and references
  ]
}

CRITICAL: Engineers need EVERY mechanical element and specification value EXACTLY as shown - complete accuracy is essential for proper system design, ordering, and installation.
    """,
    
    "Plumbing": """
    You are an expert AI assistant extracting detailed information from plumbing drawings, schedules, and notes. Your goal is to create a comprehensive and structured JSON output containing ALL relevant information presented.

    Analyze the provided text, which may include various schedules (fixtures, water heaters, pumps, valves, etc.), legends, and general notes. Structure your response into a single JSON object with the following top-level keys:

    1.  `metadata`: (Object) Capture any project identifiers, drawing numbers, titles, dates, or revisions found.
    2.  `fixture_schedule`: (Array of Objects) Extract details for EVERY item listed in the main plumbing fixture schedule(s). Include items like sinks (S1, S2, S3, HS, MS), drains (FD, FS, HD), cleanouts (WCO, FCO, CO), lavatories (SW-05), urinals (SW-03), water closets (SW-01), trap guards (TG), shock arrestors (SA), backflow preventers (DCBP), etc. For each item, include:
        - `fixture_id`: The exact mark or identifier (e.g., "S1", "SW-05", "WCO").
        - `description`: The full description provided.
        - `manufacturer`: Manufacturer name, if available.
        - `model`: Model number, if available.
        - `mounting`: Mounting details.
        - `connections`: (Object) Use the 'Connection Schedule' table to populate waste, vent, cold water (CW), and hot water (HW) sizes where applicable.
        - `notes`: Any specific notes related to this fixture.
    3.  `water_heater_schedule`: (Array of Objects) Extract details for EACH water heater (e.g., WH-1, WH-2). Include:
        - `mark`: The exact identifier (e.g., "WH-1").
        - `location`: Installation location.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `specifications`: (Object) Capture ALL technical specs like storage_gallons, operating_water_temp, tank_dimensions, recovery_rate, electric_power, kW_input, etc.
        - `mounting`: Mounting details (e.g., "Floor mounted").
        - `notes`: (Array of Strings) Capture ALL general notes associated specifically with the water heater schedule.
    4.  `pump_schedule`: (Array of Objects) Extract details for EACH pump (e.g., CP). Include:
        - `mark`: The exact identifier (e.g., "CP").
        - `location`: Installation location.
        - `serves`: What the pump serves.
        - `type`: Pump type (e.g., "IN-LINE").
        - `gpm`: Gallons Per Minute.
        - `tdh_ft`: Total Dynamic Head (in feet).
        - `hp`: Horsepower.
        - `rpm`: Max RPM.
        - `electrical`: Volts/Phase/Cycle.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `notes`: Any remarks or specific notes.
    5.  `mixing_valve_schedule`: (Array of Objects) Extract details for EACH thermostatic mixing valve (e.g., TM). Include:
        - `designation`: Identifier (e.g., "TM").
        - `location`: Service location.
        - `inlet_temp_F`: Hot water inlet temperature.
        - `outlet_temp_F`: Blended water temperature.
        - `pressure_drop_psi`: Pressure drop.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `notes`: Full description or notes.
    6.  `shock_absorber_schedule`: (Array of Objects) Extract details for EACH shock arrestor size listed (e.g., SA-A, SA-B,... SA-F, plus the general SA). Include:
        - `mark`: The exact identifier (e.g., "SA-A", "SA").
        - `fixture_units`: Applicable fixture units range.
        - `manufacturer`: Manufacturer name.
        - `model`: Model number.
        - `description`: Full description if provided separately.
    7.  `material_legend`: (Object) Capture the pipe material specifications (e.g., "SANITARY SEWER PIPING": "CAST IRON OR SCHEDULE 40 PVC").
    8.  `general_notes`: (Array of Strings) Extract ALL numbered or lettered general notes found in the text (like notes A-T).
    9.  `insulation_notes`: (Array of Strings) Extract ALL notes specifically related to plumbing insulation (like notes A-F).
    10. `symbols`: (Array of Objects, Optional) If needed, extract symbol descriptions.
    11. `abbreviations`: (Array of Objects, Optional) If needed, extract abbreviation definitions.

    CRITICAL:
    - Capture ALL items listed in EVERY schedule table or list. Do not omit any fixtures, equipment, or sizes.
    - Extract ALL general notes and insulation notes sections completely.
    - Preserve the exact details, model numbers, specifications, and text provided.
    - Ensure your entire response is a single, valid JSON object adhering to this structure. Missing information can lead to system failures or installation errors.
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
Extract specification content using a clean, direct structure.

1. Create a straightforward 'specifications' array containing objects with:
   - 'section_title': EXACT section number and title (e.g., "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS")
   - 'content': COMPLETE text of the section with ALL parts and subsections
   
2. For the 'content' field:
   - Preserve the EXACT text - no summarizing or paraphrasing
   - Maintain ALL hierarchical structure (PART > SECTION > SUBSECTION)
   - Keep ALL numbering and lettering (1.1, A., etc.)
   - Include ALL paragraphs, tables, lists, and requirements

3. DO NOT add interpretations, summaries, or analysis
   - Your ONLY task is to preserve the original text in the correct sections
   - The structure should be simple and flat (just title + content for each section)
   - Handle each section as a complete unit

Example structure:
{
  "specifications": [
    {
      "section_title": "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS",
      "content": "PART 1 - GENERAL\\n\\n1.1 RELATED DOCUMENTS\\n\\nA. DRAWINGS AND GENERAL PROVISIONS...\\n\\n[COMPLETE TEXT HERE]"
    },
    {
      "section_title": "SECTION 16123 - BUILDING WIRE AND CABLE",
      "content": "PART 1 GENERAL\\n\\n1.01 SECTION INCLUDES\\n\\nA. WIRE AND CABLE...\\n\\n[COMPLETE TEXT HERE]"
    }
  ]
}

CRITICAL: Construction decisions rely on complete, unaltered specifications. Even minor omissions or changes can cause legal and safety issues.
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

    # Enhanced specification detection - check this first for efficiency
    if "specification" in drawing_type.lower() or any(term in file_name_lower for term in
                                                       ["spec", "specification", ".spec", "e0.01"]):
        return DrawingCategory.SPECIFICATIONS.value

    # Electrical subtypes
    if drawing_type == DrawingCategory.ELECTRICAL.value:
        # Look for stronger panel schedule indicators using regex
        panel_indicators_regex = [
            r"panel", r"schedule", r"panelboard", r"circuit",
            r"breaker", r"distribution", r"single line", r"riser", r"one line", # Added riser/one-line
            # Panel naming patterns (simple examples, case-insensitive matching)
            r"^[a-z][0-9]+[a-z]?$",                  # Matches names like h1, k1s, l1a at start of filename
            r"[a-z][0-9]+[a-z]?(-| panel| schedule)", # Matches names like h1-panel, k1s schedule anywhere
            r"^[0-9]{1,2}[a-z]{1,2}(p|h)?-[0-9]+$",   # Matches names like 21lp-1, 20h-1 at start
            # Add more specific project patterns here if known, e.g. r"^lp-.*"
        ]
        # Match against the base name without extension, case-insensitive
        base_name_lower = os.path.splitext(os.path.basename(file_name))[0].lower()
        is_panel_schedule = False
        for pattern in panel_indicators_regex:
            # Try searching anywhere in the full filename first
            if re.search(pattern, file_name.lower()):
                 is_panel_schedule = True
                 logging.debug(f"Panel indicator '{pattern}' found in filename '{file_name}'")
                 break
            # Try matching specific patterns (like ^...$) against the base name
            # Adjust regex if they shouldn't be start/end anchored (remove ^ and $)
            if pattern.startswith('^') and re.match(pattern, base_name_lower):
                 is_panel_schedule = True
                 logging.debug(f"Panel pattern '{pattern}' matched base name '{base_name_lower}'")
                 break

        if is_panel_schedule:
            logging.info(f"Detected PANEL_SCHEDULE subtype for '{file_name}'")
            return f"{drawing_type}_{ElectricalSubtype.PANEL_SCHEDULE.value}"
        # Lighting fixtures and controls
        elif any(term in file_name_lower for term in ["light", "lighting", "fixture", "lamp", "luminaire", "rcp", "ceiling"]):
            return f"{drawing_type}_{ElectricalSubtype.LIGHTING.value}"
        # Power distribution
        elif any(term in file_name_lower for term in ["power", "outlet", "receptacle", "equipment", "connect", "riser", "metering"]):
            return f"{drawing_type}_{ElectricalSubtype.POWER.value}"
        # Fire alarm systems
        elif any(term in file_name_lower for term in ["fire", "alarm", "fa", "detection", "smoke", "emergency", "evacuation"]):
            return f"{drawing_type}_{ElectricalSubtype.FIREALARM.value}"
        # Low voltage systems
        elif any(term in file_name_lower for term in ["tech", "data", "comm", "security", "av", "low voltage", "telecom", "network"]):
            return f"{drawing_type}_{ElectricalSubtype.TECHNOLOGY.value}"
        # Specifications (if not caught earlier)
        elif any(term in file_name_lower for term in ["spec", "specification", "requirement"]):
             return DrawingCategory.SPECIFICATIONS.value # Map directly to main spec type

    # Architectural subtypes
    elif drawing_type == DrawingCategory.ARCHITECTURAL.value:
        # Reflected ceiling plans
        if any(term in file_name_lower for term in ["rcp", "ceiling", "reflected"]):
            return f"{drawing_type}_{ArchitecturalSubtype.CEILING.value}"
        # Wall types and partitions
        elif any(term in file_name_lower for term in ["partition", "wall type", "wall-type", "wall", "room wall"]):
            return f"{drawing_type}_{ArchitecturalSubtype.WALL.value}"
        # Floor plans
        elif any(term in file_name_lower for term in ["floor", "plan", "layout", "room"]):
            return f"{drawing_type}_{ArchitecturalSubtype.ROOM.value}"
        # Door and window schedules
        elif any(term in file_name_lower for term in ["door", "window", "hardware", "schedule"]):
            return f"{drawing_type}_{ArchitecturalSubtype.DOOR.value}"
        # Architectural details
        elif any(term in file_name_lower for term in ["detail", "section", "elevation", "assembly"]):
            return f"{drawing_type}_{ArchitecturalSubtype.DETAIL.value}"

    # Mechanical subtypes
    elif drawing_type == DrawingCategory.MECHANICAL.value:
        # Equipment schedules
        if any(term in file_name_lower for term in ["equip", "unit", "ahu", "rtu", "vav", "schedule"]):
            return f"{drawing_type}_{MechanicalSubtype.EQUIPMENT.value}"
        # Ventilation systems
        elif any(term in file_name_lower for term in ["vent", "air", "supply", "return", "diffuser", "grille"]):
            return f"{drawing_type}_{MechanicalSubtype.VENTILATION.value}"
        # Piping systems
        elif any(term in file_name_lower for term in ["pipe", "chilled", "heating", "cooling", "refrigerant"]):
            return f"{drawing_type}_{MechanicalSubtype.PIPING.value}"

    # Plumbing subtypes
    elif drawing_type == DrawingCategory.PLUMBING.value:
        # Fixture schedules
        if any(term in file_name_lower for term in ["fixture", "sink", "toilet", "shower", "schedule"]):
            return f"{drawing_type}_{PlumbingSubtype.FIXTURE.value}"
        # Equipment
        elif any(term in file_name_lower for term in ["equip", "heater", "pump", "water", "schedule"]):
            return f"{drawing_type}_{PlumbingSubtype.EQUIPMENT.value}"
        # Piping systems
        elif any(term in file_name_lower for term in ["pipe", "riser", "water", "sanitary", "vent"]):
            return f"{drawing_type}_{PlumbingSubtype.PIPE.value}"

    # If no subtype detected, return the main type
    return drawing_type

class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"

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
    from dotenv import load_dotenv
    load_dotenv(override=True) # Reload to ensure we get the latest env values

    from config.settings import get_force_mini_model # Import the function instead

    content_length = len(raw_content)

    # Default parameters
    params = {
        "model_type": ModelType.GPT_4O_MINI,
        "temperature": 0.1, # Reduced default temperature for more consistent output
        "max_tokens": 16000, # Default max tokens for mini model
    }

    # Determine the appropriate model based on FORCE_MINI_MODEL flag and content length/type
    use_mini_model = get_force_mini_model()
    use_large_model = False

    # Conditions to potentially upgrade to the larger model (GPT-4O)
    if not use_mini_model:
        if content_length > 50000 or "specification" in drawing_type.lower():
            use_large_model = True
            logging.info(f"Content length ({content_length}) or type ({drawing_type}) suggests using GPT-4o for {file_name}")
        elif ("Electrical" in drawing_type and "PanelSchedule" in drawing_type and content_length > 15000):
             use_large_model = True
             logging.info(f"Complex panel schedule ({content_length}) suggests using GPT-4o for {file_name}")
        elif ("Architectural" in drawing_type and content_length > 20000):
             use_large_model = True
             logging.info(f"Large architectural drawing ({content_length}) suggests using GPT-4o for {file_name}")
        elif ("Mechanical" in drawing_type and content_length > 20000):
             use_large_model = True
             logging.info(f"Large mechanical drawing ({content_length}) suggests using GPT-4o for {file_name}")

    # Set the model type
    if use_large_model:
        params["model_type"] = ModelType.GPT_4O
        # Adjust max_tokens for the larger model
        estimated_input_tokens = min(128000, len(raw_content) // 4) # Cap at 128k tokens maximum
        # Reserve tokens for output, adjust based on model context window (approx 128k input, 4k output generally safe)
        params["max_tokens"] = max(4096, min(16000, 128000 - estimated_input_tokens - 1000)) # Ensure at least 4k, max 16k, within context
    elif use_mini_model:
         logging.info(f"Forcing gpt-4o-mini model for testing: {file_name}")
         # Keep default params["model_type"] = ModelType.GPT_4O_MINI
         # Keep default params["max_tokens"] = 16000

    # Adjust temperature based on drawing type
    if "Electrical" in drawing_type:
        # Panel schedules need lowest temperature for precision
        if "PanelSchedule" in drawing_type:
            params["temperature"] = 0.05
        # Other electrical drawings need precision too
        else:
            params["temperature"] = 0.1
    elif "Architectural" in drawing_type:
         # Requires precision but might need some inference for relationships
         params["temperature"] = 0.1
    elif "Mechanical" in drawing_type:
         # Slightly higher temperature might help with varied schedule formats
         params["temperature"] = 0.2 # Reduced from 0.3 for better consistency
    elif "Specification" in drawing_type:
        # Needs precision, very low temperature
        params["temperature"] = 0.05

    # Safety check: ensure max_tokens is within reasonable limits for the chosen model
    if params["model_type"] == ModelType.GPT_4O_MINI:
        params["max_tokens"] = min(params["max_tokens"], 16000) # Hard cap for mini
    elif params["model_type"] == ModelType.GPT_4O:
         # Let the calculated value stand, but ensure it doesn't exceed common practical limits like 16k unless necessary
         params["max_tokens"] = min(params["max_tokens"], 16000) # Re-cap at 16k for safety/cost unless specifically needed higher


    logging.info(f"Using model {params['model_type'].value} with temperature {params['temperature']} and max_tokens {params['max_tokens']} for {file_name}")

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
        Get the default system message for the given drawing type using the prompt templates.

        Args:
            drawing_type: Type of drawing (Architectural, Electrical, etc.) or subtype

        Returns:
            System message string
        """
        # Use the prompt template module to get the appropriate template
        return get_prompt_template(drawing_type)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process(self, request: AiRequest) -> AiResponse[Dict[str, Any]]:
        """
        Process an AI request. (Used by panel schedule processor primarily)

        Args:
            request: AiRequest object containing parameters

        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing content of length {len(request.content)} using model {request.model_type.value}")

            response = await self.client.chat.completions.create(
                model=request.model_type.value,
                messages=[
                    {"role": "system", "content": request.system_message},
                    {"role": "user", "content": request.content}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format={"type": "json_object"} # Ensure JSON response
            )

            content = response.choices[0].message.content

            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...") # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}", content=content) # Return raw content on error
        except Exception as e:
            self.logger.error(f"Error during AI processing: {str(e)}")
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
    ) -> AiResponse:
        """
        Process a drawing using the OpenAI API, returning an AiResponse.
        (Removed few-shot example logic)

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
            self.logger.info(f"Processing {drawing_type} drawing with {len(raw_content)} characters using model {model_type.value}")

            messages = [
                {"role": "system", "content": system_message or self._get_default_system_message(drawing_type)},
                {"role": "user", "content": raw_content}
            ]

            response = await self.client.chat.completions.create(
                model=model_type.value,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"} # Ensure JSON response
            )

            content = response.choices[0].message.content

            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...") # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}", content=content) # Return raw content on error
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
    ) -> str:
        """
        Process raw content using a specific prompt, ensuring full content is sent to the API.
        Returns the raw JSON string response.
        (Removed few-shot example logic)

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
            ValueError: If the JSON structure is invalid or context length exceeded
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

        # Context length warnings (remain relevant)
        if content_length > 250000 and model_type == ModelType.GPT_4O_MINI:
            self.logger.warning(f"Content length ({content_length} chars) is large for GPT-4o-mini context window. Consider upgrading model if issues occur.")
        if content_length > 500000 and model_type == ModelType.GPT_4O:
             self.logger.warning(f"Content length ({content_length} chars) is very large, approaching GPT-4o context window limits. Processing may be incomplete.")


        try:
            messages = [
                {"role": "system", "content": final_system_message},
                {"role": "user", "content": raw_content}
            ]

            # Calculate rough token estimate for logging
            estimated_tokens = content_length // 4
            self.logger.info(f"Estimated input tokens: ~{estimated_tokens}")

            try:
                response = await self.client.chat.completions.create(
                    model=model_type.value,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"} # Ensure JSON response
                )
                content = response.choices[0].message.content

                # Process usage information if available
                if hasattr(response, 'usage') and response.usage:
                    self.logger.info(f"Token usage - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")

                try:
                    # Validate JSON parsing immediately
                    parsed_content = json.loads(content)
                    if not self.validate_json(parsed_content):
                         self.logger.warning("JSON validation failed - missing required keys")
                         # Still return the content, as it might be usable even with missing keys

                    return content
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decoding error: {str(e)}")
                    self.logger.error(f"Raw content received: {content[:500]}...") # Log the first 500 chars for debugging
                    raise # Re-raise the JSONDecodeError

            except Exception as e:
                if "maximum context length" in str(e).lower() or "token limit" in str(e).lower():
                    self.logger.error(f"Context length exceeded: {str(e)}")
                    raise ValueError(f"Content too large for model context window: {str(e)}")
                else:
                    self.logger.error(f"API error: {str(e)}")
                    raise # Re-raise other API errors

        except Exception as e:
            self.logger.error(f"Error preparing or initiating AI processing: {str(e)}")
            raise # Re-raise any other unexpected errors

    def validate_json(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate the JSON structure.

        Args:
            json_data: Parsed JSON data

        Returns:
            True if the JSON has all required keys, False otherwise
        """
        # Basic validation - check for required top-level keys
        # Allow flexibility, just log warnings if keys are missing for now
        required_keys = ["metadata", "schedules", "notes"]
        missing_keys = [key for key in required_keys if key not in json_data]
        if missing_keys:
            self.logger.warning(f"JSON response missing expected keys: {', '.join(missing_keys)}")
            # Return True for now, but logging helps identify inconsistent outputs.
            # Could return False here if strict validation is needed later.

        # Specifications validation - check structure and convert if needed
        if "specifications" in json_data:
            specs = json_data["specifications"]
            if isinstance(specs, list) and specs:
                # Convert string arrays to object arrays if needed
                if isinstance(specs[0], str):
                    self.logger.warning("Converting specifications from string array to object array")
                    json_data["specifications"] = [{"section_title": spec, "content": ""} for spec in specs]
        # No else needed - if 'specifications' isn't there or is empty, that's okay

        # For architectural drawings, log if rooms array is missing
        if isinstance(json_data.get("metadata"), dict) and \
           "architectural" in json_data["metadata"].get("drawing_type", "").lower() and \
           "rooms" not in json_data:
            self.logger.warning("Architectural drawing missing 'rooms' array in JSON response")
            # Again, return True for now, but log the issue.

        return True # Return True unless strict validation requires False on warnings

# --- REMOVED get_example_output function ---

@time_operation("ai_processing")
async def process_drawing(raw_content: str, drawing_type: str, client, file_name: str = "") -> str:
    """
    Use GPT to parse PDF text and table data into structured JSON based on the drawing type.
    (Removed few-shot example logic)

    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        client: OpenAI client
        file_name: Optional name of the file being processed

    Returns:
        Structured JSON as a string

    Raises:
        ValueError: If the content is empty or too large for processing
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

        # --- REMOVED call to get_example_output ---

        # Check if this is a specification document
        is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"

        # Determine the appropriate system message based on detected subtype/type
        if is_specification:
            # Simplified system message retrieval for specifications
            system_message = ai_service._get_default_system_message(DrawingCategory.SPECIFICATIONS.value)
            logging.info("Using specification-specific system prompt.")
        else:
            # Get the appropriate system message based on detected subtype
            system_message = ai_service._get_default_system_message(subtype)
            logging.info(f"Using system prompt for subtype: {subtype}")


        # Process the drawing using the simplified prompt method
        try:
            response_str = await ai_service.process_with_prompt(
                raw_content=raw_content,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                model_type=params["model_type"],
                system_message=system_message,
                # --- REMOVED example_output parameter ---
            )

            # Basic validation check after receiving the string
            try:
                json.loads(response_str) # Try parsing to ensure it's valid JSON
                logging.info(f"Successfully processed {subtype} drawing ({len(response_str)} chars output)")
                return response_str
            except json.JSONDecodeError as json_err:
                 logging.error(f"Invalid JSON response from AI service for {file_name}: {json_err}")
                 logging.error(f"Raw response snippet: {response_str[:500]}...")
                 raise # Re-raise the JSON error

        except ValueError as val_err: # Catch context length errors from process_with_prompt
            logging.error(f"Value error processing {file_name}: {val_err}")
            raise # Re-raise the value error
        except Exception as proc_err: # Catch other API/processing errors
            logging.error(f"Error during AI processing call for {file_name}: {proc_err}")
            raise # Re-raise other errors


    except Exception as e:
        logging.error(f"Unexpected error setting up processing for {drawing_type} drawing '{file_name}': {str(e)}")
        raise # Re-raise any setup errors

# --- REMOVED process_drawing_with_examples function ---