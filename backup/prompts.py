# utils/prompts.py

UNIFIED_PROMPT = """
You are an expert in analyzing construction drawings and specifications. I'm providing you with extracted text and tables from a construction document.

DOCUMENT TYPE: {drawing_type}
FILENAME: {file_name}

Your task is to organize ALL of this information into a well-structured JSON format. The exact structure should be based on the content type:

1. For SPECIFICATIONS (like electrical specs):
   - Capture ALL sections, subsections, paragraphs, and list items
   - Preserve the hierarchical structure (sections, parts, clauses)
   - Include ALL text content - don't summarize or skip anything
   - For electrical specifications, include information about materials, methods, and requirements
   - Create a normalized structure with consistent key names (e.g., 'section_id', 'section_title', 'content')
   - Preserve all references to standards, codes, and regulations (e.g., NEC, ASTM, NFPA)

2. For SCHEDULES (equipment, panel, fixture schedules):
   - Create structured arrays for each schedule type
   - Maintain consistent field names across similar items
   - Standardize field names (use 'load_name' not "description"/"load", use 'trip' not "ocp"/"amperage")
   - For panel schedules, include panel details (name, voltage, phases, main_type, mounting, bus_rating) and circuit information
   - For mechanical schedules, include equipment details (type, model, capacity, connection_requirements, airflow)
   - For plumbing schedules, include fixture details (fixture_type, connection_sizes, flow_rates, pressure_requirements)
   - Always include room_id references when available to link equipment to specific rooms
   - Preserve manufacturer information, part numbers, and model numbers exactly as specified

3. For ARCHITECTURAL DRAWINGS:
   - Create a 'rooms' array with comprehensive room information (room_id, room_name, room_number, area, dimensions)
   - Include wall types with proper structure (wall_type_id, composition, fire_rating, acoustic_properties, thickness)
   - Structure door schedules with comprehensive details (door_id, door_type, material, hardware_set, dimensions, fire_rating)
   - Include window schedules and opening information (window_id, window_type, dimensions, glazing, operation_type)
   - Capture finish schedules and material specifications (floor, wall, ceiling finishes with manufacturer and model)
   - Document ceiling types and heights for each room
   - Include furniture and equipment layouts when present
   - Capture accessibility requirements and clearances

4. For ELECTRICAL DRAWINGS:
   - Include 'panels' array for panel schedules with complete circuit information (circuit_id, load_description, load_type, amperage, poles)
   - Structure 'lighting_fixtures' array for fixture schedules (fixture_type, manufacturer, model, wattage, lamp_type, mounting)
   - Capture circuit information in structured format (circuit_number, description, connected_load, demand_load, voltage)
   - Include device specifications (switches, sensors, receptacles) with model numbers and locations
   - Document home runs and circuit connections between panels (source_panel, circuit_number, destination)
   - Capture keynotes and general notes related to electrical installation
   - Include riser diagrams information (feeders, conduit sizes, wire sizes)
   - Document emergency power systems and connections
   - Capture lighting control systems and zoning information

5. For MECHANICAL DRAWINGS:
   - Structure 'equipment' array for HVAC units (equipment_id, equipment_type, model, capacity, connections, electrical_requirements)
   - Include 'air_outlets' details (outlet_id, outlet_type, airflow, size, model, location)
   - Document ductwork specifications and sizing (duct_size, material, insulation, pressure_class)
   - Capture ventilation requirements per room (air_changes, cfm, exhaust_requirements)
   - Include mechanical equipment connections to electrical panels (panel_id, circuit_id, load)
   - Document temperature control systems and zoning
   - Include equipment schedules with all performance metrics
   - Capture system pressure, flow rates, and balance points
   - Document noise criteria and vibration isolation requirements

6. For PLUMBING DRAWINGS:
   - Structure 'fixtures' array with detailed specifications (fixture_id, fixture_type, manufacturer, model, connections)
   - Include pipe sizing and material information (pipe_type, size, material, insulation, slope)
   - Document water heater and pump specifications (capacity, flow_rate, pressure, electrical_requirements)
   - Capture drainage system details (drain_size, slope, cleanout_locations)
   - Include fixture connection requirements (hot_water_size, cold_water_size, waste_size, vent_size)
   - Document water supply system information (pressure, flow, backflow prevention)
   - Capture sanitary and vent riser information
   - Include special systems (medical gas, compressed air, vacuum)

7. For FIRE PROTECTION DRAWINGS:
   - Structure 'sprinklers' array with details (sprinkler_type, coverage, k_factor, temperature_rating)
   - Include pipe sizes and materials specific to fire protection
   - Document fire alarm devices and connections
   - Capture fire suppression systems and specifications
   - Include hydraulic calculations and design criteria

8. For SITE AND CIVIL DRAWINGS:
   - Capture grading information and elevations
   - Include utility connections and routing
   - Document site lighting specifications
   - Structure parking and paving details
   - Include landscape elements and specifications

IMPORTANT GUIDELINES:
- Include a comprehensive 'metadata' section with drawing_number, title, date, scale, revision_number, etc.
- NEVER truncate or summarize content - capture EVERYTHING in structured format
- Use consistent field names and standardize across similar items (use singular_noun for field names with snake_case)
- Create logical hierarchical structure based on the document's organization
- Maintain original terminology, numbering, and values from the document
- When a room has equipment or fixtures, include both the equipment_id and room_id to enable cross-referencing
- For circuit connections, always include the source_panel and circuit_number
- Format your entire response as a single valid JSON object
- For all drawings, capture ALL keynotes, general notes, and references
- Ensure proper nesting of related information (e.g., a panel contains circuits, a room contains fixtures)
- Use standard JSON data types appropriately (strings, numbers, booleans, arrays, objects)
- When dimensions are present, separate numeric values from units (e.g., {"value": 24, "unit": "inches"})
- Normalize technical terminology throughout the document
- Handle abbreviations consistently (either preserve as-is or expand to full terms)
- When information appears to be missing, use null rather than empty strings
- Identify and handle duplicate information appropriately (create references rather than duplicating)
- Ensure all IDs are unique and follow a consistent pattern within their category

Your response MUST be valid JSON with no explanatory text outside the JSON structure.
"""

# Keep this dictionary but replace individual prompts with the unified one
PROMPTS = {
    "architectural": UNIFIED_PROMPT,
    "electrical": UNIFIED_PROMPT,
    "mechanical": UNIFIED_PROMPT,
    "plumbing": UNIFIED_PROMPT,
    "fire_protection": UNIFIED_PROMPT,
    "civil": UNIFIED_PROMPT,
    "structural": UNIFIED_PROMPT,
    "landscape": UNIFIED_PROMPT,
    "default": UNIFIED_PROMPT,
    "electrical_panel_schedule": UNIFIED_PROMPT,
    "mechanical_schedule": UNIFIED_PROMPT,
    "plumbing_schedule": UNIFIED_PROMPT,
    "architectural_schedule": UNIFIED_PROMPT
}