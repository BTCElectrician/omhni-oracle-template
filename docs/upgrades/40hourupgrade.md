# Comprehensive Prompt Optimization Implementation Plan

## Overview

This implementation plan addresses prompt optimization and code organization for the construction drawing processing system. The plan focuses on:

1. Creating specialized prompt templates with few-shot examples for different drawing types
2. Implementing default handlers for each trade category 
3. Reorganizing the code to improve modularity and maintainability
4. Ensuring flexibility to accommodate different naming conventions across projects
5. Reducing template duplication through a base template system
6. Adding type safety with enums for drawing categories and subtypes
7. Creating a flexible prompt registry for easier management

## Module Structure

```
templates/
  ├── __init__.py
  ├── prompt_types.py       # Enums for drawing types and subtypes
  ├── base_templates.py     # Base templates to reduce duplication
  ├── prompt_registry.py    # Registry system for prompt management
  ├── prompt_templates.py   # Main import interface
  ├── prompts/
  │   ├── __init__.py
  │   ├── architectural.py  # Architectural prompts
  │   ├── electrical.py     # Electrical prompts
  │   ├── mechanical.py     # Mechanical prompts
  │   ├── plumbing.py       # Plumbing prompts
  │   └── general.py        # General/default prompts
```

## Implementation Tasks

### Task 1: Define Type Enums

Create `templates/prompt_types.py` with enums for drawing categories and subtypes:

```python
# templates/prompt_types.py
from enum import Enum, auto

class DrawingCategory(Enum):
    """Main drawing categories."""
    ARCHITECTURAL = "Architectural"
    ELECTRICAL = "Electrical"
    MECHANICAL = "Mechanical"
    PLUMBING = "Plumbing"
    GENERAL = "General"
    SPECIFICATIONS = "Specifications"

class ArchitecturalSubtype(Enum):
    """Architectural drawing subtypes."""
    ROOM = "ROOM"
    CEILING = "CEILING"
    WALL = "WALL"
    DOOR = "DOOR"
    DETAIL = "DETAIL"
    DEFAULT = "DEFAULT"

class ElectricalSubtype(Enum):
    """Electrical drawing subtypes."""
    PANEL_SCHEDULE = "PANEL_SCHEDULE"
    LIGHTING = "LIGHTING"
    POWER = "POWER"
    FIREALARM = "FIREALARM"
    TECHNOLOGY = "TECHNOLOGY"
    SPEC = "SPEC"
    DEFAULT = "DEFAULT"

class MechanicalSubtype(Enum):
    """Mechanical drawing subtypes."""
    EQUIPMENT = "EQUIPMENT"
    VENTILATION = "VENTILATION"
    PIPING = "PIPING"
    DEFAULT = "DEFAULT"

class PlumbingSubtype(Enum):
    """Plumbing drawing subtypes."""
    FIXTURE = "FIXTURE"
    EQUIPMENT = "EQUIPMENT"
    PIPE = "PIPE"
    DEFAULT = "DEFAULT"
```

### Task 2: Create Base Template System

Create `templates/base_templates.py` with reusable prompt templates:

```python
# templates/base_templates.py
"""
Base prompt templates to reduce duplication across specific drawing types.
"""

BASE_DRAWING_TEMPLATE = """
You are extracting information from a {drawing_type} drawing.
Document ALL elements following this general structure, adapting to project-specific terminology.

EXTRACTION PRIORITIES:
1. Identify and extract ALL {element_type} elements
2. Document specifications EXACTLY as shown for each element
3. Preserve ALL notes, reference numbers, and special requirements

{specific_instructions}

EXAMPLE STRUCTURE (adapt based on what you find in the drawing):
{example_structure}

CRITICAL INSTRUCTIONS:
1. CAPTURE everything in the drawing
2. PRESERVE original terminology and organization
3. GROUP similar elements together in logical sections
4. DOCUMENT all specifications and detailed information

{industry_context}
"""

SCHEDULE_TEMPLATE = """
You are extracting {schedule_type} information from {drawing_category} drawings. 
Document ALL {item_type} following the structure in this example, while adapting to project-specific terminology.

EXTRACTION PRIORITIES:
1. Capture EVERY {item_type} with ALL specifications
2. Document ALL {key_properties} EXACTLY as shown
3. Include ALL notes, requirements, and special conditions

EXAMPLE OUTPUT STRUCTURE (field names may vary by project):
{example_structure}

CRITICAL INSTRUCTIONS:
1. EXTRACT all {item_type}s shown on the {source_location}
2. PRESERVE exact {preservation_focus}
3. INCLUDE all technical specifications and requirements 
4. ADAPT the structure to match this specific drawing
5. MAINTAIN the overall hierarchical organization shown in the example

{stake_holders} rely on this information for {use_case}.
Complete accuracy is essential for {critical_purpose}.
"""

def create_general_template(drawing_type, element_type, instructions, example, context):
    """Create a general prompt template with the provided parameters."""
    return BASE_DRAWING_TEMPLATE.format(
        drawing_type=drawing_type,
        element_type=element_type,
        specific_instructions=instructions,
        example_structure=example,
        industry_context=context
    )

def create_schedule_template(
    schedule_type, 
    drawing_category,
    item_type,
    key_properties,
    example_structure,
    source_location,
    preservation_focus,
    stake_holders,
    use_case,
    critical_purpose
):
    """Create a schedule prompt template with the provided parameters."""
    return SCHEDULE_TEMPLATE.format(
        schedule_type=schedule_type,
        drawing_category=drawing_category,
        item_type=item_type,
        key_properties=key_properties,
        example_structure=example_structure,
        source_location=source_location,
        preservation_focus=preservation_focus,
        stake_holders=stake_holders,
        use_case=use_case,
        critical_purpose=critical_purpose
    )
```

### Task 3: Create Prompt Registry

Create `templates/prompt_registry.py` for prompt registration and retrieval:

```python
# templates/prompt_registry.py
"""
Registry system for managing prompt templates.
"""
from typing import Dict, Callable, Optional

# Define prompt registry as a dictionary of factories
PROMPT_REGISTRY: Dict[str, Callable[[], str]] = {}

def register_prompt(category: str, subtype: Optional[str] = None):
    """
    Decorator to register a prompt factory function.
    
    Args:
        category: Drawing category (e.g., "Electrical")
        subtype: Drawing subtype (e.g., "PanelSchedule")
        
    Returns:
        Decorator function that registers the decorated function
    """
    key = f"{category}_{subtype}" if subtype else category
    
    def decorator(func: Callable[[], str]):
        PROMPT_REGISTRY[key.upper()] = func
        return func
    
    return decorator

def get_registered_prompt(drawing_type: str) -> str:
    """
    Get prompt using registry with fallbacks.
    
    Args:
        drawing_type: Type of drawing (e.g., "Electrical_PanelSchedule")
        
    Returns:
        Prompt template string
    """
    # Handle case where drawing_type is None
    if not drawing_type:
        return PROMPT_REGISTRY.get("GENERAL", lambda: "")()
        
    # Normalize the key
    key = drawing_type.upper().replace("_", "_")
    
    # Try exact match first
    if key in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[key]()
    
    # Try main category
    main_type = key.split("_")[0]
    if main_type in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[main_type]()
    
    # Fall back to general
    return PROMPT_REGISTRY.get("GENERAL", lambda: "")()
```

### Task 4: Implement General Prompts

Create `templates/prompts/general.py` with default prompt templates:

```python
# templates/prompts/general.py
"""
General prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template

@register_prompt("General")
def general_prompt():
    """General purpose prompt for any drawing type."""
    return create_general_template(
        drawing_type="construction",
        element_type="construction",
        instructions="""
Extract ALL elements following a logical structure, while adapting to project-specific terminology.
""",
        example="""
{
  "metadata": {
    "drawing_number": "X101",
    "title": "DRAWING TITLE",
    "date": "2023-05-15",
    "revision": "2"
  },
  "schedules": [
    {
      "type": "schedule_type",
      "data": [
        {"item_id": "X1", "description": "Item description", "specifications": "Technical details"}
      ]
    }
  ],
  "notes": ["Note 1", "Note 2"]
}
""",
        context="Engineers need EVERY element and specification value EXACTLY as shown - complete accuracy is essential for proper system design, ordering, and installation."
    )

# Register general prompt to ensure it's always available
GENERAL_PROMPT = general_prompt()
```

### Task 5: Implement Electrical Prompts

Create `templates/prompts/electrical.py`:

```python
# templates/prompts/electrical.py
"""
Electrical prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Electrical")
def default_electrical_prompt():
    """Default prompt for electrical drawings."""
    return create_general_template(
        drawing_type="ELECTRICAL",
        element_type="electrical",
        instructions="""
Focus on identifying and extracting ALL electrical elements (panels, fixtures, devices, connections, etc.).
""",
        example="""
{
  "ELECTRICAL": {
    "metadata": {
      "drawing_number": "E101",
      "title": "ELECTRICAL FLOOR PLAN",
      "date": "2023-05-15",
      "revision": "2"
    },
    "elements": {
      "panels": [],
      "fixtures": [],
      "devices": [],
      "connections": []
    },
    "notes": []
  }
}
""",
        context="Electrical engineers and installers rely on this information for proper system design and construction."
    )

@register_prompt("Electrical", "PANEL_SCHEDULE")
def panel_schedule_prompt():
    """Prompt for electrical panel schedules."""
    return create_schedule_template(
        schedule_type="panel schedule",
        drawing_category="electrical",
        item_type="circuit",
        key_properties="circuit assignments, breaker sizes, and load descriptions",
        example_structure="""
{
  "ELECTRICAL": {
    "PANEL_SCHEDULE": {
      "panel": {
        "name": "K1S",
        "voltage": "120/208 Wye",
        "phases": 3,
        "main_breaker": "30 A Main Breaker",
        "marks": "K1S",
        "aic_rating": "65K",
        "type": "MLO",
        "rating": "600 A",
        "specifications": {
          "sections": "1 Section(s)",
          "nema_enclosure": "Nema 1 Enclosure",
          "amps": "125 Amps",
          "phases": "3 Phase 4 Wire",
          "voltage": "480Y/277V",
          "frequency": "50/60 Hz",
          "interrupt_rating": "65kA Fully Rated",
          "incoming_feed": "Bottom",
          "fed_from": "1 inch conduit with 4#10's and 1#10 ground",
          "mounting": "Surface Mounted",
          "circuits_count": 12
        },
        "circuits": [
          {
            "circuit": 1,
            "load_name": "E-117(*)",
            "trip": "15 A",
            "poles": 1,
            "wires": 4,
            "info": "GFCI Circuit Breaker",
            "load_classification": "Kitchen Equipment",
            "connected_load": "1200 VA",
            "demand_factor": "65.00%",
            "equipment_ref": "E01",
            "room_id": ["Room_2104", "Room_2105"]
          }
        ],
        "panel_totals": {
          "total_connected_load": "5592 VA",
          "total_estimated_demand": "3635 VA", 
          "total_connected_amps": "16 A",
          "total_estimated_demand_amps": "10 A"
        }
      }
    }
  }
}
""",
        source_location="panel schedule",
        preservation_focus="circuit numbers, trip sizes, and load descriptions",
        stake_holders="Electrical engineers and installers",
        use_case="critical electrical system design and installation",
        critical_purpose="preventing safety hazards and ensuring proper function"
    )

@register_prompt("Electrical", "LIGHTING")
def lighting_fixture_prompt():
    """Prompt for lighting fixtures."""
    return create_schedule_template(
        schedule_type="lighting fixture",
        drawing_category="electrical",
        item_type="lighting fixture",
        key_properties="model numbers, descriptions, and performance data",
        example_structure="""
{
  "ELECTRICAL": {
    "LIGHTING_FIXTURE": {
      "type_mark": "CL-US-18",
      "count": 13,
      "manufacturer": "Mullan",
      "product_number": "MLP323",
      "description": "Essense Vintage Prismatic Glass Pendant Light",
      "finish": "Antique Brass",
      "lamp_type": "E27, 40W, 120V, 2200K",
      "mounting": "Ceiling",
      "dimensions": "15.75\\" DIA x 13.78\\" HEIGHT",
      "location": "Restroom Corridor and Raised Playspace",
      "wattage": "40W",
      "ballast_type": "LED Driver",
      "dimmable": "Yes",
      "remarks": "Refer to architectural",
      "catalog_series": "RA1-24-A-35-F2-M-C"
    },
    "LIGHTING_ZONE": {
      "zone": "Z1",
      "area": "Dining 103",
      "circuit": "L1-13",
      "fixture_type": "LED",
      "dimming_control": "ELV",
      "notes": "Shuffleboard Tables 3,4",
      "quantities_or_linear_footage": "16"
    }
  }
}
""",
        source_location="fixture schedule",
        preservation_focus="fixture types, models, and specifications",
        stake_holders="Lighting designers and electrical contractors",
        use_case="product selection, energy calculations, and installation",
        critical_purpose="proper lighting design and code compliance"
    )

@register_prompt("Electrical", "POWER")
def power_connection_prompt():
    """Prompt for power connections."""
    return create_schedule_template(
        schedule_type="power connection",
        drawing_category="electrical",
        item_type="power connection",
        key_properties="circuit assignments, breaker sizes, and loads",
        example_structure="""
{
  "ELECTRICAL": {
    "POWER_CONNECTION": {
      "item": "E101A",
      "connection_type": "JUNCTION BOX",
      "quantity": 2,
      "description": "Door Heater / Conden. Drain Line Heater / Heated Vent Port",
      "breaker_size": "15A",
      "voltage": "120",
      "phase": 1,
      "mounting": "Ceiling",
      "height": "108\\"",
      "current": "7.4A",
      "remarks": "Branch to connection, verify compressor location"
    },
    "HOME_RUN": {
      "id": "HR1",
      "circuits": [
        "28N",
        "47",
        "49",
        "51",
        "N"
      ]
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="circuit assignments and specifications",
        stake_holders="Electrical contractors",
        use_case="proper installation and coordination",
        critical_purpose="proper power distribution and equipment function"
    )

@register_prompt("Electrical", "SPEC")
def electrical_spec_prompt():
    """Prompt for electrical specifications."""
    return create_schedule_template(
        schedule_type="specification",
        drawing_category="electrical",
        item_type="specification section",
        key_properties="requirements, standards, and installation details",
        example_structure="""
{
  "ELECTRICAL": {
    "ELECTRICAL_SPEC": {
      "section": "16050",
      "title": "BASIC ELECTRICAL MATERIALS AND METHODS",
      "details": [
        "Installation completeness",
        "Compliance with NEC, OSHA, IEEE, UL, NFPA, and local codes",
        "Submittals for proposed schedule and deviations",
        "Listed and labeled products per NFPA 70",
        "Uniformity of manufacturer for similar equipment",
        "Coordination with construction and other trades",
        "Trenching and backfill requirements",
        "Warranty: Minimum one year",
        "Safety guards and equipment arrangement",
        "Protection of materials and apparatus"
      ],
      "subsection_details": {
        "depth_and_backfill_requirements": {
          "details": [
            "Trenches support on solid ground",
            "First backfill layer: 6 inches above the top of the conduit with select fill or pea gravel",
            "Minimum buried depth: 24 inches below finished grade for underground cables per NEC"
          ]
        },
        "wiring_requirements_and_wire_sizing": {
          "section": "16123",
          "details": [
            "Wire and cable for 600 volts and less",
            "Use THHN in metallic conduit for dry interior locations",
            "Use THWN in non-metallic conduit for underground installations",
            "Solid conductors for feeders and branch circuits 10 AWG and smaller",
            "Stranded conductors for control circuits",
            "Minimum conductor size for power and lighting circuits: 12 AWG",
            "Use 10 AWG for longer branch circuits as specified",
            "Conductor sizes are based on copper unless indicated as aluminum"
          ]
        }
      }
    }
  }
}
""",
        source_location="document",
        preservation_focus="section numbers, titles, and requirements",
        stake_holders="Contractors and installers",
        use_case="code compliance and proper installation",
        critical_purpose="meeting building code requirements"
    )

# Dictionary of all electrical prompts for backward compatibility
ELECTRICAL_PROMPTS = {
    "DEFAULT": default_electrical_prompt(),
    "PANEL_SCHEDULE": panel_schedule_prompt(),
    "LIGHTING": lighting_fixture_prompt(),
    "POWER": power_connection_prompt(),
    "SPEC": electrical_spec_prompt()
}
```

### Task 6: Implement Architectural Prompts

Create `templates/prompts/architectural.py`:

```python
# templates/prompts/architectural.py
"""
Architectural prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Architectural")
def default_architectural_prompt():
    """Default prompt for architectural drawings."""
    return create_general_template(
        drawing_type="ARCHITECTURAL",
        element_type="architectural",
        instructions="""
Focus on identifying and extracting ALL architectural elements (rooms, walls, doors, finishes, etc.).
""",
        example="""
{
  "ARCHITECTURAL": {
    "metadata": {
      "drawing_number": "A101",
      "title": "FLOOR PLAN",
      "date": "2023-05-15",
      "revision": "2"
    },
    "elements": {
      "rooms": [],
      "walls": [],
      "doors": [],
      "windows": [],
      "finishes": []
    },
    "notes": []
  }
}
""",
        context="Architects, contractors, and other trades rely on this information for coordination and construction."
    )

@register_prompt("Architectural", "ROOM")
def room_schedule_prompt():
    """Prompt for room schedules."""
    return create_schedule_template(
        schedule_type="room",
        drawing_category="architectural",
        item_type="room",
        key_properties="wall types, finishes, and dimensions",
        example_structure="""
{
  "ARCHITECTURAL": {
    "ROOM": {
      "room_id": "Room_2104",
      "room_name": "CONFERENCE 2104",
      "circuits": {
        "lighting": ["21LP-1"],
        "power": ["21LP-17"]
      },
      "light_fixtures": {
        "fixture_ids": ["F3", "F4"],
        "fixture_count": {
          "F3": 14,
          "F4": 2
        }
      },
      "outlets": {
        "regular_outlets": 3,
        "controlled_outlets": 1
      },
      "data": 4,
      "floor_boxes": 2,
      "mechanical_equipment": [
        {
          "mechanical_id": "fpb-21.03"
        }
      ],
      "switches": {
        "type": "vacancy sensor",
        "model": "WSX-PDT",
        "dimming": "0 to 10V",
        "quantity": 2,
        "mounting_type": "wall-mounted",
        "line_voltage": true
      }
    }
  }
}
""",
        source_location="floor plan",
        preservation_focus="room numbers and names",
        stake_holders="Architects, contractors, and other trades",
        use_case="coordination and building construction",
        critical_purpose="proper space planning and construction"
    )

@register_prompt("Architectural", "DOOR")
def door_schedule_prompt():
    """Prompt for door schedules."""
    return create_schedule_template(
        schedule_type="door",
        drawing_category="architectural",
        item_type="door",
        key_properties="hardware, frame types, and dimensions",
        example_structure="""
{
  "ARCHITECTURAL": {
    "DOOR": {
      "door_id": "2100-01",
      "door_type": "A",
      "door_material": "Solid Core Wood",
      "hardware_type": "Standard Hardware",
      "finish": "PT-4 Paint",
      "louvers": "None",
      "dimensions": {
        "height": "7'-9\\"",
        "width": "3'-0\\"",
        "thickness": "1-3/4\\""
      },
      "frame_type": "Type II (Snap-On Cover)",
      "glass_type": "None",
      "notes": "All private office doors to receive coat hook on interior side at 70\" AFF.",
      "use": "Office"
    },
    "DOOR_HARDWARE": {
      "hardware_type": "Standard Hardware",
      "components": [
        {
          "component": "Push/Pull",
          "model": "CRL 84LPBS",
          "finish": "Brushed Stainless",
          "lever_style": "03 Lever",
          "dimensions": "4-1/2\"",
          "type": "Full Side Closer",
          "note": "Integrated with card reader and motion detector",
          "notes": "Bi-Pass configuration"
        }
      ]
    }
  }
}
""",
        source_location="door schedule",
        preservation_focus="door numbers, types, and hardware groups",
        stake_holders="Architects, contractors, and hardware suppliers",
        use_case="procurement and installation",
        critical_purpose="proper door function and security"
    )

@register_prompt("Architectural", "WALL")
def wall_type_prompt():
    """Prompt for wall types."""
    return create_schedule_template(
        schedule_type="wall type",
        drawing_category="architectural",
        item_type="wall assembly",
        key_properties="assembly details, materials, and dimensions",
        example_structure="""
{
  "ARCHITECTURAL": {
    "WALL_TYPE": {
      "wallTypeId": "Type 1A",
      "description": "1/1A - Full Height Partition",
      "structure": {
        "metalDeflectionTrack": {
          "anchoredTo": "Building Structural Deck",
          "fasteners": "Ballistic Pins"
        },
        "studs": {
          "size": "3 5/8\"",
          "gauge": "20 GA",
          "spacing": "24\" O.C.",
          "fasteners": "1/2\" Type S-12 Pan Head Screws"
        },
        "gypsumBoard": {
          "layers": 1,
          "type": "5/8\" Type 'X'",
          "fastening": "Mechanically fastened to both sides of studs with 1\" Type S-12 screws, stagger joints 24\" O.C.",
          "orientation": "Apply vertically"
        },
        "insulation": {
          "type": "Acoustical Batt",
          "thickness": "3 1/2\"",
          "installation": "Continuous, friction fit"
        },
        "plywood": {
          "type": "Fire retardant treated",
          "thickness": "1/2\"",
          "location": "West side"
        }
      },
      "partition_width": "7 5/8\""
    }
  }
}
""",
        source_location="drawings",
        preservation_focus="materials, dimensions, and assembly details",
        stake_holders="Architects, contractors, and other trades",
        use_case="proper construction",
        critical_purpose="code compliance and building performance"
    )

# Dictionary of all architectural prompts for backward compatibility
ARCHITECTURAL_PROMPTS = {
    "DEFAULT": default_architectural_prompt(),
    "ROOM": room_schedule_prompt(),
    "DOOR": door_schedule_prompt(),
    "WALL": wall_type_prompt()
}
```

### Task 7: Implement Mechanical and Plumbing Prompts

Create similar files for mechanical and plumbing prompts following the same pattern. These can be implemented similarly to the electrical and architectural prompts above.

### Task 8: Create Main Prompt Template Interface

Create `templates/prompt_templates.py`:

```python
# templates/prompt_templates.py
"""
Main interface module for accessing prompt templates.
"""

from typing import Dict, Optional

# Import prompt dictionaries from each category
from templates.prompts.architectural import ARCHITECTURAL_PROMPTS
from templates.prompts.electrical import ELECTRICAL_PROMPTS
from templates.prompts.mechanical import MECHANICAL_PROMPTS
from templates.prompts.plumbing import PLUMBING_PROMPTS
from templates.prompts.general import GENERAL_PROMPT

# Import registry for more flexible prompt retrieval
from templates.prompt_registry import get_registered_prompt

# Mapping of main drawing types to prompt dictionaries (for backward compatibility)
PROMPT_CATEGORIES = {
    "Architectural": ARCHITECTURAL_PROMPTS,
    "Electrical": ELECTRICAL_PROMPTS, 
    "Mechanical": MECHANICAL_PROMPTS,
    "Plumbing": PLUMBING_PROMPTS
}

def get_prompt_template(drawing_type: str) -> str:
    """
    Get the appropriate prompt template based on drawing type.
    
    Args:
        drawing_type: Type of drawing (e.g., "Architectural", "Electrical_PanelSchedule")
        
    Returns:
        Prompt template string appropriate for the drawing type
    """
    # Default to general prompt if no drawing type provided
    if not drawing_type:
        return GENERAL_PROMPT
    
    # Try to get prompt from registry first (preferred method)
    registered_prompt = get_registered_prompt(drawing_type)
    if registered_prompt:
        return registered_prompt
    
    # Legacy fallback using dictionaries
    # Parse drawing type to determine category and subtype
    parts = drawing_type.split('_', 1)
    main_type = parts[0]
    
    # If main type not recognized, return general prompt
    if main_type not in PROMPT_CATEGORIES:
        return GENERAL_PROMPT
    
    # Get prompt dictionary for this main type
    prompt_dict = PROMPT_CATEGORIES[main_type]
    
    # Determine subtype (if any)
    subtype = parts[1].upper() if len(parts) > 1 else "DEFAULT"
    
    # Return the specific subtype prompt if available, otherwise the default for this category
    return prompt_dict.get(subtype, prompt_dict["DEFAULT"])

def get_available_subtypes(main_type: Optional[str] = None) -> Dict[str, list]:
    """
    Get available subtypes for a main drawing type or all types.
    
    Args:
        main_type: Optional main drawing type (e.g., "Architectural")
        
    Returns:
        Dictionary of available subtypes by main type
    """
    if main_type and main_type in PROMPT_CATEGORIES:
        # Return subtypes for specific main type
        return {main_type: list(PROMPT_CATEGORIES[main_type].keys())}
    
    # Return all subtypes by main type
    return {category: list(prompts.keys()) for category, prompts in PROMPT_CATEGORIES.items()}
```

### Task 9: Update Drawing Type Detection

Modify the `detect_drawing_subtype` function in `services/ai_service.py`:

```python
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
    
    from templates.prompt_types import (
        DrawingCategory, 
        ArchitecturalSubtype, 
        ElectricalSubtype,
        MechanicalSubtype,
        PlumbingSubtype
    )
    
    file_name_lower = file_name.lower()
    
    # Enhanced specification detection - check this first for efficiency
    if "specification" in drawing_type.lower() or any(term in file_name_lower for term in 
                                                   ["spec", "specification", ".spec", "e0.01"]):
        return DrawingCategory.SPECIFICATIONS.value
    
    # Electrical subtypes
    if drawing_type == DrawingCategory.ELECTRICAL.value:
        # Panel schedules
        if any(term in file_name_lower for term in ["panel", "schedule", "panelboard", "circuit", "h1", "l1", "k1", "k1s", "21lp-1", "20h-1"]):
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
        # Specifications
        elif any(term in file_name_lower for term in ["spec", "specification", "requirement"]):
            return f"{drawing_type}_{ElectricalSubtype.SPEC.value}"
    
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
```

### Task 10: Update AI Service to Use New Prompt Templates

Modify the `services/ai_service.py` file to use the new prompt templates:

```python
# In services/ai_service.py

# Replace imports
from templates.prompt_templates import get_prompt_template

# Replace the existing _get_default_system_message method with this:
def _get_default_system_message(self, drawing_type: str) -> str:
    """
    Get the default system message for the given drawing type with few-shot examples.
    
    Args:
        drawing_type: Type of drawing (Architectural, Electrical, etc.) or subtype
            
    Returns:
        System message string with examples
    """
    # Use the new prompt template module to get the appropriate template
    return get_prompt_template(drawing_type)
```

## Testing and Validation Plan

### Unit Testing

1. Create test module `tests/test_prompt_templates.py`:

```python
import unittest
from templates.prompt_templates import get_prompt_template, get_available_subtypes
from templates.prompt_types import DrawingCategory, ElectricalSubtype

class PromptTemplateTests(unittest.TestCase):
    def test_basic_template_retrieval(self):
        """Test that basic template retrieval works for main types."""
        self.assertIsNotNone(get_prompt_template("Electrical"))
        self.assertIsNotNone(get_prompt_template("Architectural"))
        self.assertIsNotNone(get_prompt_template("Mechanical"))
        self.assertIsNotNone(get_prompt_template("Plumbing"))
    
    def test_subtype_template_retrieval(self):
        """Test that subtype template retrieval works."""
        self.assertIsNotNone(get_prompt_template("Electrical_PANEL_SCHEDULE"))
        self.assertIsNotNone(get_prompt_template("Architectural_ROOM"))
    
    def test_unknown_type_fallback(self):
        """Test that unknown types fall back to general prompt."""
        general_prompt = get_prompt_template("General")
        unknown_prompt = get_prompt_template("UnknownType")
        self.assertEqual(general_prompt, unknown_prompt)
    
    def test_available_subtypes(self):
        """Test retrieval of available subtypes."""
        all_subtypes = get_available_subtypes()
        self.assertIn("Electrical", all_subtypes)
        self.assertIn("DEFAULT", all_subtypes["Electrical"])
        
        electrical_subtypes = get_available_subtypes("Electrical")
        self.assertIn("Electrical", electrical_subtypes)
        self.assertIn("PANEL_SCHEDULE", electrical_subtypes["Electrical"])

if __name__ == '__main__':
    unittest.main()
```

2. Create test module for drawing type detection:

```python
import unittest
from services.ai_service import detect_drawing_subtype

class DrawingTypeDetectionTests(unittest.TestCase):
    def test_electrical_panel_detection(self):
        """Test that electrical panel schedules are detected correctly."""
        self.assertEqual(
            detect_drawing_subtype("Electrical", "K1S_panel_schedule.pdf"),
            "Electrical_PANEL_SCHEDULE"
        )
        
    def test_architectural_room_detection(self):
        """Test that architectural floor plans are detected correctly."""
        self.assertEqual(
            detect_drawing_subtype("Architectural", "A101_floor_plan.pdf"),
            "Architectural_ROOM"
        )
        
    def test_specification_detection(self):
        """Test that specifications are detected correctly."""
        self.assertEqual(
            detect_drawing_subtype("Electrical", "electrical_spec.pdf"),
            "Electrical_SPEC"
        )
        
    def test_fallback_to_main_type(self):
        """Test fallback to main type when subtype not detected."""
        self.assertEqual(
            detect_drawing_subtype("Electrical", "unknown_drawing.pdf"),
            "Electrical"
        )

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

Test the integration with the AI service by processing sample drawings:

```python
import asyncio
import logging
from services.ai_service import process_drawing
from openai import AsyncOpenAI

async def test_drawing_processing():
    """Test drawing processing with different types."""
    client = AsyncOpenAI(api_key="your_api_key")
    
    # Test electrical panel schedule
    with open("test_data/panel_schedule.txt", "r") as f:
        raw_content = f.read()
    
    panel_result = await process_drawing(
        raw_content=raw_content,
        drawing_type="Electrical_PANEL_SCHEDULE",
        client=client,
        file_name="test_panel.pdf"
    )
    
    # Test architectural floor plan
    with open("test_data/floor_plan.txt", "r") as f:
        raw_content = f.read()
    
    room_result = await process_drawing(
        raw_content=raw_content,
        drawing_type="Architectural_ROOM",
        client=client,
        file_name="test_floor_plan.pdf"
    )
    
    # Print results
    print("Panel Schedule Result:")
    print(panel_result[:500] + "...")
    print("\nFloor Plan Result:")
    print(room_result[:500] + "...")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_drawing_processing())
```

## Implementation Steps

1. Create the necessary directory structure:
```
mkdir -p templates/prompts
```

2. Create base files:
```
touch templates/__init__.py
touch templates/prompt_types.py
touch templates/base_templates.py
touch templates/prompt_registry.py
touch templates/prompt_templates.py
touch templates/prompts/__init__.py
touch templates/prompts/general.py
touch templates/prompts/electrical.py
touch templates/prompts/mechanical.py
touch templates/prompts/plumbing.py
touch templates/prompts/architectural.py
```

3. Implement each module in the following order:
   - prompt_types.py (enums)
   - base_templates.py (template functions)
   - prompt_registry.py (registry system)
   - general.py (general prompts)
   - electrical.py, architectural.py, etc. (category-specific prompts)
   - prompt_templates.py (main interface)

4. Update AI service to use the new templates:
   - Modify _get_default_system_message in services/ai_service.py
   - Update detect_drawing_subtype to use the new prompt types

5. Add unit tests:
   - Test template retrieval
   - Test drawing type detection
   - Test integration with AI service

## Expected Benefits

### Improved Output Quality

- **More Consistent Structure**: Standard templates for each drawing type ensure consistent output format
- **Better Subtype Handling**: Specialized prompts for subtypes improve extraction quality
- **Example-Based Guidance**: Few-shot examples guide the model to produce properly structured output

### Better Code Organization

- **Modular Prompt System**: Clear separation between different drawing types
- **Single Responsibility**: Each module handles one category of prompts
- **Reduced Complexity**: AI service is simplified by removing prompt logic

### Enhanced Flexibility

- **Adaptability**: System explicitly instructs AI to adapt to project-specific terminology
- **Easy Extensions**: New drawing types and subtypes can be added without modifying core code
- **Multiple Access Methods**: Both registry-based and dictionary-based access for compatibility

### Performance Improvements

- **Selective Loading**: Registry pattern allows for lazy loading of prompts
- **Reduced Token Usage**: More focused prompts with specific examples
- **Streamlined Processing**: Clearer detection of drawing subtypes