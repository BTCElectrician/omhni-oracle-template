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
        key_properties="""CRITICAL: First, create a 'DRAWING_METADATA' object with drawing number, title, revision, date, job number, project name.
Next, create an 'ARCHITECTURAL' object containing a 'ROOMS' list.
For EACH room, extract 'room_number' and 'room_name' EXACTLY. Also extract any associated text like 'dimensions' or 'notes'.
Extract general notes into 'ARCHITECTURAL.general_notes'.""",
        example_structure="""
{
  "DRAWING_METADATA": {
    "drawing_number": "A2.2",
    "title": "DIMENSION FLOOR PLAN",
    "revision": "4",
    "date": "2024.06.25",
    "job_number": "GA1323",
    "project_name": "ELECTRIC SHUFFLE"
  },
  "ARCHITECTURAL": {
    "ROOMS": [
      {
        "room_id": "Room_101",
        "room_number": "101",
        "room_name": "ENTRY",
        "dimensions": "± 10' - 11 7/8\\"",
        "wall_type": "existing wall to remain", 
        "notes": ["Note directly found within or pointing to room 101"]
      },
      {
        "room_id": "Room_103",
        "room_number": "103",
        "room_name": "DINING 103",
        "dimensions": "± 27' - 7 3/4\\"", 
        "wall_type": "new wall", 
        "notes": []
      }
      // ... other rooms
    ],
    "general_notes": [ 
        "Contractor to verify all dimensions...",
        "Glazing, casework, millwork suppliers shall field verify..."
    ]
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
