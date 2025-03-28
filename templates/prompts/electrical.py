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
    base_prompt = create_schedule_template(
        schedule_type="panel schedule",
        drawing_category="electrical",
        item_type="circuit",
        # Instructions integrated into key_properties for clarity
        key_properties="CRITICAL: First, create a 'DRAWING_METADATA' object containing drawing metadata like drawing_number, title, revision, date, job_number, project_name found in the title block. Second, create a main 'ELECTRICAL' object. Inside 'ELECTRICAL', create a 'PANEL_SCHEDULE' object. Inside 'PANEL_SCHEDULE', create a 'panel' object holding ALL panel metadata found (e.g., name, location, voltage, phases, wires, main_breaker/MLO, rating, aic_rating, mounting, enclosure, fed_from). This 'panel' object MUST also contain a 'circuits' list. For EACH circuit listed in the schedule, create a JSON object in the 'circuits' list containing AT LEAST 'circuit' number, 'trip' size (breaker amps), 'poles', and 'load_name'. Include ANY other details provided per circuit (VA, room, notes, GFCI status, etc.). Extract any general notes related to panels into an 'ELECTRICAL.general_notes' list.",
        example_structure="""
{
  "DRAWING_METADATA": {
    "drawing_number": "E4.01",
    "title": "PANEL SCHEDULES",
    "revision": "1",
    "date": "2024-01-15",
    "job_number": "P12345",
    "project_name": "Sample Project"
  },
  "ELECTRICAL": {
    "PANEL_SCHEDULE": {
      "panel": {
        "name": "K1",
        "location": "Kitchen 118",
        "voltage": "120/208 Wye",
        "phases": 3,
        "wires": 4,
        "main_breaker": "30 A Main Breaker",
        "rating": "225 A",
        "aic_rating": "14K",
        "mounting": "Surface",
        "enclosure": "NEMA 1",
        "fed_from": "MDP",
        "circuits": [
          {
            "circuit": "1",
            "load_name": "Kitchen Equipment - Refrigerator",
            "trip": "20 A",
            "poles": 1,
            "va_per_pole": 1200,
            "room_id": ["Kitchen 118"],
            "notes": "GFCI Breaker"
          },
          {
            "circuit": "3,5",
            "load_name": "Oven",
            "trip": "50 A",
            "poles": 2,
            "va_per_pole": 4800,
            "room_id": ["Kitchen 118"]
          }
        ],
        "panel_totals": {
           "total_connected_load_va": 25600,
           "total_demand_load_va": 21800,
           "total_connected_amps": 71.1,
           "total_demand_amps": 60.5
        }
      }
    },
    "general_notes": [
        "Verify all panel locations with architectural drawings.",
        "All breakers to be bolt-on type."
    ]
  }
}
""",
        source_location="panel schedule drawings or tables",
        preservation_focus="panel metadata, ALL circuit numbers, trip sizes, poles, and load descriptions",
        stake_holders="Electrical engineers, estimators, and installers",
        use_case="critical electrical system design, load calculation, and installation",
        critical_purpose="preventing safety hazards, ensuring code compliance, and proper circuit protection"
    )

    # Append critical formatting requirements
    formatting_reqs = """

CRITICAL FORMATTING REQUIREMENTS (Strict JSON):
1. Ensure ALL property names (keys) are enclosed in double quotes (e.g., "name": "K1").
2. ALL string values must be enclosed in double quotes (e.g., "voltage": "120/208 Wye"). Numeric values should NOT be quoted (e.g., "poles": 1). Boolean values (true/false) should NOT be quoted.
3. Ensure ALL items in an array (like 'circuits' or 'room_id') are separated by commas.
4. There must be NO trailing comma after the last item in an array or the last key-value pair in an object.
5. Objects must start with '{' and end with '}'. Arrays must start with '[' and end with ']'.
6. Each circuit in the 'circuits' array MUST be a complete JSON object '{...}'.

Example of correct circuit array formatting:
"circuits": [
  { "circuit": "1", "load_name": "Equipment", "trip": "20 A", "poles": 1 },
  { "circuit": "2", "load_name": "Lighting", "trip": "15 A", "poles": 1 }
]
(NO comma after the last circuit object '}')
"""
    return base_prompt + formatting_reqs

@register_prompt("Electrical", "LIGHTING")
def lighting_fixture_prompt():
    """Prompt for lighting fixtures."""
    return create_schedule_template(
        schedule_type="lighting fixture",
        drawing_category="electrical",
        item_type="lighting fixture",
        key_properties="CRITICAL: Extract all metadata from the drawing's title block, including drawing_number, title, revision, date, job_number, and project_name, placing it in the 'metadata' object. Also capture model numbers, descriptions, and performance data for fixtures.",
        example_structure="""
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
