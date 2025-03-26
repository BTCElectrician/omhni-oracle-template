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
