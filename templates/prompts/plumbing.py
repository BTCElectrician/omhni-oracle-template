"""
Plumbing prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Plumbing")
def default_plumbing_prompt():
    """Default prompt for plumbing drawings."""
    return create_general_template(
        drawing_type="PLUMBING",
        element_type="plumbing",
        instructions="""
Focus on identifying and extracting ALL plumbing elements (fixtures, equipment, piping, etc.) with their specifications.
""",
        example="""
{
  "PLUMBING": {
    "metadata": {
      "drawing_number": "P101",
      "title": "PLUMBING PLAN",
      "date": "2023-05-15",
      "revision": "2"
    },
    "fixtures": [],
    "equipment": [],
    "piping": {
      "domestic_water": [],
      "waste": [],
      "vent": []
    },
    "notes": []
  }
}
""",
        context="Plumbing engineers and contractors rely on this information for proper system design, coordination, and installation."
    )

@register_prompt("Plumbing", "FIXTURE")
def fixture_schedule_prompt():
    """Prompt for plumbing fixture schedules."""
    return create_schedule_template(
        schedule_type="fixture",
        drawing_category="plumbing",
        item_type="plumbing fixture",
        key_properties="model numbers, connection sizes, and flow rates",
        example_structure="""
{
  "PLUMBING": {
    "FIXTURE": {
      "fixture_id": "P-1",
      "description": "WATER CLOSET",
      "manufacturer": "American Standard",
      "model": "2234.001",
      "mounting": "Floor mounted",
      "type": "1.28 GPF, elongated bowl",
      "connections": {
        "cold_water": "1/2 inch",
        "waste": "4 inch"
      },
      "accessories": [
        "Toilet seat: Church 9500NSSC",
        "Carrier: Josam 12674",
        "Flush valve: Sloan Royal 111-1.28"
      ],
      "ada_compliant": true,
      "location": "Restrooms 101, 102, 103",
      "rough_in_height": "15 inches to rim",
      "notes": "Provide floor flange and wax ring"
    }
  }
}
""",
        source_location="schedule",
        preservation_focus="fixture types, models, and connection requirements",
        stake_holders="Plumbing engineers and contractors",
        use_case="fixture selection and installation coordination",
        critical_purpose="proper fixture function and water conservation"
    )

@register_prompt("Plumbing", "EQUIPMENT")
def equipment_schedule_prompt():
    """Prompt for plumbing equipment schedules."""
    return create_schedule_template(
        schedule_type="equipment",
        drawing_category="plumbing",
        item_type="plumbing equipment",
        key_properties="capacities, connection sizes, and electrical requirements",
        example_structure="""
{
  "PLUMBING": {
    "EQUIPMENT": {
      "equipment_id": "WH-1",
      "type": "WATER HEATER",
      "manufacturer": "A.O. Smith",
      "model": "DRE-120-24",
      "capacity": "120 gallons",
      "heating_input": "24 kW",
      "recovery_rate": "99 GPH at 100°F rise",
      "electrical": {
        "voltage": "480/3/60",
        "full_load_amps": "28.9 A",
        "minimum_circuit_ampacity": "36.1 A",
        "maximum_overcurrent_protection": "40 A"
      },
      "connections": {
        "cold_water_inlet": "1-1/2 inch",
        "hot_water_outlet": "1-1/2 inch",
        "recirculation": "3/4 inch",
        "relief_valve": "1 inch"
      },
      "dimensions": "26\" DIA x 71\" H",
      "weight": "650 lbs empty, 1,650 lbs full",
      "location": "Mechanical Room 151",
      "notes": "Provide seismic restraints per detail P5.1"
    }
  }
}
""",
        source_location="schedule",
        preservation_focus="equipment specifications and performance requirements",
        stake_holders="Plumbing engineers and contractors",
        use_case="equipment selection and installation coordination",
        critical_purpose="proper hot water system function and energy efficiency"
    )

@register_prompt("Plumbing", "PIPE")
def pipe_schedule_prompt():
    """Prompt for plumbing pipe schedules."""
    return create_schedule_template(
        schedule_type="pipe",
        drawing_category="plumbing",
        item_type="pipe",
        key_properties="materials, sizes, and connection methods",
        example_structure="""
{
  "PLUMBING": {
    "PIPE": {
      "system_type": "DOMESTIC WATER",
      "pipe_material": "Copper Type L",
      "insulation": {
        "type": "Fiberglass",
        "thickness": "1 inch for cold water, 1.5 inch for hot water",
        "jacket": "All-service vapor barrier jacket"
      },
      "joining_method": "Soldered, lead-free",
      "design_pressure": "80 PSI working pressure",
      "testing_pressure": "125 PSI for 4 hours",
      "sizes": [
        {
          "size": "3 inch",
          "location": "Main distribution",
          "flow_rate": "150 GPM",
          "velocity": "6.1 ft/s"
        },
        {
          "size": "1-1/2 inch", 
          "location": "Branch to Restrooms",
          "flow_rate": "45 GPM",
          "velocity": "5.8 ft/s"
        }
      ],
      "accessories": [
        {
          "type": "Ball Valve",
          "size": "3 inch",
          "location": "Main shut-off",
          "specification": "Bronze body, full port, 600 WOG"
        },
        {
          "type": "Pressure Reducing Valve",
          "size": "3 inch",
          "location": "Service entrance",
          "specification": "Watts 223, set at 65 PSI"
        }
      ],
      "notes": "Provide water hammer arrestors at all quick-closing valves"
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="pipe materials, sizes, and installation requirements",
        stake_holders="Plumbing engineers and contractors",
        use_case="piping system installation and coordination",
        critical_purpose="proper water distribution and system performance"
    )

# Dictionary of all plumbing prompts for backward compatibility
PLUMBING_PROMPTS = {
    "DEFAULT": default_plumbing_prompt(),
    "FIXTURE": fixture_schedule_prompt(),
    "EQUIPMENT": equipment_schedule_prompt(),
    "PIPE": pipe_schedule_prompt()
}
