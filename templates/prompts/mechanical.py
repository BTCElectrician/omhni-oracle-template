"""
Mechanical prompt templates for construction drawing processing.
"""
from templates.prompt_registry import register_prompt
from templates.base_templates import create_general_template, create_schedule_template

@register_prompt("Mechanical")
def default_mechanical_prompt():
    """Default prompt for mechanical drawings."""
    return create_general_template(
        drawing_type="MECHANICAL",
        element_type="mechanical",
        instructions="""
Focus on identifying and extracting ALL mechanical elements (equipment, ductwork, piping, etc.) with their specifications.
""",
        example="""
{
  "MECHANICAL": {
    "metadata": {
      "drawing_number": "M101",
      "title": "MECHANICAL PLAN",
      "date": "2023-05-15",
      "revision": "2"
    },
    "equipment": {
      "air_handling_units": [],
      "fans": [],
      "vav_boxes": [],
      "pumps": []
    },
    "distribution": {
      "ductwork": [],
      "piping": []
    },
    "notes": []
  }
}
""",
        context="Mechanical engineers and contractors rely on this information for proper system design, coordination, and installation."
    )

@register_prompt("Mechanical", "EQUIPMENT")
def equipment_schedule_prompt():
    """Prompt for mechanical equipment schedules."""
    return create_schedule_template(
        schedule_type="equipment",
        drawing_category="mechanical",
        item_type="mechanical equipment",
        key_properties="model numbers, capacities, and performance data",
        example_structure="""
{
  "MECHANICAL": {
    "EQUIPMENT": {
      "equipment_id": "AHU-1",
      "type": "AIR HANDLING UNIT",
      "manufacturer": "Trane",
      "model": "CSAA012",
      "capacity": {
        "cooling": "12 Tons",
        "heating": "150 MBH",
        "airflow": "4,800 CFM"
      },
      "electrical": {
        "voltage": "460/3/60",
        "fla": "22.4 A",
        "mca": "28 A",
        "mocp": "45 A"
      },
      "dimensions": "96\" L x 60\" W x 72\" H",
      "weight": "2,500 lbs",
      "location": "Roof",
      "accessories": [
        "Economizer",
        "VFD",
        "MERV 13 Filters"
      ],
      "notes": "Provide seismic restraints per detail M5.1"
    }
  }
}
""",
        source_location="schedule",
        preservation_focus="equipment tags, specifications, and performance data",
        stake_holders="Mechanical engineers and contractors",
        use_case="equipment ordering and installation coordination",
        critical_purpose="proper mechanical system function and energy efficiency"
    )

@register_prompt("Mechanical", "VENTILATION")
def ventilation_prompt():
    """Prompt for ventilation elements."""
    return create_schedule_template(
        schedule_type="ventilation",
        drawing_category="mechanical",
        item_type="ventilation element",
        key_properties="airflow rates, dimensions, and connection types",
        example_structure="""
{
  "MECHANICAL": {
    "VENTILATION": {
      "element_id": "EF-1",
      "type": "EXHAUST FAN",
      "airflow": "1,200 CFM",
      "static_pressure": "0.75 in. w.g.",
      "motor": {
        "horsepower": "1/2 HP",
        "voltage": "120/1/60",
        "fla": "4.8 A"
      },
      "location": "Roof",
      "serving": "Restrooms 101, 102, 103",
      "dimensions": "24\" x 24\"",
      "mounting": "Curb mounted",
      "duct_connection": "16\" diameter",
      "controls": "Controlled by Building Management System",
      "operation": "Continuous during occupied hours"
    },
    "AIR_TERMINAL": {
      "element_id": "VAV-1",
      "type": "VARIABLE AIR VOLUME BOX",
      "size": "8 inch",
      "max_airflow": "450 CFM",
      "min_airflow": "100 CFM",
      "heating_capacity": "5 kW",
      "pressure_drop": "0.25 in. w.g.",
      "location": "Above ceiling in Room 201",
      "controls": "DDC controller with pressure-independent control"
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="airflow rates, equipment specifications, and control requirements",
        stake_holders="Mechanical engineers and contractors",
        use_case="ventilation system design and balancing",
        critical_purpose="proper indoor air quality and comfort"
    )

@register_prompt("Mechanical", "PIPING")
def piping_prompt():
    """Prompt for mechanical piping."""
    return create_schedule_template(
        schedule_type="piping",
        drawing_category="mechanical",
        item_type="piping system",
        key_properties="pipe sizes, materials, and flow rates",
        example_structure="""
{
  "MECHANICAL": {
    "PIPING": {
      "system_type": "CHILLED WATER",
      "pipe_material": "Copper Type L",
      "insulation": {
        "type": "Closed cell foam",
        "thickness": "1 inch",
        "jacket": "All-service vapor barrier jacket"
      },
      "design_pressure": "125 PSI",
      "design_temperature": "40°F supply, 55°F return",
      "flow_rate": "120 GPM",
      "sizes": [
        {
          "size": "2-1/2 inch",
          "location": "Main distribution",
          "flow_rate": "120 GPM",
          "velocity": "4.5 ft/s"
        },
        {
          "size": "2 inch", 
          "location": "Branch to AHU-1",
          "flow_rate": "60 GPM",
          "velocity": "4.2 ft/s"
        }
      ],
      "accessories": [
        {
          "type": "Ball Valve",
          "size": "2 inch",
          "location": "Each branch takeoff",
          "specification": "Bronze body, full port"
        },
        {
          "type": "Flow Meter",
          "size": "2-1/2 inch",
          "location": "Main supply",
          "specification": "Venturi type with pressure ports"
        }
      ],
      "notes": "Provide 3D coordination with all other trades prior to installation"
    }
  }
}
""",
        source_location="drawing",
        preservation_focus="pipe sizes, materials, and system specifications",
        stake_holders="Mechanical engineers and plumbing contractors",
        use_case="piping installation and coordination",
        critical_purpose="proper fluid distribution and system performance"
    )

# Dictionary of all mechanical prompts for backward compatibility
MECHANICAL_PROMPTS = {
    "DEFAULT": default_mechanical_prompt(),
    "EQUIPMENT": equipment_schedule_prompt(),
    "VENTILATION": ventilation_prompt(),
    "PIPING": piping_prompt()
}
