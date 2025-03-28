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
Focus on identifying and extracting ALL plumbing elements with a comprehensive structure that includes:

1. Complete fixture schedules with every fixture type (sinks, water closets, urinals, lavatories, drains, cleanouts, etc.)
2. All equipment (water heaters, pumps, mixing valves, shock absorbers, etc.)
3. Pipe materials and connection requirements
4. All general notes, insulation notes, and special requirements
5. Capture each distinct schedule as a separate section with appropriate field structure

Pay special attention to equipment like pumps (CP), mixing valves (TM), and shock absorbers (SA) which must be captured even if located in separate tables or areas of the drawing.
""",
        example="""
{
  "PLUMBING": {
    "metadata": {
      "drawing_number": "P601",
      "title": "PLUMBING SCHEDULES",
      "date": "2023-05-15",
      "revision": "2"
    },
    "FIXTURE": [
      {
        "fixture_id": "S1",
        "description": "SINGLE COMPARTMENT SINK",
        "manufacturer": "McGuire Supplies",
        "model": "N/A",
        "mounting": "Contractor installed",
        "type": "17 gauge brass P-trap",
        "connections": {
          "cold_water": "1/2 inch",
          "waste": "2 inch",
          "vent": "2 inch",
          "hot_water": "1/2 inch"
        },
        "notes": "Contractor installed"
      }
    ],
    "WATER_HEATER": [
      {
        "mark": "WH-1",
        "location": "Mechanical Room",
        "manufacturer": "A.O. Smith",
        "model": "DRE-120-24",
        "specifications": {
          "storage_gallons": "120",
          "operating_water_temp": "140°F",
          "recovery_rate": "99 GPH"
        },
        "mounting": "Floor mounted",
        "notes": ["Provide T&P relief valve"]
      }
    ],
    "PUMP": [
      {
        "mark": "CP",
        "location": "Mechanical Room",
        "serves": "Hot Water Recirculation",
        "type": "IN-LINE",
        "gpm": "10",
        "tdh_ft": "20",
        "hp": "1/2",
        "electrical": "120V/1PH/60HZ",
        "manufacturer": "Bell & Gossett",
        "model": "Series 100"
      }
    ],
    "MIXING_VALVE": [
      {
        "designation": "TM",
        "location": "Mechanical Room",
        "manufacturer": "Powers",
        "model": "LFLM495-1",
        "notes": "Master thermostatic mixing valve"
      }
    ],
    "SHOCK_ABSORBER": [
      {
        "mark": "SA-A",
        "fixture_units": "1-11",
        "manufacturer": "Sioux Chief",
        "model": "660-A"
      }
    ],
    "MATERIAL_LEGEND": {
      "SANITARY SEWER PIPING": "CAST IRON OR SCHEDULE 40 PVC",
      "DOMESTIC WATER PIPING": "TYPE L COPPER"
    },
    "GENERAL_NOTES": [
      "A. All fixtures shall be installed per manufacturer's recommendations."
    ],
    "INSULATION_NOTES": [
      "A. Insulate all domestic hot water piping with 1\" thick fiberglass insulation."
    ]
  }
}
""",
        context="Plumbing engineers and contractors rely on ALL schedules, notes, and specifications for proper system design, coordination, and installation. Missing information can lead to serious installation issues or code violations."
    )

@register_prompt("Plumbing", "FIXTURE")
def fixture_schedule_prompt():
    """Prompt for plumbing fixture schedules."""
    return create_schedule_template(
        schedule_type="fixture",
        drawing_category="plumbing",
        item_type="plumbing fixture and equipment",
        key_properties="identifiers, models, specifications, connections, notes and all related schedules",
        example_structure="""
{
  "PLUMBING": {
    "metadata": {
      "drawing_number": "P601",
      "title": "PLUMBING SCHEDULES",
      "date": "2023-05-15",
      "revision": "2"
    },
    "FIXTURE": [
      {
        "fixture_id": "S1",
        "description": "SINGLE COMPARTMENT SINK",
        "manufacturer": "McGuire Supplies",
        "model": "N/A",
        "mounting": "Contractor installed",
        "type": "17 gauge brass P-trap",
        "connections": {
          "cold_water": "1/2 inch",
          "waste": "2 inch",
          "vent": "2 inch",
          "hot_water": "1/2 inch"
        },
        "location": "Refer to architect for specification",
        "notes": "Contractor installed"
      },
      {
        "fixture_id": "SW-01",
        "description": "WATER CLOSET",
        "manufacturer": "American Standard",
        "model": "2234.001",
        "mounting": "Floor mounted",
        "type": "1.28 GPF, elongated bowl",
        "connections": {
          "cold_water": "1/2 inch",
          "waste": "4 inch",
          "vent": "2 inch"
        },
        "location": "Various",
        "notes": "Provide floor flange and wax ring"
      }
    ],
    "WATER_HEATER": [
      {
        "mark": "WH-1",
        "location": "Mechanical Room",
        "manufacturer": "A.O. Smith",
        "model": "DRE-120-24",
        "specifications": {
          "storage_gallons": "120",
          "operating_water_temp": "140°F",
          "tank_dimensions": "26\" DIA x 71\" H",
          "recovery_rate": "99 GPH at 100°F rise",
          "electric_power": "480V, 3PH, 60HZ",
          "kW_input": "24"
        },
        "mounting": "Floor mounted",
        "notes": [
          "Provide T&P relief valve",
          "Provide seismic restraints per detail P5.1",
          "Provide expansion tank"
        ]
      }
    ],
    "PUMP": [
      {
        "mark": "CP",
        "location": "Mechanical Room",
        "serves": "Hot Water Recirculation",
        "type": "IN-LINE",
        "gpm": "10",
        "tdh_ft": "20",
        "hp": "1/2",
        "rpm": "1750",
        "electrical": "120V/1PH/60HZ",
        "manufacturer": "Bell & Gossett",
        "model": "Series 100",
        "notes": "Provide spring isolation hangers"
      }
    ],
    "MIXING_VALVE": [
      {
        "designation": "TM",
        "location": "Mechanical Room",
        "inlet_temp_F": "140",
        "outlet_temp_F": "120",
        "pressure_drop_psi": "5",
        "manufacturer": "Powers",
        "model": "LFLM495-1",
        "notes": "Master thermostatic mixing valve for domestic hot water system"
      }
    ],
    "SHOCK_ABSORBER": [
      {
        "mark": "SA-A",
        "fixture_units": "1-11",
        "manufacturer": "Sioux Chief",
        "model": "660-A",
        "description": "Water hammer arrestor, size A"
      },
      {
        "mark": "SA-B",
        "fixture_units": "12-32",
        "manufacturer": "Sioux Chief",
        "model": "660-B",
        "description": "Water hammer arrestor, size B"
      }
    ],
    "MATERIAL_LEGEND": {
      "SANITARY SEWER PIPING": "CAST IRON OR SCHEDULE 40 PVC",
      "VENT PIPING": "CAST IRON OR SCHEDULE 40 PVC",
      "DOMESTIC WATER PIPING": "TYPE L COPPER",
      "STORM DRAIN PIPING": "CAST IRON OR SCHEDULE 40 PVC"
    },
    "GENERAL_NOTES": [
      "A. All fixtures and equipment shall be installed per manufacturer's recommendations.",
      "B. Verify all rough-in dimensions with architectural drawings and manufacturer's cut sheets.",
      "C. All hot and cold water piping to be insulated per specifications."
    ],
    "INSULATION_NOTES": [
      "A. Insulate all domestic hot water piping with 1\" thick fiberglass insulation.",
      "B. Insulate all domestic cold water piping with 1/2\" thick fiberglass insulation."
    ]
  }
}
""",
        source_location="schedule",
        preservation_focus="ALL fixture types, equipment, pumps, valves, shock absorbers, materials, and notes",
        stake_holders="Plumbing engineers and contractors",
        use_case="comprehensive fixture and equipment selection and installation coordination",
        critical_purpose="proper system function, water conservation, and code compliance"
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
