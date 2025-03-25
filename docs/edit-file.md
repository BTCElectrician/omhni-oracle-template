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
          "circuits_count": 12,
          "dimensions": {
            "height": "25.5 Inches",
            "width": "20 Inches",
            "depth": "5.75 Inches"
          },
          "main_lugs_rating": "600 Amps Main Lugs"
        },
        "dimensions": {
          "height": "25.5 Inches",
          "width": "20 Inches",
          "depth": "5.75 Inches"
        },
        "main_breaker_rating": "100 A Main Breaker THQB",
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
        },
        "revisions": [
          {
            "description": "75% SET",
            "date": "10/06/2023"
          }
        ],
        "license_info": {
          "job_no": "30J7925",
          "pa": "TM",
          "license_expires": "11/30/2025"
        }
      }
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
      "dimensions": "15.75\" DIA x 13.78\" HEIGHT",
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
    },
    "POWER_CONNECTION": {
      "item": "E101A",
      "connection_type": "JUNCTION BOX",
      "quantity": 2,
      "description": "Door Heater / Conden. Drain Line Heater / Heated Vent Port",
      "breaker_size": "15A",
      "voltage": "120",
      "phase": 1,
      "mounting": "Ceiling",
      "height": "108\"",
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
    },
    "RISER_DIAGRAM_COMPONENT": {
      "equipment": "Main Service Entrance",
      "conductors": [
        {
          "type": "two sets of four 500 aluminum conductors with one number 2-op ground",
          "purpose": "Grounded Service Conductor",
          "notes": [
            "Main service switchboard MSB feeds Panel H1",
            "Provide two 3-inch EMT conduits for these conductors"
          ]
        }
      ],
      "grounding": {
        "main_ground_bar": "1/4\" x 2\" x 24\" CU",
        "ground_sources": [
          "Metal Underground Water Pipe",
          "Building Steel",
          "Ground Rod",
          "Concrete Encased Electrode"
        ],
        "ground_rods": "3/4\" x 10' CU Ground Rods"
      },
      "load_analysis": {
        "existing_demand": "1480 KVA",
        "existing_service": "3000 amps at 480/277 volts, 3PH, 4W",
        "new_load_added": {
          "retail": {
            "load_kva": "410.00",
            "demand_factor": "88.70%",
            "estimated_demand_kva": "363.67",
            "amps": "438"
          }
        },
        "total_load": {
          "baseline_kva": "1480",
          "new_load_kva": "363.67",
          "total_kva": "1843.67",
          "total_amps": "2219"
        }
      },
      "panel_info": {
        "id": "H1",
        "voltage": "480Y/277 volts",
        "phases": "3Ø, 4W",
        "rating": "600 amps",
        "type": "MLO",
        "circuits": "42CKTS"
      },
      "transformer_info": {
        "id": "TL1",
        "rating": "112.5 KVA",
        "voltage": "480 volts to 208Y/120 volts",
        "phases": "3 phase, four wire",
        "conductors": "4 one-aught conductors, 1 number four ground, 1 one-aught grounding electrode"
      },
      "connection_info": {
        "from": "Existing 3000A Main Switch",
        "to": "H1 Panel",
        "conductors": "2 sets of 4#500 aluminum, 1 #2 AWG ground",
        "conduit": "Two 3 inch EMT conduits"
      }
    },
    "ELECTRICAL_DETAIL": {
      "description": "Tech cupboard electrical works",
      "details": {
        "tech_cupboard": "Tech cupboard",
        "lamp_dimming_drivers": "Lamp dimming drivers located in tech cupboard",
        "number_of_lamps": 8,
        "cables": "8 cables",
        "power_by": "Power by Elec Contractor",
        "data_by": "Data by ES Vendor",
        "cabling_by": "Cabling by RE Tech Team",
        "plan_view": "Tech cupboard plan view",
        "fan": "Fan (MC supply, mains voltage)",
        "data_conduit": "1\" EMT for data",
        "tablet_cord": "Tablet cord FBO.",
        "light_cord": "Light cord FBO.",
        "bell_box": "2 gang bell box",
        "power_conduit": "3/4\" EMT for power",
        "data_cables": "3 data cables by Elec Contractor",
        "junction_box": "Junction box power supplies by EC",
        "tech_team": {
          "tx": "TX",
          "kvm": "KVM",
          "usb": "USB",
          "hdmi": "HDMI",
          "control_unit": "USB cable connection & control unit supply & install by Tech Team"
        }
      }
    },
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
        },
        "surge_protection_device_specification": {
          "section": "16200",
          "details": [
            "Transient voltage surge suppressor (TVSS)",
            "Listed by a nationally registered testing laboratory",
            "Five-year warranty with unlimited replacement",
            "TVSS rated for 120/208 VAC and 277/480 VAC",
            "Capacitive filtering system for EMI/RFI noise attenuation",
            "NEMA 2 rated, non-metallic enclosure",
            "Install with shortest lead conductor length possible"
          ]
        }
      }
    }
  },
  "MECHANICAL": {
    "EQUIPMENT": {
      "designation": "MAU-2",
      "manufacturer": "Greenheck",
      "model": "MSX-P116-H22-MF",
      "dimensions": "155 x 44 x 45",
      "weight": "153 lbs",
      "electric_preheat": "-",
      "volt_ph": "460/3",
      "circuit": "H1-25, 27, 29",
      "wiring": "3#12's",
      "protection": "15 amps",
      "horsepower": "3 HP",
      "location": "Ceiling",
      "serves": "General exhaust",
      "cfm": "5290 CFM",
      "notes": [
        "1. Factory-mounted, non-fused, disconnect switch, and single-point power connection.",
        "2. Factory-mounted supply air smoke detector.",
        "3. Makeup air unit to be interlocked to run when EF-2 is energized.",
        "4. Electric heater in reheat position with SCR control."
      ]
    },
    "ROOM_VENTILATION": {
      "room_number": "2104",
      "room_name": "Conference",
      "area": "589 SF",
      "supply_airflow": "1000 CFM",
      "exhaust_airflow": "1000 CFM",
      "ventilation_system": "Office",
      "air_outlets": [
        {
          "designation": "R1",
          "damper_type": "OBD",
          "face_size": "24x24",
          "max_pressure_drop": "0.05 in-wg",
          "max_noise_level": "30",
          "manufacturer": "TITUS",
          "model": "PXP"
        }
      ]
    },
    "VENTILATION_COMPONENT": {
      "designation": "S1",
      "damper_type": "OBD",
      "face_size": "24x24",
      "max_pressure_drop": "0.05 in-wg",
      "max_noise_level": "30",
      "manufacturer": "TITUS",
      "model": "OMNI",
      "service": "Supply",
      "max_airflow": "230 CFM",
      "neck_size": "8 inches"
    }
  },
  "PLUMBING": {
    "WATER_HEATER": {
      "mark": "WH-1",
      "location": "Kitchen",
      "storage_gallons_per_tank": "119",
      "operating_water_temp": "140°F",
      "tank_dimensions": "29.50\" x 62.25\"",
      "recovery_rate": "208 GPH",
      "elec_power_per_unit": "480/3",
      "kw_input": "40.5",
      "manufacturer_model_no": "A.O. Smith DRE-120",
      "remarks": "120 gallon electric water heater, floor mounted",
      "quantity": 1
    },
    "PUMP": {
      "mark": "CP",
      "location": "Kitchen",
      "serves": "Domestic Return",
      "type": "In-Line",
      "gpm": "8",
      "tdh_ft": "6",
      "hp": "1/6",
      "maximum_rpm": "3600",
      "volts_phase": "120/1",
      "cycle": "60",
      "manufacturer_model_number": "Armstrong Compass 20-20SS",
      "remarks": "Variable speed pump with SS body"
    },
    "FIXTURE": {
      "fixture_type": "Sink",
      "designation": "SK-1",
      "quantity": 1,
      "material": "Stainless Steel",
      "manufacturer": "See Arch Plans",
      "model_number": "See Arch Plans",
      "remarks": "ADA sink. Faucet to be single hole mount, single handle. Install with ADA covering over all exposed piping beneath sink.",
      "waste": "1-1/2\"",
      "vent": "1-1/4\"",
      "cw": "1/2\"",
      "hw": "1/2\""
    },
    "PIPE_MATERIAL": {
      "service": "Drain",
      "material": "Service Weight Cast Iron",
      "fittings": "Hub and Spigot"
    },
    "PIPE_SLOPE": {
      "pipe_size": "2 1/2 inches or less",
      "minimum_slope": "1/4\" per foot"
    }
  },
  "ARCHITECTURAL": {
    "ROOM": {
      "_comment": "This section is derived from the architectural template and is not part of the few-shot examples.",
      "Room Name": "DINING",
      "Room Number": 103,
      "North": "Exterior of space",
      "East": "Exterior of Space",
      "South": "S6BP",
      "West": "Not listed",
      "Notes": "Wing wall in the middle of the room is type \"S6BP\"",
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
      },
      "additional_equipment": {
        "electric_water_heater": 1,
        "dishwasher": 1,
        "furniture_feeds": 2
      }
    },
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
      "ceilingIntegration": {
        "ceilingType": "Scheduled",
        "reference": "Refer to RCP"
      },
      "floorIntegration": {
        "anchoring": {
          "baseTrack": "Metal deflection track anchored to building slab with ballistic pins",
          "sealant": "Fire rated sealant"
        },
        "finishBase": "Reference finish plan"
      },
      "partition_width": "7 5/8\""
    },
    "DOOR": {
      "door_id": "2100-01",
      "door_type": "A",
      "door_material": "Solid Core Wood",
      "hardware_type": "Standard Hardware",
      "finish": "PT-4 Paint",
      "louvers": "None",
      "dimensions": {
        "height": "7'-9\"",
        "width": "3'-0\"",
        "thickness": "1-3/4\""
      },
      "frame_type": "Type II (Snap-On Cover)",
      "glass_type": "None",
      "notes": "All private office doors to receive coat hook on interior side at 70\" AFF.",
      "use": "Office",
      "material": "Solid Core Wood",
      "associated_doors": [
        {
          "door_id": "2100-01",
          "door_type": "A",
          "material": "Solid Core Wood",
          "hardware_type": "Standard Hardware",
          "notes": "Private office door with coat hook at 70\" AFF."
        }
      ]
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