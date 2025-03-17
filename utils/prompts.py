# utils/prompts.py
PROMPTS = {
    "electrical_panel_schedule": """
You are an expert in parsing electrical panel schedules. The text may contain multiple panels, each with their own specifications and circuits. Extract the data and structure it into a JSON array, where each object represents a panel with two main sections: 'marks' and 'panel'. The 'marks' section should include keys like 'section', 'amps', 'interrupt_rating', 'feed', 'circuits', 'certifications' (as an array), 'dimensions' (with 'height', 'width', 'depth'), and 'breaker'. The 'panel' section should include 'name', 'voltage', 'feed', and a 'circuits' array, where each circuit has 'circuit' (number), 'load_name', 'trip', 'poles', and 'equipment_ref' or 'equipment_refs' (array if multiple). Ensure all panels and their details are captured accurately, preserving units and formatting as in the text.
""",
    "mechanical_schedule": """
You are an expert in parsing mechanical equipment schedules. The text contains details about equipment like exhaust fans, water heaters, and condensing units. Extract the data and structure it into a JSON object with a 'metadata' section (including 'type', 'project', 'sheet_number', 'issue_date', 'general_notes' array) and an 'equipment' array. Each equipment object should include 'designation', 'manufacturer', 'model', 'dimensions', 'weight', 'electric_preheat', 'volt_ph', 'circuit', 'wiring', 'protection', and 'notes'. Group equipment by type if indicated, and preserve all specified values and units as provided in the text.
""",
    "plumbing_schedule": """
You are an expert in parsing plumbing schedules. The text includes details about water heaters, pumps, and other plumbing fixtures. Extract the data and structure it into a JSON object with sections: 'electric_water_heater_schedule' (array of water heaters with 'mark', 'location', 'storage_gallons_per_tank', 'operating_water_temp', 'tank_dimensions', 'recovery_rate', 'elec_power_per_unit', 'kw_input', 'manufacturer_model_no', 'remarks'), 'pump_schedule' (array of pumps with 'mark', 'location', 'serves', 'type', 'gpm', 'tdh_ft', 'hp', 'maximum_rpm', 'volts_phase', 'cycle', 'manufacturer_model_number', 'remarks'), 'plumbing_general_notes' (array), 'plumbing_symbols' (array with 'symbol' and 'description'), and 'connection_schedule' (array with 'type_of_fixture', 'waste', 'vent', 'cw', 'hw'). Capture all details accurately, maintaining units and formatting.
""",
    "architectural_schedule": """
You are an expert in parsing architectural details, such as wall types or partition types (sometimes called finish schedules). The text contains descriptions of different wall types with their properties. Extract the data and structure it into a JSON object with a 'wall_types' array, where each object has 'type' and a 'details' object containing 'material', 'stud_width', 'partition_width', and any additional properties. Ensure all wall types and their specifications are captured, preserving measurements and descriptions as in the text.
""",
    "default": """
Extract all the text from the drawing and organize it into a structured JSON format. Preserve the original wording and structure as much as possible. Identify sections, lists, and key-value pairs where applicable, and output them in a logical JSON object.
"""
}