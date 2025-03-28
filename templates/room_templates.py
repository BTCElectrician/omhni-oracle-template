import json
import os
import logging
import copy  # Added for deepcopy

logger = logging.getLogger(__name__)

def load_template(template_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, f"{template_name}_template.json")
    try:
        with open(template_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logger.error(f"Template file not found: {template_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {template_path}: {e}")
        return {}

def generate_rooms_data(parsed_data, room_type):
    """
    Generates room data by merging AI parsed data into a predefined template.
    Ensures all fields from the template are present in the output for each room.
    """
    base_template = load_template(room_type) # Load the template structure (e.g., a_rooms_template)
    if not base_template:
        logger.error(f"Failed to load base template for {room_type}. Cannot generate room data.")
        return {"metadata": {}, "project_name": "", "floor_number": "", "rooms": []} # Return empty structure

    # Extract metadata and project name from DRAWING_METADATA if it exists
    metadata = parsed_data.get('DRAWING_METADATA', {}) # Look for DRAWING_METADATA
    project_name = metadata.get('project_name', '') # Get project name from there

    rooms_data_output = {
        "metadata": metadata, # Store the extracted metadata here
        "project_name": project_name,
        "floor_number": '', # Placeholder, extract if available
        "rooms": []
    }

    # Navigate potentially nested structure to find rooms list
    parsed_rooms = []
    if 'ARCHITECTURAL' in parsed_data and isinstance(parsed_data.get('ARCHITECTURAL'), dict) and 'ROOMS' in parsed_data['ARCHITECTURAL'] and isinstance(parsed_data['ARCHITECTURAL'].get('ROOMS'), list):
         parsed_rooms = parsed_data['ARCHITECTURAL']['ROOMS']
         logger.info(f"Found {len(parsed_rooms)} rooms under ARCHITECTURAL.ROOMS for {room_type}")
    elif 'rooms' in parsed_data and isinstance(parsed_data.get('rooms'), list): # Fallback for flat structure
         parsed_rooms = parsed_data.get('rooms', [])
         logger.info(f"Found {len(parsed_rooms)} rooms under top-level 'rooms' for {room_type}")

    if not parsed_rooms:
        drawing_num = metadata.get('drawing_number', 'N/A')
        logger.warning(f"No rooms found in parsed data for {room_type}. Checked 'ARCHITECTURAL.ROOMS' and 'rooms'. File: {drawing_num}")
        return rooms_data_output # Return structure with empty rooms list

    for parsed_room in parsed_rooms:
        # Try different keys for room number and name
        room_number_str = str(parsed_room.get('room_number', parsed_room.get('number', '')))
        if not room_number_str and 'room_id' in parsed_room: # Try parsing from room_id if number missing
            room_number_str = str(parsed_room['room_id']).replace('Room_', '')

        room_name = parsed_room.get('room_name', parsed_room.get('name', ''))

        if not room_number_str: # Check only for room number
            logger.warning(f"Skipping room due to missing 'room_number'/'number'/'room_id': {parsed_room}")
            continue

        # Create a fresh copy of the template for each room
        room_data = copy.deepcopy(base_template)

        # --- Core Logic ---
        # 1. Set basic identifiers
        room_data['room_id'] = f"Room_{room_number_str}"
        room_data['room_name'] = f"{room_name}_{room_number_str}" if room_name else f"Room_{room_number_str}" # Handle missing name

        # 2. Iterate through keys in the parsed AI data for this room
        for key, value in parsed_room.items():
            # Skip the keys we already handled or don't want to overwrite directly
            if key in ['room_number', 'number', 'name']:
                continue

            # If the key exists in our template, update the template's value
            if key in room_data:
                # You might need specific logic for merging nested structures if needed
                room_data[key] = value
            # Optional: Add keys from AI output even if not in template
            # else:
            #    room_data[key] = value

        rooms_data_output['rooms'].append(room_data)

    return rooms_data_output

def process_architectural_drawing(parsed_data, file_path, output_folder):
    """
    Process architectural drawing data (parsed JSON),
    and generate both e_rooms and a_rooms JSON outputs.
    """
    drawing_metadata = parsed_data.get('DRAWING_METADATA', {})
    drawing_number = drawing_metadata.get('drawing_number', '')
    title = drawing_metadata.get('title', '').upper()
    is_reflected_ceiling = "REFLECTED CEILING" in title

    # Generate data using the corrected function and correct template names
    e_rooms_data = generate_rooms_data(parsed_data, 'e_rooms')
    a_rooms_data = generate_rooms_data(parsed_data, 'a_rooms')

    # Define output filenames using the original PDF base name
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    # Place these files directly in the type_folder (e.g., output/Architectural/)
    e_rooms_file = os.path.join(output_folder, f'{base_filename}_e_rooms_details.json')
    a_rooms_file = os.path.join(output_folder, f'{base_filename}_a_rooms_details.json')

    # Save the generated data
    try:
        with open(e_rooms_file, 'w') as f:
            json.dump(e_rooms_data, f, indent=2)
        logger.info(f"Saved electrical room details to: {e_rooms_file}")
    except Exception as e:
        logger.error(f"Failed to save e_rooms data to {e_rooms_file}: {e}")

    try:
        with open(a_rooms_file, 'w') as f:
            json.dump(a_rooms_data, f, indent=2)
        logger.info(f"Saved architectural room details to: {a_rooms_file}")
    except Exception as e:
        logger.error(f"Failed to save a_rooms data to {a_rooms_file}: {e}")

    return {
        "e_rooms_file": e_rooms_file,
        "a_rooms_file": a_rooms_file,
        "is_reflected_ceiling": is_reflected_ceiling
    }
