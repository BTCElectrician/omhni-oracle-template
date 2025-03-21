import os

DRAWING_TYPES = {
    'Architectural': ['A', 'AD'],
    'Electrical': ['E', 'ED'],
    'Mechanical': ['M', 'MD'],
    'Plumbing': ['P', 'PD'],
    'Site': ['S', 'SD'],
    'Civil': ['C', 'CD'],
    'Low Voltage': ['LV', 'LD'],
    'Fire Alarm': ['FA', 'FD'],
    'Kitchen': ['K', 'KD'],
    'Specifications': ['SPEC', 'SP']
}

def get_drawing_type(filename: str) -> str:
    """
    Detect the drawing type by examining the first 1-2 letters of the filename.
    """
    basename = os.path.basename(filename).upper()
    
    # Check by prefixes FIRST
    prefix = basename.split('.')[0][:2].upper()
    for dtype, prefixes in DRAWING_TYPES.items():
        if any(prefix.startswith(p.upper()) for p in prefixes):
            return dtype
            
    # Only use Specifications as a fallback if we couldn't determine by prefix
    if "SPEC" in basename or "SPECIFICATION" in basename:
        return "Specifications"
        
    return 'General'

def get_drawing_subtype(filename: str) -> str:
    """
    Detect the drawing subtype based on keywords in the filename.
    """
    filename_lower = filename.lower()
    if "panel schedule" in filename_lower or "electrical schedule" in filename_lower:
        return "electrical_panel_schedule"
    elif "mechanical schedule" in filename_lower:
        return "mechanical_schedule"
    elif "plumbing schedule" in filename_lower:
        return "plumbing_schedule"
    elif "wall types" in filename_lower or "partition types" in filename_lower:
        return "architectural_schedule"
    else:
        return "default"