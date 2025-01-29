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
    'Kitchen': ['K', 'KD']
}

def get_drawing_type(filename: str) -> str:
    """
    Detect the drawing type by examining the first 1-2 letters of the filename.
    """
    prefix = os.path.basename(filename).split('.')[0][:2].upper()
    for dtype, prefixes in DRAWING_TYPES.items():
        if any(prefix.startswith(p.upper()) for p in prefixes):
            return dtype
    return 'General'
