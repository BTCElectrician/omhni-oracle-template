"""
Registry system for managing prompt templates.
"""
from typing import Dict, Callable, Optional

# Define prompt registry as a dictionary of factories
PROMPT_REGISTRY: Dict[str, Callable[[], str]] = {}

def register_prompt(category: str, subtype: Optional[str] = None):
    """
    Decorator to register a prompt factory function.
    
    Args:
        category: Drawing category (e.g., "Electrical")
        subtype: Drawing subtype (e.g., "PanelSchedule")
        
    Returns:
        Decorator function that registers the decorated function
    """
    key = f"{category}_{subtype}" if subtype else category
    
    def decorator(func: Callable[[], str]):
        PROMPT_REGISTRY[key.upper()] = func
        return func
    
    return decorator

def get_registered_prompt(drawing_type: str) -> str:
    """
    Get prompt using registry with fallbacks.
    
    Args:
        drawing_type: Type of drawing (e.g., "Electrical_PanelSchedule")
        
    Returns:
        Prompt template string
    """
    # Handle case where drawing_type is None
    if not drawing_type:
        return PROMPT_REGISTRY.get("GENERAL", lambda: "")()
        
    # Normalize the key
    key = drawing_type.upper().replace(" ", "_")
    
    # Try exact match first
    if key in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[key]()
    
    # Try main category
    main_type = key.split("_")[0]
    if main_type in PROMPT_REGISTRY:
        return PROMPT_REGISTRY[main_type]()
    
    # Fall back to general
    return PROMPT_REGISTRY.get("GENERAL", lambda: "")()
