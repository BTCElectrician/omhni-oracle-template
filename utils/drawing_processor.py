"""
Drawing processing utilities that leverage the AI service.
"""
from typing import Dict, Any
import logging

from services.ai_service import DrawingAiService, ModelType, AiResponse
from utils.performance_utils import time_operation

DRAWING_INSTRUCTIONS = {
    "Electrical": "Focus on panel schedules, circuit info, equipment schedules with electrical characteristics, and installation notes.",
    "Mechanical": "Capture equipment schedules, HVAC details (CFM, capacities), and installation instructions.",
    "Plumbing": "Include fixture schedules, pump details, water heater specs, pipe sizing, and system instructions.",
    "Architectural": """
    Extract and structure the following information:
    1. Room details: Create a 'rooms' array with objects for each room, including:
       - 'number': Room number (as a string)
       - 'name': Room name
       - 'finish': Ceiling finish
       - 'height': Ceiling height
       - 'electrical_info': Any electrical specifications 
       - 'architectural_info': Any additional architectural details
    2. Room finish schedules
    3. Door/window details
    4. Wall types
    5. Architectural notes
    Ensure all rooms are captured and properly structured in the JSON output.
    """,
    "General": "Organize all relevant data into logical categories based on content type."
}


def optimize_model_parameters(
    drawing_type: str,
    raw_content: str,
    file_name: str
) -> Dict[str, Any]:
    """
    Determine optimal model parameters based on drawing type and content.
    
    Args:
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        raw_content: Raw content from the drawing
        file_name: Name of the file being processed
        
    Returns:
        Dictionary of optimized parameters
    """
    content_length = len(raw_content)
    
    # Default parameters
    params = {
        "model_type": ModelType.GPT_4O_MINI,
        "temperature": 0.2,
        "max_tokens": 16000
    }
    
    # Adjust based on drawing type
    if drawing_type == "Electrical":
        if "PANEL-SCHEDULES" in file_name.upper() or "PANEL_SCHEDULES" in file_name.upper():
            # Panel schedules need more structured output but are often simpler
            params["temperature"] = 0.1
            params["max_tokens"] = 8000
            
        elif "LIGHTING" in file_name.upper():
            # Lighting plans can be complex
            params["max_tokens"] = 12000
            
    elif drawing_type == "Architectural":
        # Architectural drawings need detailed processing
        if "REFLECTED CEILING" in file_name.upper():
            params["temperature"] = 0.15
            
    elif drawing_type == "Mechanical":
        if "SCHEDULES" in file_name.upper():
            # Mechanical schedules can be complex
            params["max_tokens"] = 12000
    
    # Adjust based on content length
    if content_length > 50000:
        # Very large documents may need more powerful model
        logging.info(f"Large document detected ({content_length} chars), using more powerful model")
        params["model_type"] = ModelType.GPT_4O
        
    elif content_length < 10000 and drawing_type not in ["Architectural", "Electrical"]:
        # Small documents for less critical drawing types could use faster models
        logging.info(f"Small document detected ({content_length} chars), optimizing for speed")
        params["max_tokens"] = 4000
    
    return params


@time_operation("ai_processing")
async def process_drawing(raw_content: str, drawing_type: str, client, file_name: str = "") -> str:
    """
    Use GPT to parse PDF text + table data into structured JSON
    based on the drawing type.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        client: OpenAI client
        file_name: Optional name of the file being processed
        
    Returns:
        Structured JSON as a string
    """
    try:
        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
        
        # Get optimized parameters for this drawing
        params = optimize_model_parameters(drawing_type, raw_content, file_name)
        
        logging.info(f"Using model {params['model_type'].value} with temperature {params['temperature']} " +
                    f"and max_tokens {params['max_tokens']} for {drawing_type} drawing")
        
        # Enhanced system message that emphasizes room extraction for architectural drawings
        system_message = f"""
        You are processing a construction drawing. Extract all relevant information and organize it into a JSON object with the following sections:
        - 'metadata': Include drawing number, title, date, etc.
        - 'schedules': Array of schedules with type and data.
        - 'notes': Array of notes.
        - 'specifications': Array of specification sections.
        - 'rooms': For architectural drawings, include an array of rooms with 'number', 'name', 'electrical_info', and 'architectural_info'.
        
        {DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"])}
        
        IMPORTANT: For architectural drawings, ALWAYS include a 'rooms' array, even if you have to infer room information from context.
        Ensure the output is valid JSON.
        """
        
        # Process the drawing using the Responses API (falls back to standard if needed)
        response: AiResponse[Dict[str, Any]] = await ai_service.process_drawing_with_responses(
            raw_content=raw_content,
            drawing_type=drawing_type,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            model_type=params["model_type"],
            system_message=system_message  # Pass the enhanced system message
        )
        
        if response.success:
            return response.content
        else:
            logging.error(f"Error processing {drawing_type} drawing: {response.error}")
            raise Exception(f"Error processing {drawing_type} drawing: {response.error}")
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing: {str(e)}")
        raise
