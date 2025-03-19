Simplified Processing Implementation
This document contains all code changes needed to implement the simplified drawing processing approach.
1. Add to services/ai_service.py
pythonCopy@time_operation("ai_processing")
async def process_drawing_simple(raw_content: str, drawing_type: str, client, file_name: str = "") -> str:
    """
    Process a drawing using a simple, universal prompt - similar to pasting into ChatGPT UI.
    Requires fewer specific instructions but relies on the model's understanding of construction drawings.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (for logging and minimal customization)
        client: OpenAI client
        file_name: Optional name of the file being processed
        
    Returns:
        Structured JSON as a string
    """
    try:
        # Log processing attempt
        logging.info(f"Processing {drawing_type} drawing with {len(raw_content)} characters using simplified approach")
        
        # Create minimal customization based on drawing type
        drawing_hint = ""
        if drawing_type == "Architectural":
            drawing_hint = " Include room information where available."
        elif drawing_type == "Electrical":
            drawing_hint = " Pay attention to panel schedules and circuit information."
        elif drawing_type == "Specifications":
            drawing_hint = " Preserve ALL specification text content completely."
        
        # Single, universal prompt
        system_message = f"""
        Structure this construction drawing content into well-organized JSON.{drawing_hint}
        Include all relevant information from the document and preserve the relationships between elements.
        Ensure your response is ONLY valid JSON with no additional text.
        """
        
        # Make the API call
        response = await client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_content}
            ],
            temperature=0.1,
            max_tokens=16000,
            response_format={"type": "json_object"}
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logging.error(f"Error in simplified processing of {drawing_type} drawing: {str(e)}")
        raise
2. Add to config/settings.py
Add this near the top with other settings:
pythonCopy# Processing Mode Configuration
USE_SIMPLIFIED_PROCESSING = os.getenv('USE_SIMPLIFIED_PROCESSING', 'false').lower() == 'true'
Then update the get_all_settings() function to include the new setting:
pythonCopydef get_all_settings() -> Dict[str, Any]:
    return {
        "OPENAI_API_KEY": "***REDACTED***" if OPENAI_API_KEY else None,
        "LOG_LEVEL": LOG_LEVEL,
        "BATCH_SIZE": BATCH_SIZE,
        "API_RATE_LIMIT": API_RATE_LIMIT,
        "TIME_WINDOW": TIME_WINDOW,
        "TEMPLATE_DIR": TEMPLATE_DIR,
        "DEBUG_MODE": DEBUG_MODE,
        # New setting
        "USE_SIMPLIFIED_PROCESSING": USE_SIMPLIFIED_PROCESSING
    }
3. Update processing/file_processor.py
Add this import at the top:
pythonCopyfrom config.settings import USE_SIMPLIFIED_PROCESSING
from services.ai_service import process_drawing, process_drawing_simple
Then modify the section in process_pdf_async function where it calls process_drawing:
pythonCopy# Find this line:
# structured_json = await process_drawing(raw_content, drawing_type, client, file_name)

# Replace with:
if USE_SIMPLIFIED_PROCESSING:
    logger.info(f"Using simplified processing approach for {file_name}")
    structured_json = await process_drawing_simple(raw_content, drawing_type, client, file_name)
else:
    structured_json = await process_drawing(raw_content, drawing_type, client, file_name)