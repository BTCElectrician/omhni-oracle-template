import json
import logging
from enum import Enum
from typing import Dict, Any, Optional
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.performance_utils import time_operation

class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"

class DrawingAiService:
    """
    Specialized AI service for processing construction drawings.
    """
    def __init__(self, client: AsyncOpenAI, logger: Optional[logging.Logger] = None):
        """
        Initialize the DrawingAiService.

        Args:
            client: AsyncOpenAI client instance
            logger: Optional logger instance
        """
        self.client = client
        self.logger = logger or logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process_with_prompt(
        self,
        raw_content: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI,
        system_message: Optional[str] = None
    ) -> str:
        """
        Process a construction drawing using a specific prompt.

        Args:
            raw_content: Raw content from the drawing
            temperature: Temperature parameter for the AI model
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            system_message: Optional custom system message to override default

        Returns:
            Processed content as a JSON string

        Raises:
            JSONDecodeError: If the response is not valid JSON
            ValueError: If the JSON structure is invalid
            Exception: For other processing errors
        """
        default_system_message = """
        You are an AI assistant tasked with processing construction drawings. Your job is to extract all relevant information from the provided content and organize it into a structured JSON object with the following sections:

        - "metadata": An object containing drawing metadata such as "drawing_number", "title", "date", and "revision". Include any available information; if a field is missing, omit it.

        - "schedules": An array of schedule objects. Each schedule should have a "type" (e.g., "electrical_panel", "mechanical") and a "data" array containing objects with standardized field names. For example, use "load_name" for fields like "Load" or "Load Type", and "trip" for "Breaker Size" or "Amperage". If a field doesn't match a standard name, include it as-is.

        - "notes": An array of strings containing any notes or instructions found in the drawing.

        - "specifications": An array of objects, each with a "section_title" and "content" for specification sections.
        
        - "rooms": For architectural drawings, include an array of rooms with 'number', 'name', 'electrical_info', and 'architectural_info'.

        IMPORTANT: For architectural drawings, ALWAYS include a 'rooms' array, even if you have to infer room information from context.
        
        Ensure that the entire response is a single, valid JSON object. Do not include any additional text or explanations outside of the JSON.
        """

        # Use the provided system message or fall back to default
        final_system_message = system_message if system_message else default_system_message

        try:
            response = await self.client.chat.completions.create(
                model=model_type.value,
                messages=[
                    {"role": "system", "content": final_system_message},
                    {"role": "user", "content": raw_content}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content
            parsed_content = json.loads(content)
            if not self.validate_json(parsed_content):
                raise ValueError("Invalid JSON structure: missing required keys")
            return content
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decoding error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing drawing: {str(e)}")
            raise

    def validate_json(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate the JSON structure.

        Args:
            json_data: Parsed JSON data

        Returns:
            True if the JSON has all required keys, False otherwise
        """
        required_keys = ["metadata", "schedules", "notes", "specifications"]
        return all(key in json_data for key in required_keys)