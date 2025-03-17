"""
AI service interface and implementations for text processing with GPT models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, TypeVar, Generic
import logging
import json
import asyncio
import time
import random
from enum import Enum

from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from utils.performance_utils import time_operation


class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini"


class AiError(Exception):
    """Base exception for AI service errors."""
    pass


class AiRateLimitError(AiError):
    """Exception raised when AI service rate limit is hit."""
    pass


class AiConnectionError(AiError):
    """Exception raised when connection to AI service fails."""
    pass


class AiResponseError(AiError):
    """Exception raised when AI service returns an unexpected response."""
    pass


T = TypeVar('T')


class AiRequest:
    """Request object for AI service."""
    def __init__(
        self,
        content: str,
        model_type: ModelType,
        temperature: float,
        max_tokens: int,
        response_format: Dict[str, str],
        system_message: str
    ):
        self.content = content
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.system_message = system_message


class AiResponse(Generic[T]):
    """Response object from AI service."""
    def __init__(
        self,
        success: bool,
        content: Optional[T] = None,
        error: Optional[str] = None,
        usage: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.content = content
        self.error = error
        self.usage = usage or {}


class JsonAiService(ABC):
    """
    Abstract base class for AI services that return JSON responses.
    """
    def __init__(
        self,
        client: AsyncOpenAI,
        logger: Optional[logging.Logger] = None
    ):
        self.client = client
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    async def process(self, request: AiRequest) -> AiResponse[Dict[str, Any]]:
        """
        Process content using an AI model.
        
        Args:
            request: AiRequest object containing the content to process
            
        Returns:
            AiResponse containing the processed content
        """
        pass


class DrawingAiService(JsonAiService):
    """
    Specialized AI service for processing construction drawings.
    """
    def __init__(
        self,
        client: AsyncOpenAI,
        drawing_instructions: Dict[str, str],
        logger: Optional[logging.Logger] = None
    ):
        super().__init__(client, logger)
        self.drawing_instructions = drawing_instructions

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def process(self, request: AiRequest) -> AiResponse[Dict[str, Any]]:
        """
        Process content using an AI model.
        
        Args:
            request: AiRequest object containing the content to process
            
        Returns:
            AiResponse containing the processed content
        """
        try:
            response = await self.client.chat.completions.create(
                model=request.model_type.value,
                messages=[
                    {"role": "system", "content": request.system_message},
                    {"role": "user", "content": request.content}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format=request.response_format
            )
            content = response.choices[0].message.content
            return AiResponse(success=True, content=content)
        except Exception as e:
            self.logger.error(f"AI processing error: {str(e)}")
            return AiResponse(success=False, error=str(e))

    @time_operation("ai_processing")
    async def process_with_prompt(
        self,
        raw_content: str,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI
    ) -> str:
        """
        Process a drawing using a specific prompt.
        
        Args:
            raw_content: Raw content from the drawing
            prompt: Custom prompt to use for processing
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            
        Returns:
            Processed content as a string
        """
        request = AiRequest(
            content=raw_content,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            system_message=prompt
        )
        
        response = await self.process(request)
        if response.success:
            return response.content
        else:
            self.logger.error(f"AI processing failed: {response.error}")
            raise Exception(f"AI processing failed: {response.error}")

    @time_operation("ai_processing")
    async def process_drawing(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI
    ) -> AiResponse[Dict[str, Any]]:
        """
        Process a construction drawing using the AI service.
        
        Args:
            raw_content: Raw content from the drawing
            drawing_type: Type of drawing (Architectural, Electrical, etc.)
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            
        Returns:
            AiResponse containing the processed drawing
        """
        instruction = self.drawing_instructions.get(drawing_type, self.drawing_instructions.get("General", ""))
        
        system_message = f"""
        Parse this {drawing_type} drawing/schedule into a structured JSON format. Guidelines:
        1. For text: Extract key information, categorize elements.
        2. For tables: Preserve structure, use nested arrays/objects.
        3. Create a hierarchical structure, use consistent key names.
        4. Include metadata (drawing number, scale, date) if available.
        5. {instruction}
        6. For all drawing types, if room information is present, always include a 'rooms' array in the JSON output, 
           with each room having at least 'number' and 'name' fields.
        Ensure the entire response is a valid JSON object.
        """
        
        request = AiRequest(
            content=raw_content,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            system_message=system_message
        )
        
        return await self.process(request)
    
    @time_operation("ai_processing")
    async def process_drawing_with_responses(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI
    ) -> AiResponse[Dict[str, Any]]:
        """
        Process a construction drawing using OpenAI's Responses API.
        
        Args:
            raw_content: Raw content from the drawing
            drawing_type: Type of drawing (Architectural, Electrical, etc.)
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            
        Returns:
            AiResponse containing the processed drawing
        """
        instruction = self.drawing_instructions.get(drawing_type, self.drawing_instructions.get("General", ""))
        
        system_message = f"""
        Parse this {drawing_type} drawing/schedule into a structured JSON format. Guidelines:
        1. For text: Extract key information, categorize elements.
        2. For tables: Preserve structure, use nested arrays/objects.
        3. Create a hierarchical structure, use consistent key names.
        4. Include metadata (drawing number, scale, date) if available.
        5. {instruction}
        6. For all drawing types, if room information is present, always include a 'rooms' array in the JSON output, 
           with each room having at least 'number' and 'name' fields.
        Ensure the entire response is a valid JSON object.
        """
        
        try:
            # Check if client has responses API support
            if hasattr(self.client, 'responses') and callable(getattr(self.client.responses, 'create', None)):
                self.logger.info(f"Using Responses API for {drawing_type} drawing")
                
                # Define schema based on drawing type
                schema = self._get_schema_for_drawing_type(drawing_type)
                
                start_time = time.time()
                # Call the Responses API
                try:
                    response = await self.client.responses.create(
                        model=model_type.value,
                        system_prompt=system_message,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        response_schema=schema,
                        input=raw_content
                    )
                    
                    ai_content = json.dumps(response.outputs.structured_output)
                    elapsed_time = time.time() - start_time
                    
                    self.logger.info(f"Successfully processed content with Responses API in {elapsed_time:.2f}s")
                    
                    return AiResponse(
                        content=ai_content,
                        success=True,
                        parsed_content=response.outputs.structured_output,
                        usage={
                            "processing_time": elapsed_time
                        }
                    )
                except Exception as api_error:
                    self.logger.error(f"Error during Responses API call: {str(api_error)}")
                    raise api_error
            else:
                self.logger.info(f"Responses API not available, falling back to standard API")
                # Fall back to standard method
                return await self.process_drawing(
                    raw_content=raw_content,
                    drawing_type=drawing_type,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model_type=model_type
                )
                
        except Exception as e:
            self.logger.error(f"Error using Responses API: {str(e)}")
            self.logger.info(f"Falling back to standard API method")
            
            # Fall back to standard method
            return await self.process_drawing(
                raw_content=raw_content,
                drawing_type=drawing_type,
                temperature=temperature,
                max_tokens=max_tokens,
                model_type=model_type
            )
    
    def _get_schema_for_drawing_type(self, drawing_type: str) -> Dict[str, Any]:
        """
        Get the appropriate schema for a drawing type.
        
        Args:
            drawing_type: Type of drawing
            
        Returns:
            JSON schema for the drawing type
        """
        # Base schema for all drawing types
        base_schema = {
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "properties": {
                        "drawing_number": {"type": "string"},
                        "title": {"type": "string"},
                        "date": {"type": "string"},
                        "scale": {"type": "string"},
                        "project": {"type": "string"}
                    }
                },
                "notes": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["metadata"]
        }
        
        # Add drawing-specific schema elements
        if drawing_type == "Architectural":
            base_schema["properties"]["rooms"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "number": {"type": "string"},
                        "name": {"type": "string"},
                        "finish": {"type": "string"},
                        "height": {"type": "string"}
                    },
                    "required": ["number", "name"]
                }
            }
            base_schema["required"].append("rooms")
            
        elif drawing_type == "Electrical":
            base_schema["properties"]["panels"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "location": {"type": "string"},
                        "voltage": {"type": "string"},
                        "circuits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "number": {"type": ["string", "number"]},
                                    "description": {"type": "string"},
                                    "load": {"type": ["string", "number"]}
                                }
                            }
                        }
                    }
                }
            }
            
        elif drawing_type == "Mechanical":
            base_schema["properties"]["equipment"] = {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "tag": {"type": "string"},
                        "description": {"type": "string"},
                        "capacity": {"type": "string"},
                        "model": {"type": "string"}
                    }
                }
            }
            
        return base_schema 