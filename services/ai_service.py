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
    GPT_4O = "gpt-4o"


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


class AiRequest(Generic[T]):
    """Generic request object for AI service."""
    def __init__(
        self,
        content: str,
        model_type: ModelType = ModelType.GPT_4O_MINI,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        response_format: Optional[Dict[str, Any]] = None,
        system_message: Optional[str] = None,
    ):
        self.content = content
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.response_format = response_format
        self.system_message = system_message


class AiResponse(Generic[T]):
    """Generic response object from AI service."""
    def __init__(
        self,
        content: str,
        success: bool,
        error: Optional[str] = None,
        parsed_content: Optional[T] = None,
        usage: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.success = success
        self.error = error
        self.parsed_content = parsed_content
        self.usage = usage or {}


class AiService(ABC, Generic[T]):
    """
    Abstract base class defining the interface for AI services.
    """
    @abstractmethod
    async def process(self, request: AiRequest[T]) -> AiResponse[T]:
        """
        Process content using an AI model.
        
        Args:
            request: AiRequest object containing the content to process
            
        Returns:
            AiResponse containing the processed content
        """
        pass


class JsonAiService(AiService[Dict[str, Any]]):
    """
    AI service implementation that returns JSON responses.
    """
    def __init__(
        self,
        client: AsyncOpenAI,
        logger: Optional[logging.Logger] = None
    ):
        self.client = client
        self.logger = logger or logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((AiRateLimitError, AiConnectionError)),
        reraise=True
    )
    async def _call_ai_with_retry(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Call the AI service with retry logic.
        
        Args:
            model: Model name
            messages: List of message dictionaries
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            response_format: Format for the response
            
        Returns:
            Raw response from the AI service
            
        Raises:
            AiRateLimitError: If rate limit is hit
            AiConnectionError: If connection fails
            AiResponseError: If response is invalid
        """
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            if response_format:
                kwargs["response_format"] = response_format
                
            response = await self.client.chat.completions.create(**kwargs)
            return response
        except Exception as e:
            error_message = str(e).lower()
            
            if "rate limit" in error_message:
                # Add jitter to avoid thundering herd
                jitter = random.uniform(0.1, 1.0)
                await asyncio.sleep(jitter)
                self.logger.warning(f"Rate limit hit: {error_message}")
                raise AiRateLimitError(f"Rate limit exceeded: {str(e)}")
            elif any(term in error_message for term in ["connection", "timeout", "network"]):
                self.logger.warning(f"Connection error: {error_message}")
                raise AiConnectionError(f"Connection error: {str(e)}")
            else:
                self.logger.error(f"AI service error: {error_message}")
                raise AiResponseError(f"AI service error: {str(e)}")

    async def process(self, request: AiRequest[Dict[str, Any]]) -> AiResponse[Dict[str, Any]]:
        """
        Process content using an AI model and return structured JSON.
        
        Args:
            request: AiRequest object containing the content to process
            
        Returns:
            AiResponse containing the processed content
        """
        try:
            start_time = time.time()
            self.logger.info(f"Processing content with {request.model_type.value}")
            
            # Prepare messages
            messages = []
            if request.system_message:
                messages.append({"role": "system", "content": request.system_message})
            messages.append({"role": "user", "content": request.content})
            
            # Set default response format if not provided
            response_format = request.response_format or {"type": "json_object"}
            
            try:
                # Call AI service with retry
                response = await self._call_ai_with_retry(
                    model=request.model_type.value,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_format=response_format
                )
                
                ai_content = response.choices[0].message.content
                
                # Try to parse JSON content
                try:
                    parsed_content = json.loads(ai_content)
                    
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"Successfully processed content in {elapsed_time:.2f}s")
                    
                    return AiResponse(
                        content=ai_content,
                        success=True,
                        parsed_content=parsed_content,
                        usage={
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                            "processing_time": elapsed_time
                        }
                    )
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse JSON response: {str(e)}")
                    return AiResponse(
                        content=ai_content,
                        success=False,
                        error=f"Invalid JSON response: {str(e)}"
                    )
                    
            except RetryError as e:
                original_error = e.last_attempt.exception()
                self.logger.error(f"Max retries reached: {str(original_error)}")
                return AiResponse(
                    content="",
                    success=False,
                    error=f"Max retries reached: {str(original_error)}"
                )
                
            except (AiRateLimitError, AiConnectionError, AiResponseError) as e:
                self.logger.error(f"AI service error: {str(e)}")
                return AiResponse(
                    content="",
                    success=False,
                    error=str(e)
                )
                
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Unexpected error processing content: {str(e)}")
            return AiResponse(
                content="",
                success=False,
                error=f"Unexpected error: {str(e)}",
                usage={"processing_time": elapsed_time}
            )


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