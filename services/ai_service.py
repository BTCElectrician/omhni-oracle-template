"""
AI service interface and implementations for text processing with GPT models.
"""
import logging
from enum import Enum
from typing import Dict, Any, Optional, TypeVar, Generic
from openai import AsyncOpenAI

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from utils.performance_utils import time_operation

T = TypeVar('T')


class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"


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


class DrawingAiService:
    """
    Specialized AI service for processing construction drawings.
    """
    def __init__(
        self,
        client: AsyncOpenAI,
        drawing_instructions: Dict[str, str],
        logger: Optional[logging.Logger] = None
    ):
        self.client = client
        self.drawing_instructions = drawing_instructions
        self.logger = logger or logging.getLogger(__name__)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def process(self, request: AiRequest) -> AiResponse[str]:
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
            
            usage_info = {}
            if hasattr(response, "usage") and response.usage:
                usage_info["total_tokens"] = response.usage.total_tokens
                
            return AiResponse(success=True, content=content, usage=usage_info)
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
    ) -> AiResponse[str]:
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