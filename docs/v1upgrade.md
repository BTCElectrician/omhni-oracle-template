# Ohmni Oracle Refactoring Implementation

## Status

**Complete**: The upgrade plan has been fully implemented.

## New Directory Structure

```
/Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template
├── config
│   └── settings.py (updated)
├── processing
│   ├── __init__.py
│   ├── batch_processor.py (updated)
│   ├── file_processor.py (updated)
│   └── job_processor.py (updated)
├── services  # New directory
│   ├── __init__.py (new)
│   ├── ai_service.py (new)
│   ├── extraction_service.py (new)
│   └── storage_service.py (new)
├── templates
│   ├── __init__.py
│   ├── a_rooms_template.json
│   ├── e_rooms_template.json
│   └── room_templates.py
├── utils
│   ├── __init__.py
│   ├── api_utils.py (updated)
│   ├── constants.py
│   ├── drawing_processor.py (updated)
│   ├── file_utils.py
│   ├── logging_utils.py (updated)
│   ├── pdf_processor.py (updated)
│   └── pdf_utils.py (updated)
└── ...
```

## New Files

### services/__init__.py
```python
# Services package initialization
```

### services/extraction_service.py
```python
"""
Extraction service interface and implementations for PDF content extraction.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio

import pymupdf as fitz


class ExtractionResult:
    """
    Domain model representing the result of a PDF extraction operation.
    """
    def __init__(
        self,
        raw_text: str,
        tables: List[Dict[str, Any]],
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.raw_text = raw_text
        self.tables = tables
        self.success = success
        self.error = error
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary."""
        return {
            "raw_text": self.raw_text,
            "tables": self.tables,
            "success": self.success,
            "error": self.error,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create an ExtractionResult from a dictionary."""
        return cls(
            raw_text=data.get("raw_text", ""),
            tables=data.get("tables", []),
            success=data.get("success", False),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


class PdfExtractor(ABC):
    """
    Abstract base class defining the interface for PDF extraction services.
    """
    @abstractmethod
    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractionResult containing the extracted content
        """
        pass


class PyMuPdfExtractor(PdfExtractor):
    """
    PDF content extractor implementation using PyMuPDF.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text and tables from a PDF file using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractionResult containing the extracted content
        """
        try:
            self.logger.info(f"Starting extraction for {file_path}")
            
            # Use run_in_executor to move CPU-bound work off the main thread
            loop = asyncio.get_event_loop()
            raw_text, tables, metadata = await loop.run_in_executor(
                None, self._extract_content, file_path
            )
            
            self.logger.info(f"Successfully extracted content from {file_path}")
            return ExtractionResult(
                raw_text=raw_text,
                tables=tables,
                success=True,
                metadata=metadata
            )
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {str(e)}")
            return ExtractionResult(
                raw_text="",
                tables=[],
                success=False,
                error=str(e)
            )

    def _extract_content(self, file_path: str) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Internal method to extract content from a PDF file.
        This method runs in a separate thread.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (raw_text, tables, metadata)
        """
        with fitz.open(file_path) as doc:
            raw_text = ""
            tables = []
            
            # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": len(doc)
            }
            
            # Extract text and tables from each page
            for i, page in enumerate(doc):
                # Extract text
                page_text = page.get_text()
                raw_text += f"PAGE {i+1}:\n{page_text}\n\n"
                
                # Extract tables
                page_tables = page.find_tables()
                for j, table in enumerate(page_tables):
                    table_dict = {
                        "page": i+1,
                        "table_index": j,
                        "rows": len(table),
                        "cols": table.rect.width,
                        "content": table.to_markdown()
                    }
                    tables.append(table_dict)
            
            return raw_text, tables, metadata
```

### services/ai_service.py
```python
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

    async def process_drawing(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI
    ) -> AiResponse[Dict[str, Any]]:
        """
        Process a construction drawing using an AI model.
        
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
```

### services/storage_service.py
```python
"""
Storage service interface and implementations for saving processing results.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, BinaryIO
import os
import json
import logging
import aiofiles
import asyncio


class StorageService(ABC):
    """
    Abstract base class defining the interface for storage services.
    """
    @abstractmethod
    async def save_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Save JSON data to a file.
        
        Args:
            data: JSON-serializable data to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def save_text(self, text: str, file_path: str) -> bool:
        """
        Save text data to a file.
        
        Args:
            text: Text content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def save_binary(self, data: bytes, file_path: str) -> bool:
        """
        Save binary data to a file.
        
        Args:
            data: Binary content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read JSON data from a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Parsed JSON data if successful, None otherwise
        """
        pass


class FileSystemStorage(StorageService):
    """
    Storage service implementation using the local file system.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def save_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        Save JSON data to a file.
        
        Args:
            data: JSON-serializable data to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file asynchronously
            async with aiofiles.open(file_path, 'w') as f:
                json_str = json.dumps(data, indent=2)
                await f.write(json_str)
                
            self.logger.info(f"Successfully saved JSON to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON to {file_path}: {str(e)}")
            return False

    async def save_text(self, text: str, file_path: str) -> bool:
        """
        Save text data to a file.
        
        Args:
            text: Text content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file asynchronously
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(text)
                
            self.logger.info(f"Successfully saved text to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving text to {file_path}: {str(e)}")
            return False

    async def save_binary(self, data: bytes, file_path: str) -> bool:
        """
        Save binary data to a file.
        
        Args:
            data: Binary content to save
            file_path: Path where the file should be saved
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file asynchronously
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(data)
                
            self.logger.info(f"Successfully saved binary data to {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving binary data to {file_path}: {str(e)}")
            return False

    async def read_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Read JSON data from a file.
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            Parsed JSON data if successful, None otherwise
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found: {file_path}")
                return None
                
            # Read the file asynchronously
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                
            # Parse JSON
            data = json.loads(content)
            self.logger.info(f"Successfully read JSON from {file_path}")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON from {file_path}: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error reading JSON from {file_path}: {str(e)}")
            return None
```

## Updated Files

### utils/pdf_processor.py (updated)
```python
"""
PDF processing utilities that leverage the extraction service.
"""
import os
import json
import logging
from typing import Dict, Any, Tuple, Optional

from services.extraction_service import PyMuPdfExtractor, ExtractionResult
from services.ai_service import DrawingAiService, AiResponse, ModelType


async def extract_text_and_tables_from_pdf(pdf_path: str) -> str:
    """
    Extract text and tables from a PDF file.
    This is a legacy wrapper around the new extraction service.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted content as a string
    """
    extractor = PyMuPdfExtractor()
    result = await extractor.extract(pdf_path)
    
    all_content = ""
    if result.success:
        all_content += "TEXT:\n" + result.raw_text + "\n"
        
        for table in result.tables:
            all_content += "TABLE:\n"
            all_content += table["content"] + "\n"
    
    return all_content


async def structure_panel_data(client, raw_content: str) -> Dict[str, Any]:
    """
    Structure panel data using the AI service.
    
    Args:
        client: OpenAI client
        raw_content: Raw content from the panel PDF
        
    Returns:
        Structured panel data as a dictionary
    """
    from utils.drawing_processor import DRAWING_INSTRUCTIONS
    
    ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
    
    system_message = """
    You are an expert in electrical engineering and panel schedules. 
    Please structure the following content from an electrical panel schedule into a valid JSON format. 
    The content includes both text and tables. Extract key information such as panel name, voltage, amperage, circuits, 
    and any other relevant details.
    Pay special attention to the tabular data, which represents circuit information.
    Ensure your entire response is a valid JSON object.
    """
    
    response = await ai_service.process_drawing(
        raw_content=raw_content,
        drawing_type="Electrical",
        temperature=0.2,
        max_tokens=2000,
        model_type=ModelType.GPT_4O_MINI
    )
    
    if response.success and response.parsed_content:
        return response.parsed_content
    else:
        logging.error(f"Failed to structure panel data: {response.error}")
        raise Exception(f"Failed to structure panel data: {response.error}")


async def process_pdf(pdf_path: str, output_folder: str, client) -> Tuple[str, Dict[str, Any]]:
    """
    Process a PDF file and save the structured data.
    
    Args:
        pdf_path: Path to the PDF file
        output_folder: Folder to save the output
        client: OpenAI client
        
    Returns:
        Tuple of (raw_content, structured_data)
    """
    from services.storage_service import FileSystemStorage
    
    print(f"Processing PDF: {pdf_path}")
    extractor = PyMuPdfExtractor()
    storage = FileSystemStorage()
    
    # Extract content
    extraction_result = await extractor.extract(pdf_path)
    if not extraction_result.success:
        raise Exception(f"Failed to extract content: {extraction_result.error}")
    
    # Convert to the format expected by structure_panel_data
    raw_content = ""
    raw_content += "TEXT:\n" + extraction_result.raw_text + "\n"
    for table in extraction_result.tables:
        raw_content += "TABLE:\n"
        raw_content += table["content"] + "\n"
    
    # Structure data
    structured_data = await structure_panel_data(client, raw_content)
    
    # Save the result
    panel_name = structured_data.get('panel_name', 'unknown_panel').replace(" ", "_").lower()
    filename = f"{panel_name}_electric_panel.json"
    filepath = os.path.join(output_folder, filename)
    
    await storage.save_json(structured_data, filepath)
    
    print(f"Saved structured panel data: {filepath}")
    return raw_content, structured_data
```

### utils/drawing_processor.py (updated)
```python
"""
Drawing processing utilities that leverage the AI service.
"""
from typing import Dict, Any
import logging

from services.ai_service import DrawingAiService, ModelType, AiResponse

# Drawing instructions remain the same
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
    2. Room finish schedules
    3. Door/window details
    4. Wall types
    5. Architectural notes
    Ensure all rooms are captured and properly structured in the JSON output.
    """,
    "General": "Organize all relevant data into logical categories based on content type."
}


async def process_drawing(raw_content: str, drawing_type: str, client) -> str:
    """
    Use GPT to parse PDF text + table data into structured JSON
    based on the drawing type.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        client: OpenAI client
        
    Returns:
        Structured JSON as a string
    """
    try:
        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
        
        # Process the drawing
        response: AiResponse[Dict[str, Any]] = await ai_service.process_drawing(
            raw_content=raw_content,
            drawing_type=drawing_type,
            temperature=0.2,
            max_tokens=16000,
            model_type=ModelType.GPT_4O_MINI
        )
        
        if response.success:
            return response.content
        else:
            logging.error(f"Error processing {drawing_type} drawing: {response.error}")
            raise Exception(f"Error processing {drawing_type} drawing: {response.error}")
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing: {str(e)}")
        raise
```

### utils/api_utils.py (updated)
```python
"""
API utilities for making safe API calls.
"""
import asyncio
import logging
import random
from typing import Dict, Any

from services.ai_service import AiRateLimitError, AiConnectionError, AiResponseError

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


async def async_safe_api_call(client, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely call the OpenAI API with retries and backoff.
    This is a legacy function. Consider using the AI service instead.
    
    Args:
        client: OpenAI client
        *args: Positional arguments for the API call
        **kwargs: Keyword arguments for the API call
        
    Returns:
        API response
        
    Raises:
        Exception: If the API call fails after maximum retries
    """
    retries = 0
    delay = 1  # initial backoff

    while retries < MAX_RETRIES:
        try:
            return await client.chat.completions.create(*args, **kwargs)
        except Exception as e:
            if "rate limit" in str(e).lower():
                logging.warning(f"Rate limit hit, retrying in {delay} seconds...")
                retries += 1
                delay = min(delay * 2, 60)  # cap backoff at 60s
                await asyncio.sleep(delay + random.uniform(0, 1))  # add jitter
            else:
                logging.error(f"API call failed: {e}")
                await asyncio.sleep(RETRY_DELAY)
                retries += 1

    logging.error("Max retries reached for API call")
    raise Exception("Failed to make API call after maximum retries")
```

### utils/logging_utils.py (updated)
```python
"""
Logging utilities with structured logging support.
"""
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredLogger:
    """
    Logger that produces structured log messages.
    """
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
        
    def add_context(self, **kwargs):
        """Add context to all log messages."""
        self.context.update(kwargs)
        
    def info(self, message: str, **kwargs):
        """Log an info message with structured data."""
        self._log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        """Log a warning message with structured data."""
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        """Log an error message with structured data."""
        self._log(logging.ERROR, message, **kwargs)
        
    def _log(self, level: int, message: str, **kwargs):
        """Internal method to log a message with context."""
        log_data = {**self.context, **kwargs, "message": message}
        self.logger.log(level, json.dumps(log_data))


def setup_logging(output_folder: str) -> None:
    """
    Configure and initialize logging for the application.
    Creates a 'logs' folder in the output directory.
    
    Args:
        output_folder: Folder to store log files
    """
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"process_log_{timestamp}.txt")
    
    # Configure basic logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add console handler for visibility
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    
    print(f"Logging to: {log_file}")


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """
    Get a structured logger with the given name and context.
    
    Args:
        name: Logger name
        context: Optional context dictionary
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name, context)
```

### utils/pdf_utils.py (updated)
```python
"""
PDF processing utilities that directly use PyMuPDF.
This module is being phased out in favor of the extraction service.
"""
import pymupdf as fitz
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Consider using the extraction service instead.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text
    """
    logger.info(f"Starting text extraction for {file_path}")
    try:
        with fitz.open(file_path) as doc:
            logger.info(f"Successfully opened {file_path}")
            text = ""
            for i, page in enumerate(doc):
                logger.info(f"Processing page {i+1} of {len(doc)}")
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logger.warning(f"No text extracted from page {i+1}")
        
        if not text:
            logger.warning(f"No text extracted from {file_path}")
        else:
            logger.info(f"Successfully extracted text from {file_path}")
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        raise


def extract_images(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract images from a PDF file using PyMuPDF.
    Consider using the extraction service instead.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of image metadata dictionaries
    """
    try:
        images = []
        with fitz.open(file_path) as doc:
            for page_index, page in enumerate(doc):
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # PyMuPDF doesn't directly provide bounding box for images
                    # We'd need to process rect information from the page
                    # For compatibility, we'll create a similar structure to pdfplumber
                    
                    images.append({
                        'page': page_index + 1,
                        'bbox': (0, 0, base_image["width"], base_image["height"]),  # Placeholder bbox
                        'width': base_image["width"],
                        'height': base_image["height"],
                        'type': base_image["ext"]  # Image extension/type
                    })
        
        logger.info(f"Extracted {len(images)} images from {file_path}")
        return images
    except Exception as e:
        logger.error(f"Error extracting images from {file_path}: {str(e)}")
        raise


def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata from a PDF file using PyMuPDF.
    Consider using the extraction service instead.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        PDF metadata dictionary
    """
    try:
        with fitz.open(file_path) as doc:
            # Convert PyMuPDF metadata format to match pdfplumber's format
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creationDate": doc.metadata.get("creationDate", ""),
                "modDate": doc.metadata.get("modDate", "")
            }
        logger.info(f"Successfully extracted metadata from {file_path}")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
        raise
```

### processing/file_processor.py (updated)
```python
"""
File processor module that handles individual file processing.
"""
import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from typing import Dict, Any, Optional

from services.extraction_service import PyMuPdfExtractor
from services.ai_service import DrawingAiService, ModelType
from services.storage_service import FileSystemStorage
from utils.drawing_processor import DRAWING_INSTRUCTIONS, process_drawing
from templates.room_templates import process_architectural_drawing

# Load environment variables
load_dotenv()


def is_panel_schedule(file_name: str, raw_content: str) -> bool:
    """
    Determine if a PDF is likely an electrical panel schedule
    based solely on the file name (no numeric or content checks).
    
    Args:
        file_name: Name of the PDF file
        raw_content: (Unused) Extracted text content from the PDF
        
    Returns:
        True if the file name contains panel-schedule keywords
    """
    panel_keywords = [
        "electrical panel schedule",
        "panel schedule",
        "panel schedules",
        "power schedule",
        "lighting schedule",
        # Hyphenated versions:
        "electrical-panel-schedule",
        "panel-schedule",
        "panel-schedules",
        "power-schedule",
        "lighting-schedule"
    ]
    file_name_lower = file_name.lower()
    return any(keyword in file_name_lower for keyword in panel_keywords)


async def process_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str,
    templates_created: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Process a single PDF asynchronously:
    1) Extract text/tables with PyMuPDF
    2) Use GPT to parse/structure the content
    3) Save JSON output
    
    Args:
        pdf_path: Path to the PDF file
        client: OpenAI client
        output_folder: Output folder for processed files
        drawing_type: Type of drawing
        templates_created: Dictionary tracking created templates
        
    Returns:
        Processing result dictionary
    """
    file_name = os.path.basename(pdf_path)
    logger = logging.getLogger(__name__)
    
    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)  # Start
            
            # Initialize services
            extractor = PyMuPdfExtractor(logger)
            storage = FileSystemStorage(logger)
            ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS, logger)
            
            # Extract text and tables
            extraction_result = await extractor.extract(pdf_path)
            if not extraction_result.success:
                pbar.update(100)
                logger.error(f"Extraction failed for {pdf_path}: {extraction_result.error}")
                return {"success": False, "error": extraction_result.error, "file": pdf_path}
            
            # Convert to raw_content format expected by process_drawing
            raw_content = ""
            raw_content += extraction_result.raw_text
            for table in extraction_result.tables:
                raw_content += f"\nTABLE:\n{table['content']}\n"
            
            pbar.update(20)  # PDF text/tables extracted
            
            # Process with AI
            ai_response = await ai_service.process_drawing(
                raw_content=raw_content,
                drawing_type=drawing_type,
                temperature=0.2,
                max_tokens=16000,
                model_type=ModelType.GPT_4O_MINI
            )
            
            pbar.update(40)  # GPT processing done
            
            # Create output directory
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)
            
            # Handle AI response
            if ai_response.success and ai_response.parsed_content:
                output_filename = os.path.splitext(file_name)[0] + '_structured.json'
                output_path = os.path.join(type_folder, output_filename)
                
                # Save the structured JSON
                await storage.save_json(ai_response.parsed_content, output_path)
                
                pbar.update(20)  # JSON saved
                logger.info(f"Successfully processed and saved: {output_path}")
                
                # If Architectural, generate room templates
                if drawing_type == 'Architectural':
                    result = process_architectural_drawing(ai_response.parsed_content, pdf_path, type_folder)
                    templates_created['floor_plan'] = True
                    logger.info(f"Created room templates: {result}")
                
                pbar.update(10)  # Finishing
                return {"success": True, "file": output_path, "panel_schedule": False}
                
            else:
                pbar.update(100)
                error_message = ai_response.error or "Unknown AI processing error"
                logger.error(f"AI processing error for {pdf_path}: {error_message}")
                
                # Save the raw response for debugging
                raw_output_filename = os.path.splitext(file_name)[0] + '_raw_response.json'
                raw_output_path = os.path.join(type_folder, raw_output_filename)
                
                await storage.save_text(ai_response.content, raw_output_path)
                
                logger.warning(f"Saved raw API response to {raw_output_path}")
                return {"success": False, "error": error_message, "file": pdf_path}
        
        except json.JSONDecodeError as e:
            pbar.update(100)
            logger.error(f"JSON parsing error for {pdf_path}: {str(e)}")
            return {"success": False, "error": f"Failed to parse JSON: {str(e)}", "file": pdf_path}
            
        except Exception as e:
            pbar.update(100)
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}
```

### processing/batch_processor.py (updated)
```python
"""
Batch processor module that handles batch processing of files.
"""
import time
import asyncio
import logging
from typing import List, Dict, Any

from processing.file_processor import process_pdf_async
from utils.constants import get_drawing_type

from config.settings import API_RATE_LIMIT, TIME_WINDOW


async def process_batch_async(
    batch: List[str],
    client,
    output_folder: str,
    templates_created: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """
    Given a batch of PDF file paths, process each one asynchronously,
    respecting the API rate limit (API_RATE_LIMIT calls per TIME_WINDOW).
    
    Args:
        batch: List of PDF file paths
        client: OpenAI client
        output_folder: Output folder for processed files
        templates_created: Dictionary tracking created templates
        
    Returns:
        List of processing results
    """
    tasks = []
    start_time = time.time()
    logger = logging.getLogger(__name__)

    for index, pdf_file in enumerate(batch):
        # Rate-limit control
        if index > 0 and index % API_RATE_LIMIT == 0:
            elapsed = time.time() - start_time
            if elapsed < TIME_WINDOW:
                wait_time = TIME_WINDOW - elapsed
                logger.info(f"Rate limiting: Waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
            start_time = time.time()
        
        drawing_type = get_drawing_type(pdf_file)
        tasks.append(
            process_pdf_async(
                pdf_path=pdf_file,
                client=client,
                output_folder=output_folder,
                drawing_type=drawing_type,
                templates_created=templates_created
            )
        )
    
    return await asyncio.gather(*tasks)
```

### processing/job_processor.py (updated)
```python
"""
Job processor module that orchestrates processing of a job site.
"""
import os
import logging
import asyncio
from tqdm.asyncio import tqdm
from typing import List, Dict, Any

from utils.file_utils import traverse_job_folder
from processing.batch_processor import process_batch_async
from services.extraction_service import ExtractionResult
from config.settings import BATCH_SIZE


async def process_job_site_async(job_folder: str, output_folder: str, client) -> None:
    """
    Orchestrates processing of a 'job site,' i.e., an entire folder of PDF files.
    
    Args:
        job_folder: Input folder containing PDF files
        output_folder: Output folder for processed files
        client: OpenAI client
    """
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = traverse_job_folder(job_folder)
    logger.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    
    if not pdf_files:
        logger.warning("No PDF files found. Please check the input folder.")
        return
    
    templates_created = {"floor_plan": False}
    batch_size = BATCH_SIZE
    total_batches = (len(pdf_files) + batch_size - 1) // batch_size
    
    all_results = []
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} of {total_batches}")
            
            batch_results = await process_batch_async(batch, client, output_folder, templates_created)
            all_results.extend(batch_results)
            
            successes = [r for r in batch_results if r['success']]
            failures = [r for r in batch_results if not r['success']]
            
            overall_pbar.update(len(batch))
            logger.info(f"Batch completed. Successes: {len(successes)}, Failures: {len(failures)}")
            
            for failure in failures:
                logger.error(f"Failed to process {failure['file']}: {failure['error']}")

    # Summarize results
    successes = [r for r in all_results if r['success']]
    failures = [r for r in all_results if not r['success']]
    
    logger.info(f"Processing complete. Total successes: {len(successes)}, Total failures: {len(failures)}")
    if failures:
        logger.warning("Failed files:")
        for failure in failures:
            logger.warning(f"  {failure['file']}: {failure['error']}")
```

### config/settings.py (updated)
```python
"""
Application settings loaded from environment variables.
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any, Optional

# Load environment variables from .env file
load_dotenv()

# OpenAI API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment variables")

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Processing Configuration
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '10'))
API_RATE_LIMIT = int(os.getenv('API_RATE_LIMIT', '60'))
TIME_WINDOW = int(os.getenv('TIME_WINDOW', '60'))

# Template Configuration
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')

# Additional configuration settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

# Function to get all settings as a dictionary
def get_all_settings() -> Dict[str, Any]:
    """
    Get all settings as a dictionary.
    
    Returns:
        Dictionary of all settings
    """
    return {
        "OPENAI_API_KEY": "***REDACTED***" if OPENAI_API_KEY else None,
        "LOG_LEVEL": LOG_LEVEL,
        "BATCH_SIZE": BATCH_SIZE,
        "API_RATE_LIMIT": API_RATE_LIMIT,
        "TIME_WINDOW": TIME_WINDOW,
        "TEMPLATE_DIR": TEMPLATE_DIR,
        "DEBUG_MODE": DEBUG_MODE
    }
```

## Updated requirements.txt
```txt
aiohappyeyeballs==2.4.4
aiohttp==3.11
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.8.0
asyncio>=3.4.3
attrs==24.3.0
azure-ai-documentintelligence==1.0.0
azure-core>=1.32.0
certifi==2024.12.14
cffi==1.17.1
charset-normalizer==3.4.1
cryptography==44.0.0
distro==1.9.0
frozenlist==1.4.1
h11==0.14.0
httpcore==1.0.7
httpx==0.28.1
idna==3.10
jiter==0.8.2
multidict==6.1.0
openai==1.59.8
pillow==10.4.0
pycparser==2.22
pydantic==2.10.5
pydantic_core==2.27.2
PyMuPDF==1.24.11
pypdfium2==4.30.0
python-dotenv==1.0.1
requests==2.32.3
sniffio==1.3.1
tenacity==8.2.3
tqdm==4.66.5
typing_extensions==4.12.2
urllib3==2.2.3
Wand==0.6.13
yarl==1.17.0
tiktoken==0.6.0
aiofiles>=23.2.1
```

## Main.py (updated)
```python
"""
Main application entry point.
"""
import os
import sys
import asyncio
import logging

from openai import AsyncOpenAI
from config.settings import OPENAI_API_KEY, get_all_settings
from utils.logging_utils import setup_logging
from processing.job_processor import process_job_site_async

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_folder> [output_folder]")
        sys.exit(1)
    
    job_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(job_folder, "output")
    
    if not os.path.exists(job_folder):
        print(f"Error: Input folder '{job_folder}' does not exist.")
        sys.exit(1)
    
    # 1) Set up logging
    setup_logging(output_folder)
    logging.info(f"Processing files from: {job_folder}")
    logging.info(f"Output will be saved to: {output_folder}")
    logging.info(f"Application settings: {get_all_settings()}")
    
    # 2) Create OpenAI Client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # 3) Run asynchronous job processing
    asyncio.run(process_job_site_async(job_folder, output_folder, client))
```

## Migration Guide

1. Create the new `services` directory and add the three new service files.
2. Update the existing utility modules to leverage the new services.
3. Update the processing modules to use the new services.
4. Update the main.py file.
5. Update requirements.txt to include new dependencies.

## Benefits of This Refactoring

1. **Clearer Separation of Concerns**
   - Service interfaces (extraction, AI, storage) separate concerns
   - Each component has a single responsibility

2. **Improved Error Handling**
   - Specific exception types for different error scenarios
   - Consistent error handling across all components
   - Better retry logic for API calls

3. **Enhanced Testability**
   - Service interfaces can be easily mocked for testing
   - Dependency injection makes unit tests easier to write

4. **Type Safety**
   - Comprehensive type hints throughout the codebase
   - Custom domain models (ExtractionResult, AiResponse) for better type safety

5. **Better Logging**
   - Structured logging for better observability
   - Consistent logging approach across components

6. **Improved Concurrency**
   - Better handling of asynchronous operations
   - CPU-bound operations moved off the main thread

7. **Future-Proofing**
   - Easy to add new extraction methods, AI models, or storage backends
   - Clear extension points for future enhancements