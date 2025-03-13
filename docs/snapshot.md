<file_map>
/Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template
├── config
│   └── settings.py
├── processing
│   ├── __init__.py
│   ├── batch_processor.py
│   ├── file_processor.py
│   └── job_processor.py
├── services
│   ├── __init__.py
│   ├── ai_service.py
│   ├── extraction_service.py
│   └── storage_service.py
├── templates
│   ├── __init__.py
│   ├── a_rooms_template.json
│   ├── e_rooms_template.json
│   └── room_templates.py
├── tests
│   └── test_pdf_processing.py
├── utils
│   ├── __init__.py
│   ├── api_utils.py
│   ├── constants.py
│   ├── drawing_processor.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   ├── pdf_processor.py
│   └── pdf_utils.py
├── .cursorrules
├── .env.example
├── main.py
├── README.md
└── requirements.txt

</file_map>

<file_contents>
File: config/settings.py
```py
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

File: processing/batch_processor.py
```py
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

File: processing/file_processor.py
```py
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

File: processing/__init__.py
```py
# Processing package initialization

```

File: services/__init__.py
```py
# Services package initialization 
```

File: services/ai_service.py
```py
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

File: services/extraction_service.py
```py
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
            
            # Check if we got any content
            if not raw_text and not tables:
                self.logger.warning(f"No content extracted from {file_path}")
                return ExtractionResult(
                    raw_text="No content could be extracted from this PDF.",
                    tables=[],
                    success=True,  # Still mark as success to continue processing
                    metadata=metadata
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
        # Use context manager to ensure document is properly closed
        with fitz.open(file_path) as doc:
            # Extract metadata first
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
            
            # Initialize containers for text and tables
            raw_text = ""
            tables = []
            
            # Process each page individually to avoid reference issues
            for i, page in enumerate(doc):
                # Add page header
                page_text = f"PAGE {i+1}:\n"
                
                # Get text directly and avoid storing the textpage object
                try:
                    page_text += page.get_text() + "\n\n"
                except Exception as e:
                    self.logger.warning(f"Error extracting text from page {i+1}: {str(e)}")
                    page_text += "[Error extracting text from this page]\n\n"
                
                # Add to overall text
                raw_text += page_text
                
                # Extract tables safely
                try:
                    # Find tables on the page
                    table_finder = page.find_tables()
                    if table_finder and hasattr(table_finder, "tables"):  # Check if tables exist
                        for j, table in enumerate(table_finder.tables):
                            # Convert table to markdown immediately to avoid reference issues
                            try:
                                table_markdown = table.to_markdown()
                                table_dict = {
                                    "page": i+1,
                                    "table_index": j,
                                    "rows": len(table_finder.tables[j].cells) if hasattr(table, "cells") else 0,
                                    "cols": len(table_finder.tables[j].cells[0]) if hasattr(table, "cells") and table.cells else 0,
                                    "content": table_markdown
                                }
                                tables.append(table_dict)
                            except Exception as e:
                                self.logger.warning(f"Error converting table {j} on page {i+1} to markdown: {str(e)}")
                                # Add a placeholder for the failed table
                                tables.append({
                                    "page": i+1,
                                    "table_index": j,
                                    "error": str(e),
                                    "content": f"[Error extracting table {j} from page {i+1}]"
                                })
                except Exception as e:
                    self.logger.warning(f"Error finding tables on page {i+1}: {str(e)}")
            
            return raw_text, tables, metadata 
```

File: templates/__init__.py
```py
# Processing package initialization

```

File: processing/job_processor.py
```py
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

File: templates/a_rooms_template.json
```json
{
    "room_id": "",
    "room_name": "",
    "walls": {
      "north": "",
      "south": "",
      "east": "",
      "west": ""
    },
    "ceiling_height": "",
    "dimensions": ""
}

```

File: templates/e_rooms_template.json
```json
{
    "room_id": "",
    "room_name": "",
    "circuits": {
        "lighting": [],
        "power": []
    },
    "light_fixtures": {
        "fixture_ids": [],
        "fixture_count": {}
    },
    "outlets": {
        "regular_outlets": 0,
        "controlled_outlets": 0
    },
    "data": 0,
    "floor_boxes": 0,
    "mechanical_equipment": [],
    "switches": {
        "type": "",
        "model": "",
        "dimming": ""
    }
} 
```

File: templates/room_templates.py
```py
import json
import os

def load_template(template_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, f"{template_name}_template.json")
    try:
        with open(template_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Template file not found: {template_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {template_path}")
        return {}

def generate_rooms_data(parsed_data, room_type):
    template = load_template(room_type)
    
    metadata = parsed_data.get('metadata', {})
    
    rooms_data = {
        "metadata": metadata,
        "project_name": metadata.get('project', ''),
        "floor_number": '',
        "rooms": []
    }
    
    parsed_rooms = parsed_data.get('rooms', [])
    
    if not parsed_rooms:
        print(f"No rooms found in parsed data for {room_type}.")
        return rooms_data

    for parsed_room in parsed_rooms:
        room_number = str(parsed_room.get('number', ''))
        room_name = parsed_room.get('name', '')
        
        if not room_number or not room_name:
            print(f"Skipping room with incomplete data: {parsed_room}")
            continue
        
        room_data = template.copy()
        room_data['room_id'] = f"Room_{room_number}"
        room_data['room_name'] = f"{room_name}_{room_number}"
        
        # Copy all fields from parsed_room to room_data
        for key, value in parsed_room.items():
            if key not in ['number', 'name']:
                room_data[key] = value
        
        rooms_data['rooms'].append(room_data)
    
    return rooms_data

def process_architectural_drawing(parsed_data, file_path, output_folder):
    """
    Process architectural drawing data (parsed JSON),
    and generate both e_rooms and a_rooms JSON outputs.
    """
    is_reflected_ceiling = "REFLECTED CEILING PLAN" in file_path.upper()
    
    floor_number = ''  # If floor number is available in the future, extract it here
    
    e_rooms_data = generate_rooms_data(parsed_data, 'e_rooms')
    a_rooms_data = generate_rooms_data(parsed_data, 'a_rooms')
    
    e_rooms_file = os.path.join(output_folder, f'e_rooms_details_floor_{floor_number}.json')
    a_rooms_file = os.path.join(output_folder, f'a_rooms_details_floor_{floor_number}.json')
    
    with open(e_rooms_file, 'w') as f:
        json.dump(e_rooms_data, f, indent=2)
    with open(a_rooms_file, 'w') as f:
        json.dump(a_rooms_data, f, indent=2)
    
    return {
        "e_rooms_file": e_rooms_file,
        "a_rooms_file": a_rooms_file,
        "is_reflected_ceiling": is_reflected_ceiling
    }

```

File: utils/__init__.py
```py
# Processing package initialization

```

File: services/storage_service.py
```py
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

File: utils/api_utils.py
```py
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

File: utils/constants.py
```py
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

```

File: utils/drawing_processor.py
```py
"""
Drawing processing utilities that leverage the AI service.
"""
from typing import Dict, Any
import logging

from services.ai_service import DrawingAiService, ModelType, AiResponse

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

File: utils/logging_utils.py
```py
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

File: utils/pdf_processor.py
```py
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

File: utils/pdf_utils.py
```py
"""
PDF processing utilities that directly use PyMuPDF.
This module is being phased out in favor of the extraction service.
"""
import pymupdf as fitz
import logging
import aiofiles
from typing import List, Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)

async def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    Consider using the extraction service instead.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text
    """
    logger.info(f"Starting text extraction for {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
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

async def extract_images(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract images from a PDF file using PyMuPDF.
    Consider using the extraction service instead.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of image metadata dictionaries
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
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

async def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata from a PDF file using PyMuPDF.
    Consider using the extraction service instead.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        PDF metadata dictionary
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
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

async def save_page_as_image(file_path: str, page_num: int, output_path: str, dpi: int = 300) -> str:
    """
    Save a PDF page as an image.
    
    Args:
        file_path: Path to the PDF file
        page_num: Page number to extract (0-based)
        output_path: Path to save the image
        dpi: DPI for the rendered image (default: 300)
        
    Returns:
        Path to the saved image
        
    Raises:
        FileNotFoundError: If the file does not exist
        IndexError: If the page number is out of range
        Exception: For any other errors during extraction
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with fitz.open(file_path) as doc:
            if page_num < 0 or page_num >= len(doc):
                raise IndexError(f"Page number {page_num} out of range (0-{len(doc)-1})")
                
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            pixmap.save(output_path)
            
            logger.info(f"Saved page {page_num} as image: {output_path}")
            return output_path
    except Exception as e:
        logger.error(f"Error saving page as image: {str(e)}")
        raise

```

File: utils/file_utils.py
```py
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def traverse_job_folder(job_folder: str) -> List[str]:
    """
    Traverse the job folder and collect all PDF files.
    """
    pdf_files = []
    try:
        for root, _, files in os.walk(job_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        logger.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    except Exception as e:
        logger.error(f"Error traversing job folder {job_folder}: {str(e)}")
    return pdf_files
```

File: main.py
```py
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

File: .env.example
```example
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_key_here

# Azure Document Intelligence Configuration
DOCUMENTINTELLIGENCE_ENDPOINT=<yourEndpoint>
DOCUMENTINTELLIGENCE_API_KEY=<yourKey>

# Optional Configuration
# LOG_LEVEL=INFO
# BATCH_SIZE=10
# API_RATE_LIMIT=60
# TIME_WINDOW=60 
```

File: .cursorrules
```cursorrules
Below is a **step-by-step** guide showing **how to remove the Azure Document Intelligence parts** from your original code, along with the **complete** resulting directory/files. The end result is a template that uses only **PyMuPDF** (and GPT/OpenAI) for PDF processing—**no Azure Document Intelligence**.

---

## 1. Remove Azure Document Intelligence references from the codebase

Your original project references Azure in a few places:

1. **`requirements.txt`**:
   - It includes `azure-ai-documentintelligence==1.0.0` and `azure-core>=1.32.0`.
2. **Environment variables** in `.env.example` (for Azure Document Intelligence).
3. **Imports and usage** in:
   - `panel_schedule_intelligence.py` (direct usage of Azure Document Intelligence)
   - `test_azure_panel.py` (testing that Azure-based flow)
   - `file_processor.py` (the fallback to Azure for panel schedules)
4. **Any references** to `DOCUMENTINTELLIGENCE_ENDPOINT` or `DOCUMENTINTELLIGENCE_API_KEY`.

**We’ll remove them** (or comment them out) so your code only relies on PyMuPDF extraction and GPT for everything.

---

## 2. Keep everything else (OpenAI/GPT usage, PyMuPDF, etc.)

Your original code:
- Uses PyMuPDF (via `pymupdf`) to read PDFs, extract text/tables.
- Uses GPT (via `openai` / `AsyncOpenAI`) to structure the data.
- Has various utility files, logging, templates, etc.

We leave **all** that intact.

---

## 3. Adjust `requirements.txt` (Optional)

Since you said these are your current requirements, you may keep them exactly as is. However, if you truly want to **remove** references to Azure, simply **delete** (or comment out) the following lines:

```
azure-ai-documentintelligence==1.0.0
azure-core>=1.32.0
```

That way, you won’t install the Azure libraries at all. Everything else can remain as-is.

---

## 4. Remove or replace code that references Azure Document Intelligence

- **`panel_schedule_intelligence.py`**: Either empty out the file or remove it entirely.  
- **`test_azure_panel.py`**: Remove Azure usage (the `PanelScheduleProcessor` calls). Possibly keep it as a test for PyMuPDF extraction.  
- **`file_processor.py`**: Remove the logic that checks for `panel_processor` and calls Azure.  
- Any mention of `DOCUMENTINTELLIGENCE_ENDPOINT` or `DOCUMENTINTELLIGENCE_API_KEY` can be removed or commented out.

Below is a **fully updated** directory structure and **all files** with the necessary adjustments. **Take your time** to review and copy what you need.

---

# Final Directory Structure (Minus Document Intelligence)

```
btcelectrician-ohmni-oracle-refined/
├── README.md
├── main.py
├── requirements.txt
├── snapshot.md
├── test_azure_panel.py
├── .cursorrules
├── .env.example
├── config/
│   ├── __init__.py
│   └── settings.py
├── processing/
│   ├── __init__.py
│   ├── batch_processor.py
│   ├── file_processor.py
│   ├── job_processor.py
│   ├── panel_schedule_intelligence.py
│   └── panel_schedule_intelligence_backup.py
├── templates/
│   ├── __init__.py
│   ├── a_rooms_template.json
│   ├── e_rooms_template.json
│   └── room_templates.py
└── utils/
    ├── __init__.py
    ├── api_utils.py
    ├── constants.py
    ├── drawing_processor.py
    ├── drawing_utils.py
    ├── file_utils.py
    ├── logging_utils.py
    ├── pdf_processor.py
    └── pdf_utils.py
```

Below are the **contents of every file** after removing the Azure Document Intelligence parts.

---

## File: `README.md`

```markdown
# Ohmni Oracle

This project processes various types of drawings (e.g., architectural, electrical, mechanical) by:
1. Extracting text from PDF files (using PyMuPDF / pdfplumber)
2. Converting it into structured JSON via GPT-4

## Installation


3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to `.env`

5. **Run**:
   ```bash
   python main.py <input_folder> [output_folder]
   ```

## Project Structure

```
btcelectrician-ohmni_oracle/
├── config/
│   ├── settings.py
│   └── .gitignore
├── processing/
│   ├── batch_processor.py
│   ├── file_processor.py
│   └── job_processor.py
├── templates/
│   ├── a_rooms_template.json
│   ├── e_rooms_template.json
│   └── room_templates.py
├── utils/
│   ├── api_utils.py
│   ├── constants.py
│   ├── drawing_processor.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   ├── pdf_processor.py
│   └── pdf_utils.py
├── .env
├── main.py
├── README.md
└── requirements.txt
```

## Features

- Processes multiple types of drawings (Architectural, Electrical, etc.)
- Extracts text and tables from PDFs
- Converts unstructured data to structured JSON
- Handles batch processing with rate limiting
- Generates room templates for architectural drawings
- Comprehensive logging and error handling

## Configuration

The following environment variables can be configured in `.env`:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `BATCH_SIZE`: Number of PDFs to process in parallel (default: 10)
- `API_RATE_LIMIT`: Maximum API calls per time window (default: 60)
- `TIME_WINDOW`: Time window in seconds for rate limiting (default: 60)

## License

[Your chosen license]
```

---

## File: `main.py`
```python
import os
import sys
import asyncio
import logging

from openai import AsyncOpenAI
from config.settings import OPENAI_API_KEY
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
    
    # 2) Create OpenAI Client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    
    # 3) Run asynchronous job processing
    asyncio.run(process_job_site_async(job_folder, output_folder, client))
```

---

## File: `requirements.txt`
*(Below is the list you provided. To remove Azure, comment out or remove the lines referencing Azure. Otherwise, you can keep them if you still need them for other reasons. Example:)*

```txt
aiohappyeyeballs==2.4.4
aiohttp==3.11
aiosignal==1.3.2
annotated-types==0.7.0
anyio==4.8.0
attrs==24.3.0
# azure-ai-documentintelligence==1.0.0
# azure-core>=1.32.0
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
tqdm==4.66.5
typing_extensions==4.12.2
urllib3==2.2.3
Wand==0.6.13
yarl==1.17.0
tiktoken==0.6.0
```

---

## File: `snapshot.md`
*(No Azure references here; unchanged.)*
```md
[Your original snapshot.md content remains here]
```

---

## File: `test_azure_panel.py`
Previously, this file tested Azure Document Intelligence. You can remove it entirely or replace its contents to test only the PyMuPDF logic.  
Below is an example that:

- Detects panel schedules by filename (using `is_panel_schedule`).
- Extracts text with **PyMuPDF**.
- Writes the result to JSON.  

```python
import os
import sys
import json
import logging
import asyncio

from dotenv import load_dotenv
from processing.file_processor import is_panel_schedule
from utils.pdf_processor import extract_text_and_tables_from_pdf

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

async def main():
    if len(sys.argv) < 2:
        print("Usage: python test_azure_panel.py <pdf_file_or_folder>")
        sys.exit(1)

    path_arg = sys.argv[1]
    if not os.path.exists(path_arg):
        print(f"Error: Path '{path_arg}' does not exist.")
        sys.exit(1)

    load_dotenv()

    pdf_files = []
    if os.path.isfile(path_arg) and path_arg.lower().endswith(".pdf"):
        pdf_files.append(path_arg)
    elif os.path.isdir(path_arg):
        for root, _, files in os.walk(path_arg):
            for f in files:
                if f.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, f))

    output_folder = os.path.join(os.getcwd(), "test_output")
    os.makedirs(output_folder, exist_ok=True)

    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)

        if is_panel_schedule(file_name, ""):
            logging.info(f"Detected panel schedule in '{file_name}'.")
            try:
                # Extract with PyMuPDF (async call)
                raw_content = await extract_text_and_tables_from_pdf(pdf_path)
                result_data = {
                    "extracted_content": raw_content,
                    "error": None
                }
                logging.info(f"Successfully processed '{file_name}'.")
            except Exception as e:
                logging.exception(f"Error: {e}")
                result_data = {"extracted_content": "", "error": str(e)}

            out_file = os.path.join(
                output_folder,
                f"{os.path.splitext(file_name)[0]}_test_panel.json"
            )
            with open(out_file, "w") as f:
                json.dump(result_data, f, indent=2)
            logging.info(f"Wrote output to '{out_file}'")

        else:
            logging.info(f"'{file_name}' is NOT flagged as a panel schedule.")

if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
```

*(Or remove this file if you no longer need it.)*

---

## File: `.cursorrules`
*(Unchanged.)*
```text
Commit Message Prefixes:
* "fix:" for bug fixes
* "feat:" for new features
* "perf:" for performance improvements
* "docs:" for documentation changes
* "style:" for formatting changes
* "refactor:" for code refactoring
* "test:" for adding missing tests
* "chore:" for maintenance tasks
Rules:
* Use lowercase for commit messages
* Keep the summary line concise
* Include description for non-obvious changes
* Reference issue numbers when applicable
Documentation

* Maintain clear README with setup instructions
* Document API interactions and data flows
* Keep manifest.json well-documented
* Don't include comments unless it's for complex logic
* Document permission requirements
Development Workflow

* Use proper version control
* Implement proper code review process
* Test in multiple environments
* Follow semantic versioning for releases
* Maintain changelog
```

---

## File: `.env.example`
Remove or comment out the Azure lines:
```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_key_here

# Optional lines for Azure - removed or commented out
# DOCUMENTINTELLIGENCE_ENDPOINT=<yourEndpoint>
# DOCUMENTINTELLIGENCE_API_KEY=<yourKey>

# Optional Configuration
# LOG_LEVEL=INFO
# BATCH_SIZE=10
# API_RATE_LIMIT=60
# TIME_WINDOW=60
```

---

## File: `config/settings.py`
No changes needed for removing Azure. It only checks `OPENAI_API_KEY`. 
```python
import os
from dotenv import load_dotenv

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
```

---

## File: `processing/__init__.py`
```python
# Processing package initialization
```

---

## File: `processing/batch_processor.py`
*(Unchanged; no Azure references.)*
```python
import time
import asyncio
import logging

from processing.file_processor import process_pdf_async
from utils.constants import get_drawing_type

API_RATE_LIMIT = 60  # Adjust if needed
TIME_WINDOW = 60     # Time window to respect the rate limit

async def process_batch_async(batch, client, output_folder, templates_created):
    """
    Given a batch of PDF file paths, process each one asynchronously,
    respecting the API rate limit (API_RATE_LIMIT calls per TIME_WINDOW).
    """
    tasks = []
    start_time = time.time()

    for index, pdf_file in enumerate(batch):
        # Rate-limit control
        if index > 0 and index % API_RATE_LIMIT == 0:
            elapsed = time.time() - start_time
            if elapsed < TIME_WINDOW:
                await asyncio.sleep(TIME_WINDOW - elapsed)
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

---

## File: `processing/file_processor.py`
**Removed** any reference to Azure or `PanelScheduleProcessor`. We keep `is_panel_schedule` as is.

```python
import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

from utils.pdf_processor import extract_text_and_tables_from_pdf
from utils.drawing_processor import process_drawing
from templates.room_templates import process_architectural_drawing

# Load environment variables
load_dotenv()

def is_panel_schedule(file_name: str, raw_content: str) -> bool:
    """
    Determine if a PDF is likely an electrical panel schedule
    based solely on the file name (no numeric or content checks).
    
    Args:
        file_name (str): Name of the PDF file
        raw_content (str): (Unused) Extracted text content from the PDF
        
    Returns:
        bool: True if the file name contains certain panel-schedule keywords
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
    pdf_path,
    client,
    output_folder,
    drawing_type,
    templates_created
):
    """
    Process a single PDF asynchronously:
    1) Extract text/tables with PyMuPDF
    2) Use GPT to parse/structure the content
    3) Save JSON output
    """
    file_name = os.path.basename(pdf_path)
    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)  # Start
            raw_content = await extract_text_and_tables_from_pdf(pdf_path)
            pbar.update(20)  # PDF text/tables extracted

            # Standard GPT processing for any drawing
            structured_json = await process_drawing(raw_content, drawing_type, client)
            pbar.update(40)  # GPT processing done
            
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            # Attempt to parse JSON response from GPT
            try:
                parsed_json = json.loads(structured_json)
                output_filename = os.path.splitext(file_name)[0] + '_structured.json'
                output_path = os.path.join(type_folder, output_filename)
                
                with open(output_path, 'w') as f:
                    json.dump(parsed_json, f, indent=2)
                
                pbar.update(20)  # JSON saved
                logging.info(f"Successfully processed and saved: {output_path}")
                
                # If Architectural, generate room templates
                if drawing_type == 'Architectural':
                    result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                    templates_created['floor_plan'] = True
                    logging.info(f"Created room templates: {result}")
                
                pbar.update(10)  # Finishing
                return {"success": True, "file": output_path, "panel_schedule": False}
            
            except json.JSONDecodeError as e:
                pbar.update(100)
                logging.error(f"JSON parsing error for {pdf_path}: {str(e)}")
                logging.info(f"Raw API response: {structured_json}")
                
                raw_output_filename = os.path.splitext(file_name)[0] + '_raw_response.json'
                raw_output_path = os.path.join(type_folder, raw_output_filename)
                
                with open(raw_output_path, 'w') as f:
                    f.write(structured_json)
                
                logging.warning(f"Saved raw API response to {raw_output_path}")
                return {"success": False, "error": "Failed to parse JSON", "file": pdf_path}
        
        except Exception as e:
            pbar.update(100)
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}
```

---

## File: `processing/job_processor.py`
*(Unchanged; no Azure references.)*
```python
import os
import logging
import asyncio
from tqdm.asyncio import tqdm

from utils.file_utils import traverse_job_folder
from processing.batch_processor import process_batch_async

async def process_job_site_async(job_folder, output_folder, client):
    """
    Orchestrates processing of a 'job site,' i.e., an entire folder of PDF files.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdf_files = traverse_job_folder(job_folder)
    logging.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    
    if not pdf_files:
        logging.warning("No PDF files found. Please check the input folder.")
        return
    
    templates_created = {"floor_plan": False}
    batch_size = 10
    total_batches = (len(pdf_files) + batch_size - 1) // batch_size
    
    all_results = []
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        for i in range(0, len(pdf_files), batch_size):
            batch = pdf_files[i:i+batch_size]
            logging.info(f"Processing batch {i//batch_size + 1} of {total_batches}")
            
            batch_results = await process_batch_async(batch, client, output_folder, templates_created)
            all_results.extend(batch_results)
            
            successes = [r for r in batch_results if r['success']]
            failures = [r for r in batch_results if not r['success']]
            
            overall_pbar.update(len(batch))
            logging.info(f"Batch completed. Successes: {len(successes)}, Failures: {len(failures)}")
            
            for failure in failures:
                logging.error(f"Failed to process {failure['file']}: {failure['error']}")

    successes = [r for r in all_results if r['success']]
    failures = [r for r in all_results if not r['success']]
    
    logging.info(f"Processing complete. Total successes: {len(successes)}, Total failures: {len(failures)}")
    if failures:
        logging.warning("Failures:")
        for failure in failures:
            logging.warning(f"  {failure['file']}: {failure['error']}")
```

---

## File: `processing/panel_schedule_intelligence.py`
Since we no longer need Azure-based processing, you can **empty** this file or remove it.  
Example placeholder:

```python
"""
Placeholder file after removing Azure Document Intelligence.
No code needed here anymore.
"""
```

---

## File: `processing/panel_schedule_intelligence_backup.py`
*(Likewise, if it’s strictly Azure code, you can remove or keep as a “backup” file.)*
```python
"""
Legacy backup file. Contains references to Azure Document Intelligence from older versions.
"""
```

---

## File: `templates/__init__.py`
```python
# Templates package initialization
```

---

## File: `templates/a_rooms_template.json`
```json
{
    "room_id": "",
    "room_name": "",
    "walls": {
      "north": "",
      "south": "",
      "east": "",
      "west": ""
    },
    "ceiling_height": "",
    "dimensions": ""
}
```

---

## File: `templates/e_rooms_template.json`
```json
{
    "room_id": "",
    "room_name": "",
    "circuits": {
      "lighting": [],
      "power": []
    },
    "light_fixtures": {
      "fixture_ids": [],
      "fixture_count": {}
    },
    "outlets": {
      "regular_outlets": 0,
      "controlled_outlets": 0
    },
    "data": 0,
    "floor_boxes": 0,
    "mechanical_equipment": [],
    "switches": {
      "type": "",
      "model": "",
      "dimming": ""
    }
}
```

---

## File: `templates/room_templates.py`
```python
import json
import os

def load_template(template_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, f"{template_name}_template.json")
    try:
        with open(template_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Template file not found: {template_path}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {template_path}")
        return {}

def generate_rooms_data(parsed_data, room_type):
    template = load_template(room_type)
    
    metadata = parsed_data.get('metadata', {})
    
    rooms_data = {
        "metadata": metadata,
        "project_name": metadata.get('project', ''),
        "floor_number": '',
        "rooms": []
    }
    
    parsed_rooms = parsed_data.get('rooms', [])
    
    if not parsed_rooms:
        print(f"No rooms found in parsed data for {room_type}.")
        return rooms_data

    for parsed_room in parsed_rooms:
        room_number = str(parsed_room.get('number', ''))
        room_name = parsed_room.get('name', '')
        
        if not room_number or not room_name:
            print(f"Skipping room with incomplete data: {parsed_room}")
            continue
        
        room_data = template.copy()
        room_data['room_id'] = f"Room_{room_number}"
        room_data['room_name'] = f"{room_name}_{room_number}"
        
        # Copy all fields from parsed_room to room_data
        for key, value in parsed_room.items():
            if key not in ['number', 'name']:
                room_data[key] = value
        
        rooms_data['rooms'].append(room_data)
    
    return rooms_data

def process_architectural_drawing(parsed_data, file_path, output_folder):
    """
    Process architectural drawing data (parsed JSON),
    and generate both e_rooms and a_rooms JSON outputs.
    """
    is_reflected_ceiling = "REFLECTED CEILING PLAN" in file_path.upper()
    
    floor_number = ''  # If floor number is available in the future, extract it here
    
    e_rooms_data = generate_rooms_data(parsed_data, 'e_rooms')
    a_rooms_data = generate_rooms_data(parsed_data, 'a_rooms')
    
    e_rooms_file = os.path.join(output_folder, f'e_rooms_details_floor_{floor_number}.json')
    a_rooms_file = os.path.join(output_folder, f'a_rooms_details_floor_{floor_number}.json')
    
    with open(e_rooms_file, 'w') as f:
        json.dump(e_rooms_data, f, indent=2)
    with open(a_rooms_file, 'w') as f:
        json.dump(a_rooms_data, f, indent=2)
    
    return {
        "e_rooms_file": e_rooms_file,
        "a_rooms_file": a_rooms_file,
        "is_reflected_ceiling": is_reflected_ceiling
    }
```

---

## File: `utils/__init__.py`
```python
# Utilities package initialization
```

---

## File: `utils/api_utils.py`
*(Unchanged; no Azure references.)*
```python
import asyncio
import logging
import random

MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

async def async_safe_api_call(client, *args, **kwargs):
    """
    Safely call the OpenAI API with retries and backoff.
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

---

## File: `utils/constants.py`
*(Unchanged; no Azure references.)*
```python
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
```

---

## File: `utils/drawing_processor.py`
*(Unchanged; the GPT logic for structuring data remains. No Azure references.)*
```python
from openai import AsyncOpenAI

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

async def process_drawing(raw_content: str, drawing_type: str, client: AsyncOpenAI):
    """
    Use GPT to parse PDF text + table data into structured JSON
    based on the drawing type.
    """
    system_message = f"""
    Parse this {drawing_type} drawing/schedule into a structured JSON format. Guidelines:
    1. For text: Extract key information, categorize elements.
    2. For tables: Preserve structure, use nested arrays/objects.
    3. Create a hierarchical structure, use consistent key names.
    4. Include metadata (drawing number, scale, date) if available.
    5. {DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"])}
    6. For all drawing types, if room information is present, always include a 'rooms' array in the JSON output, 
       with each room having at least 'number' and 'name' fields.
    Ensure the entire response is a valid JSON object.
    """
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": raw_content}
            ],
            temperature=0.2,
            max_tokens=16000,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error processing {drawing_type} drawing: {str(e)}")
        raise
```

---

## File: `utils/drawing_utils.py`
*(Unchanged; no Azure references.)*
```python
"""
Additional drawing-related helper functions.
"""
```

---

## File: `utils/file_utils.py`
*(Unchanged; no Azure references.)*
```python
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def traverse_job_folder(job_folder: str) -> List[str]:
    """
    Traverse the job folder and collect all PDF files.
    """
    pdf_files = []
    try:
        for root, _, files in os.walk(job_folder):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        logger.info(f"Found {len(pdf_files)} PDF files in {job_folder}")
    except Exception as e:
        logger.error(f"Error traversing job folder {job_folder}: {str(e)}")
    return pdf_files

def cleanup_temporary_files(output_folder: str) -> None:
    """
    Clean up any temporary files created during processing (not currently used).
    """
    pass

def get_project_name(job_folder: str) -> str:
    """
    Extract the project name from the job folder path.
    """
    return os.path.basename(job_folder)
```

---

## File: `utils/logging_utils.py`
*(Unchanged; no Azure references.)*
```python
import os
import logging
from datetime import datetime

def setup_logging(output_folder: str) -> None:
    """
    Configure and initialize logging for the application.
    Creates a 'logs' folder in the output directory.
    """
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"process_log_{timestamp}.txt")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    print(f"Logging to: {log_file}")
```

---

## File: `utils/pdf_processor.py`
*(Unchanged; uses PyMuPDF for text/table extraction, no Azure references.)*
```python
import pymupdf
import json
import os
from openai import AsyncOpenAI

async def extract_text_and_tables_from_pdf(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    all_content = ""
    for page in doc:
        text = page.get_text()
        all_content += "TEXT:\n" + text + "\n"
        
        tables = page.find_tables()
        for table in tables:
            all_content += "TABLE:\n"
            markdown = table.to_markdown()
            all_content += markdown + "\n"
    
    return all_content

async def structure_panel_data(client: AsyncOpenAI, raw_content: str) -> dict:
    prompt = f"""
    You are an expert in electrical engineering and panel schedules. 
    Please structure the following content from an electrical panel schedule into a valid JSON format. 
    The content includes both text and tables. Extract key information such as panel name, voltage, amperage, circuits, 
    and any other relevant details.
    Pay special attention to the tabular data, which represents circuit information.
    Ensure your entire response is a valid JSON object.
    Raw content:
    {raw_content}
    """
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that structures electrical panel data into JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=2000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

async def process_pdf(pdf_path: str, output_folder: str, client: AsyncOpenAI):
    print(f"Processing PDF: {pdf_path}")
    raw_content = await extract_text_and_tables_from_pdf(pdf_path)
    
    structured_data = await structure_panel_data(client, raw_content)
    
    panel_name = structured_data.get('panel_name', 'unknown_panel').replace(" ", "_").lower()
    filename = f"{panel_name}_electric_panel.json"
    filepath = os.path.join(output_folder, filename)
    
    with open(filepath, 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    print(f"Saved structured panel data: {filepath}")
    return raw_content, structured_data
```

---

# Done!

This completes the **template** you requested: all original functionality **minus Azure Document Intelligence**. You now have a version that uses **only PyMuPDF** (and GPT/OpenAI) to process PDFs. 

**Usage** remains:

```bash
# (Optional) comment out azure lines in requirements.txt
pip install -r requirements.txt
python main.py <input_folder> [output_folder]
```

You still have **panel-schedule detection** logic via `is_panel_schedule(...)`, but there is **no** Azure intelligence fallback. All PDFs (including panel schedules) will go through PyMuPDF + GPT.

> **Tip:** If you do not need to install Azure packages, remove those lines in `requirements.txt` before installation.

That’s it! You now have a **complete** codebase and steps for removing the Document Intelligence references while retaining everything else.
```

File: requirements.txt
```txt
# Async utilities
aiofiles>=23.2.1
asyncio>=3.4.3
aiohttp~=3.11
aiohappyeyeballs~=2.4.4
aiosignal~=1.3.2
anyio~=4.8.0
jiter~=0.8.2
sniffio~=1.3.1

# HTTP and API clients
httpx~=0.28.1
httpcore~=1.0.7
requests~=2.32.3
urllib3~=2.2.3
yarl~=1.17.0
frozenlist~=1.4.1
multidict~=6.1.0
h11~=0.14.0

# OpenAI
openai~=1.59.8  # For Responses API support
tiktoken~=0.6.0

# PDF processing
PyMuPDF~=1.24.11  # For enhanced text extraction formats and DataFrame support
pypdfium2~=4.30.0
Wand~=0.6.13
pandas>=2.0.0  # For PyMuPDF's Table.to_pandas() functionality

# Data validation and type handling
pydantic~=2.10.5  # Using v2 validation patterns (field_validator)
pydantic_core~=2.27.2
typing_extensions~=4.12.2
annotated-types~=0.7.0
attrs~=24.3.0

# Utilities
python-dotenv~=1.0.1
tqdm~=4.66.5
tenacity~=9.0.0  # For enhanced retry patterns
pillow~=10.4.0

# Security and crypto
certifi~=2024.12.14
cffi~=1.17.1
charset-normalizer~=3.4.1
cryptography~=44.0.0
pycparser~=2.22
distro~=1.9.0

# Azure (legacy, maintaining for backward compatibility)
azure-ai-documentintelligence==1.0.0
azure-core>=1.32.0

# Testing
pytest~=7.4.0
pytest-asyncio~=0.21.1  # For testing asynchronous functions
pytest-cov~=4.1.0  # For test coverage

```

File: README.md
```md
# Ohmni Oracle

PDF processing pipeline that extracts and structures drawing information using PyMuPDF and GPT-4.

## Features

- Processes multiple drawing types (Architectural, Electrical, etc.)
- Extracts text and tables from PDFs using PyMuPDF
- Structures data using GPT-4
- Handles batch processing with rate limiting
- Generates room templates for architectural drawings
- Comprehensive logging and error handling

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Copy `.env.example` to `.env` and add your OpenAI API key

## Usage

```bash
python main.py <input_folder> [output_folder]
```

## Configuration

Environment variables in `.env`:
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)
- `BATCH_SIZE`: PDFs to process in parallel (default: 10)
- `API_RATE_LIMIT`: Max API calls per time window (default: 60)
- `TIME_WINDOW`: Time window in seconds (default: 60)

## Output

Processed files are saved as JSON in the output folder, organized by drawing type. 
```

File: tests/test_pdf_processing.py
```py
import unittest
from utils.pdf_processor import extract_text_and_tables_from_pdf
from utils.pdf_utils import structure_panel_data

class TestPDFProcessing(unittest.IsolatedAsyncioTestCase):
    async def test_panel_schedule_extraction(self):
        test_file = "samples/panel_schedule.pdf"
        content = await extract_text_and_tables_from_pdf(test_file)
        
        self.assertIn("Main Panel", content)
        self.assertRegex(content, r"Circuit\s+\d+")
        
        structured = await structure_panel_data(client, content)
        self.assertIn("circuits", structured)
        self.assertTrue(len(structured["circuits"]) > 5) 
```
</file_contents>

