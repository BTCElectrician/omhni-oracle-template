# Ohmni Oracle: Updated System Improvement Analysis (March 2025)

## Current Architecture Overview

Ohmni Oracle is a PDF processing pipeline that extracts and structures construction drawing information using PyMuPDF and GPT-4. The system processes various drawing types (Architectural, Electrical, Mechanical, etc.), extracts text and tables, and uses GPT to structure the data into JSON format.

### Core Components:
- PDF extraction using PyMuPDF
- GPT-based text interpretation and structuring
- Batch processing with rate limiting
- Drawing type classification
- Room template generation
- Logging and error handling

## Areas for Improvement

### 1. Code Architecture and Organization

#### Current Limitations:
- Some modules have overlapping responsibilities (e.g., `pdf_processor.py` and `pdf_utils.py`)
- Lack of clear separation between extraction logic and business logic
- Limited type hints across the codebase
- No clear dependency injection pattern, making testing difficult

#### Recommended Improvements:
- Implement a cleaner service-layer architecture
- Use dependency injection for external services (OpenAI client)
- Create proper domain models for different drawing types
- Implement the repository pattern for data access

```python
# Example improved structure for extraction service
from typing import Dict, Any, List, Optional
import logging
from abc import ABC, abstractmethod

class ExtractionResult:
    """Domain model for extraction results"""
    def __init__(self, raw_content: str, tables: List[Dict[str, Any]], success: bool, error: Optional[str] = None):
        self.raw_content = raw_content
        self.tables = tables
        self.success = success
        self.error = error

class PdfExtractor(ABC):
    """Abstract base class for PDF extractors"""
    @abstractmethod
    async def extract(self, file_path: str) -> ExtractionResult:
        pass

class PyMuPdfExtractor(PdfExtractor):
    """PyMuPDF implementation of PdfExtractor"""
    async def extract(self, file_path: str) -> ExtractionResult:
        try:
            # Extraction logic using modern PyMuPDF features
            import pymupdf as fitz
            import asyncio
            
            # Use get_running_loop instead of get_event_loop (2025 AsyncIO pattern)
            loop = asyncio.get_running_loop()
            # Run CPU-bound work in executor
            result = await loop.run_in_executor(None, self._extract_content, file_path)
            raw_content, tables = result
            
            return ExtractionResult(raw_content, tables, success=True)
        except Exception as e:
            logging.error(f"Extraction error: {str(e)}")
            return ExtractionResult("", [], success=False, error=str(e))
            
    def _extract_content(self, file_path: str):
        """Extract content synchronously (runs in executor)"""
        with fitz.open(file_path) as doc:
            raw_content = ""
            tables = []
            
            for page_idx, page in enumerate(doc):
                # Enhanced text extraction with format options (2025 PyMuPDF feature)
                text = page.get_text()
                raw_content += f"PAGE {page_idx+1}:\n{text}\n\n"
                
                # Get HTML formatted content for richer information
                html_content = page.get_text("html")
                
                # Get blocks with positional data
                blocks = page.get_text("blocks")
                
                # Table extraction (experimental but functional)
                pdf_tables = page.find_tables()
                for table_idx, table in enumerate(pdf_tables):
                    markdown_table = table.to_markdown()
                    tables.append({
                        "page": page_idx + 1,
                        "table_idx": table_idx,
                        "content": markdown_table,
                        # Optionally convert to pandas DataFrame
                        # "dataframe": table.to_pandas()
                    })
            
            return raw_content, tables
```

### 2. Error Handling and Resiliency

#### Current Limitations:
- Inconsistent error handling approaches
- Limited retry mechanisms for API calls
- No circuit breaker patterns for external services
- Some exceptions are caught and logged but not properly propagated

#### Recommended Improvements:
- Implement consistent error handling strategy across modules
- Add proper retries with exponential backoff for all external API calls
- Implement circuit breaker pattern for API calls
- Create detailed error types for different failure scenarios

```python
# Example improved error handling
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from typing import Dict, Any
from requests.exceptions import RequestException

class ApiRateLimitError(Exception):
    """Raised when API rate limit is hit"""
    pass

class ApiConnectionError(Exception):
    """Raised when connection to API fails"""
    pass

class ApiResponseError(Exception):
    """Raised when API returns an error response"""
    pass

@retry(
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=1, max=60),
    retry=retry_if_exception_type((ApiRateLimitError, ApiConnectionError, RequestException)),
    reraise=True
)
async def call_gpt_api(content: str, drawing_type: str, client) -> Dict[str, Any]:
    """Call OpenAI API with retry logic"""
    try:
        # Use the new Responses API (OpenAI 2025 update)
        response = await client.responses.create(
            model="gpt-4-turbo-2025-03-09",  # Updated model name
            messages=[
                {"role": "system", "content": f"Parse this {drawing_type} drawing/schedule into structured JSON."},
                {"role": "user", "content": content}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            stream=True  # Enable streaming for compliance
        )
        
        # Handle streaming response
        full_content = ""
        async for chunk in response:
            if chunk.choices[0].delta.content is not None:
                full_content += chunk.choices[0].delta.content
        
        # Parse JSON
        import json
        return json.loads(full_content)
        
    except Exception as e:
        error_message = str(e).lower()
        if "rate limit" in error_message:
            raise ApiRateLimitError(f"Rate limit exceeded: {str(e)}")
        elif any(term in error_message for term in ["connection", "timeout", "network"]):
            raise ApiConnectionError(f"Connection error: {str(e)}")
        else:
            raise ApiResponseError(f"API error: {str(e)}")
```

### 3. Performance Optimization

#### Current Limitations:
- Sequential PDF processing within batches
- No caching mechanism for similar drawings
- Large PDFs might cause memory issues
- No chunking strategy for very large drawings

#### Recommended Improvements:
- Implement proper concurrency with asyncio for all I/O operations
- Add caching layer for processed results
- Implement memory-efficient PDF processing with chunking
- Consider using worker processes for CPU-bound tasks

```python
# Example improved concurrent processing
import asyncio
from typing import List, Dict, Any
import aiofiles

async def process_pdfs_concurrent(pdf_paths: List[str], max_concurrency: int = 5) -> List[Dict[str, Any]]:
    """Process PDFs concurrently with controlled concurrency"""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_with_semaphore(pdf_path: str) -> Dict[str, Any]:
        async with semaphore:
            return await process_single_pdf(pdf_path)
    
    tasks = [process_with_semaphore(pdf_path) for pdf_path in pdf_paths]
    # Use gather with return_exceptions=True for better error handling (2025 pattern)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results including exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            processed_results.append({
                "path": pdf_paths[i],
                "success": False,
                "error": str(result)
            })
        else:
            processed_results.append(result)
    
    return processed_results

async def process_single_pdf(pdf_path: str) -> Dict[str, Any]:
    """Process a single PDF with efficient file handling"""
    # Use aiofiles for non-blocking file operations
    async with aiofiles.open(pdf_path, 'rb') as f:
        content = await f.read()
        
    # Process the PDF content in chunks if it's large
    if len(content) > 10 * 1024 * 1024:  # 10MB threshold
        return await process_large_pdf(pdf_path, content)
    else:
        # Process normally
        extractor = PyMuPdfExtractor()
        result = await extractor.extract(pdf_path)
        return {
            "path": pdf_path, 
            "success": result.success, 
            "content": result.raw_content,
            "tables": result.tables
        }
```

### 4. Testing and Quality Assurance

#### Current Limitations:
- No visible unit or integration tests
- No mocking strategy for external dependencies
- Limited validation of extraction results
- No clear quality metrics for extraction accuracy

#### Recommended Improvements:
- Implement comprehensive unit testing with pytest
- Set up integration tests for the full pipeline
- Use mock objects for external dependencies
- Add validation mechanisms to verify extraction quality
- Implement continuous integration workflow

```python
# Example test for drawing processor using pytest-asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

# Example of mocking streaming response chunks
class MockStreamingResponse:
    def __init__(self, content):
        self.content = content
        self.current = 0
        
    def __aiter__(self):
        return self
        
    async def __anext__(self):
        if self.current >= len(self.content):
            raise StopAsyncIteration
            
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = self.content[self.current]
        self.current += 1
        return chunk

@pytest.fixture
def mock_openai_client():
    client = AsyncMock()
    # Configure for new Responses API
    streaming_response = MockStreamingResponse(['{"test":', ' "data"}'])
    client.responses.create.return_value = streaming_response
    return client

@pytest.mark.asyncio
async def test_process_drawing(mock_openai_client):
    from utils.drawing_processor import process_drawing
    
    # Given
    raw_content = "Test content"
    drawing_type = "Architectural"
    
    # When
    result = await process_drawing(raw_content, drawing_type, mock_openai_client)
    
    # Then
    assert result == '{"test": "data"}'
    mock_openai_client.responses.create.assert_called_once()
    assert mock_openai_client.responses.create.call_args[1]["model"] == "gpt-4-turbo-2025-03-09"
```

### 5. Enhancing GPT Integration

#### Current Limitations:
- Fixed prompts might not be optimal for all drawing types
- No feedback loop to improve extraction accuracy
- Limited prompt engineering for complex drawings
- No clear strategy for handling GPT context window limitations

#### Recommended Improvements:
- Implement dynamic prompt generation based on drawing content
- Create a feedback mechanism to improve prompts over time
- Use chunking strategy for large drawings that exceed context limits
- Consider using newer GPT models with larger context windows

```python
# Example improved prompt generation and chunking for large content
from typing import Dict, Any, List

def generate_dynamic_prompt(drawing_type: str, content_metrics: Dict[str, Any]) -> str:
    """Generate dynamic prompts based on content metrics"""
    base_prompt = DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"])
    
    # Adjust prompt based on content metrics
    if content_metrics.get("table_count", 0) > 5:
        base_prompt += "\nPay special attention to the multiple tables in this document."
    
    if content_metrics.get("content_length", 0) > 10000:
        base_prompt += "\nThis is a large document. Focus on extracting the most critical information."
    
    # Add drawing-specific enhancements
    if drawing_type == "Electrical" and "panel" in content_metrics.get("keywords", []):
        base_prompt += "\nCarefully extract all circuit details from panel schedules."
    
    return base_prompt

def chunk_large_content(content: str, max_chunk_size: int = 8000) -> List[str]:
    """Split large content into manageable chunks for API processing"""
    if len(content) <= max_chunk_size:
        return [content]
    
    chunks = []
    current_chunk = ""
    
    for line in content.split("\n"):
        if len(current_chunk) + len(line) + 1 > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = line + "\n"
        else:
            current_chunk += line + "\n"
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

async def process_chunked_content(content: str, drawing_type: str, client) -> Dict[str, Any]:
    """Process large content by chunking and combining results"""
    chunks = chunk_large_content(content)
    
    # Process each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_prompt = f"This is part {i+1} of {len(chunks)} of a large document. "
        if i == 0:
            chunk_prompt += "Focus on extracting metadata and overall structure."
        
        # Process the chunk
        chunk_result = await call_gpt_api(chunk, drawing_type, client)
        chunk_results.append(chunk_result)
    
    # Combine results (would need a more sophisticated merging strategy in practice)
    combined_result = {}
    for result in chunk_results:
        combined_result.update(result)
        
    return combined_result
```

### 6. Data Validation and Schema Enforcement

#### Current Limitations:
- Limited validation of structured data
- No schema enforcement for generated JSON
- No clear strategy for handling inconsistent GPT outputs
- Missing data quality checks

#### Recommended Improvements:
- Implement Pydantic models for all data structures
- Add validation layers for GPT outputs
- Create schema migration strategies as templates evolve
- Add data quality metrics

```python
# Example Pydantic models for data validation (updated for Pydantic v2)
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Dict, Any, Optional

class Room(BaseModel):
    """Room model with validation"""
    # Use ConfigDict for model configuration (Pydantic v2 pattern)
    model_config = ConfigDict(str_strip_whitespace=True)
    
    room_id: str
    room_name: str
    dimensions: Optional[str] = None
    ceiling_height: Optional[str] = None
    walls: Dict[str, str] = Field(default_factory=dict)
    
    # Use field_validator instead of validator (Pydantic v2 pattern)
    @field_validator('room_id')
    def validate_room_id(cls, v, info):
        if not v.startswith('Room_'):
            return f"Room_{v}"
        return v

class ArchitecturalDrawing(BaseModel):
    """Architectural drawing model with validation"""
    model_config = ConfigDict(extra='ignore')
    
    metadata: Dict[str, Any] = Field(default_factory=dict)
    project_name: str
    floor_number: Optional[str] = None
    rooms: List[Room] = Field(default_factory=list)
    
    @field_validator('rooms')
    def validate_unique_room_ids(cls, rooms, info):
        """Validate that all room IDs are unique"""
        room_ids = [room.room_id for room in rooms]
        if len(room_ids) != len(set(room_ids)):
            raise ValueError("Room IDs must be unique")
        return rooms
    
    # Use model_dump instead of dict() for serialization (Pydantic v2)
    def to_dict(self, exclude_none: bool = True):
        return self.model_dump(exclude_none=exclude_none)
```

### 7. Monitoring and Observability

#### Current Limitations:
- Basic logging with limited structured information
- No performance metrics collection
- No centralized error tracking
- Limited visibility into processing bottlenecks

#### Recommended Improvements:
- Implement structured logging with JSON format
- Add detailed performance metrics
- Integrate with monitoring tools (Prometheus, Grafana)
- Implement distributed tracing for complex workflows

```python
# Example structured logging
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

T = TypeVar('T')

class StructuredLogger:
    """Structured logger with standardized format"""
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def add_context(self, **kwargs):
        """Add context to all log messages"""
        self.context.update(kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data"""
        self._log(logging.INFO, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data"""
        self._log(logging.ERROR, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with context merging"""
        log_data = {**self.context, **kwargs, "message": message}
        self.logger.log(level, json.dumps(log_data))

def timing_decorator(logger: Optional[StructuredLogger] = None):
    """Decorator to measure and log function execution time"""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                if logger:
                    logger.info(
                        f"{func.__name__} execution completed",
                        execution_time=execution_time,
                        function=func.__name__
                    )
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                execution_time = time.time() - start_time
                if logger:
                    logger.info(
                        f"{func.__name__} execution completed",
                        execution_time=execution_time,
                        function=func.__name__
                    )
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator
```

### 8. Document Understanding Enhancements

#### Current Limitations:
- Limited extraction of spatial relationships in drawings
- No image processing for graphical elements
- No handling of drawing symbols and legends
- Text-based extraction misses important visual context

#### Recommended Improvements:
- Integrate OCR capabilities for image-based text
- Add computer vision algorithms for symbol recognition
- Implement spatial relationship extraction
- Create drawing-specific parsers for common notations

```python
# Example integration with computer vision for symbol recognition
from PIL import Image
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple

class SymbolDetector:
    """Detect common symbols in construction drawings"""
    def __init__(self, symbols_db_path: str):
        self.symbols = self._load_symbols(symbols_db_path)
    
    def _load_symbols(self, db_path: str) -> Dict[str, np.ndarray]:
        """Load symbol templates from database"""
        # Implementation
        return {}
    
    def detect_symbols(self, image_path: str) -> List[Dict[str, Any]]:
        """Detect symbols in an image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        results = []
        
        for symbol_name, template in self.symbols.items():
            matches = self._template_matching(image, template)
            for match in matches:
                results.append({
                    "symbol": symbol_name,
                    "confidence": match["confidence"],
                    "position": match["position"]
                })
        
        return results
    
    def _template_matching(self, image: np.ndarray, template: np.ndarray) -> List[Dict[str, Any]]:
        """Perform template matching to find symbols"""
        # Implementation
        return []

# Integration with PDF processing
async def process_drawing_with_symbols(pdf_path: str) -> Dict[str, Any]:
    """Process drawing with both text and symbol extraction"""
    # Extract text using enhanced PyMuPDF features
    extractor = PyMuPdfExtractor()
    extraction_result = await extractor.extract(pdf_path)
    
    # Convert PDF page to image for symbol detection
    # Implementation: Use PyMuPDF to render pages as images
    import pymupdf as fitz
    import tempfile
    import os
    
    symbol_results = []
    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            # Render page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_image_path = tmp.name
                pix.save(temp_image_path)
            
            # Detect symbols
            try:
                symbol_detector = SymbolDetector("path/to/symbols_db")
                symbols = symbol_detector.detect_symbols(temp_image_path)
                symbol_results.append({
                    "page": page_idx + 1,
                    "symbols": symbols
                })
            finally:
                # Clean up temp file
                if os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
    
    # Combine results
    results = {
        "text_content": extraction_result.raw_content,
        "tables": extraction_result.tables,
        "symbols": symbol_results
    }
    
    return results
```

## Implementation Roadmap

Here's a more realistic AI-accelerated implementation roadmap:

1. **Phase 1: Core Architecture and Dependency Updates (1 day)**
   - Update to current dependency versions
   - Replace deprecated AsyncIO patterns with `get_running_loop()`
   - Update Pydantic models to v2 syntax with `field_validator` and `model_dump()`
   - Implement service layer architecture with dependency injection

2. **Phase 2: OpenAI API and PyMuPDF Enhancements (1 day)**
   - Migrate to OpenAI Responses API with streaming
   - Implement PyMuPDF enhancements (HTML extraction, block format, DataFrame conversion)
   - Improve error handling and retry mechanisms
   - Add proper concurrent processing with controlled concurrency

3. **Phase 3: Data Quality and Validation (1 day)**
   - Implement Pydantic models for all data structures
   - Add validation layers for GPT outputs
   - Create schema migration strategies
   - Implement content chunking for large documents

4. **Phase 4: Testing and Observability (1 day)**
   - Set up unit testing with pytest-asyncio
   - Implement structured logging
   - Add performance metrics collection
   - Create end-to-end integration tests

5. **Phase 5: Advanced Features (Optional - as needed)**
   - Add computer vision capabilities for symbol detection
   - Implement spatial relationship extraction
   - Enhance document understanding with OCR
   - Create drawing-specific parsers for common notations

## Conclusion

The Ohmni Oracle system has a solid foundation but will benefit significantly from these improvements, especially with the 2025 updates to key dependencies like PyMuPDF, OpenAI, AsyncIO, and Pydantic. The most critical updates are the AsyncIO pattern changes (replacing `get_event_loop()` with `get_running_loop()`) and the Pydantic v2 validation patterns (using `field_validator` instead of `validator`).

These changes will create a more robust, maintainable, and effective system for processing construction drawings, with better performance, error handling, and data quality. The updated implementation takes advantage of modern features like PyMuPDF's enhanced text extraction formats, OpenAI's streaming responses, and improved concurrency patterns with AsyncIO.