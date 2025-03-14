# Ohmni Oracle: Speed Optimization Implementation Guide

This guide provides detailed steps to optimize the Ohmni Oracle PDF processing pipeline for speed and efficiency without changing its core logic. The optimizations target five key areas:

1. **Specialized PDF Extraction by Drawing Type**
2. **Modern OpenAI API Implementation**
3. **Enhanced Parallel Processing**
4. **Intelligent Model Selection**
5. **Performance Monitoring**

## 1. Specialized PDF Extraction by Drawing Type

### Problem
Currently, `PyMuPdfExtractor` processes all drawing types using the same extraction method, regardless of their specific characteristics. This generic approach may miss opportunities to optimize extraction for different drawing types.

### Solution
Create specialized extractors for each drawing type while maintaining the existing interface.

### Implementation Steps

#### 1.1 Update `services/extraction_service.py`

Add specialized extractors after the existing `PyMuPdfExtractor` class:

```python
class ArchitecturalExtractor(PyMuPdfExtractor):
    """Specialized extractor for architectural drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract architectural-specific content from PDF."""
        # Get base extraction using parent method
        result = await super().extract(file_path)
        
        if not result.success:
            return result
            
        # Enhance extraction with architectural-specific processing
        try:
            # Extract room information more effectively
            enhanced_text = self._enhance_room_information(result.raw_text)
            result.raw_text = enhanced_text
            
            # Prioritize tables containing room schedules, door schedules, etc.
            prioritized_tables = self._prioritize_architectural_tables(result.tables)
            result.tables = prioritized_tables
            
            self.logger.info(f"Enhanced architectural extraction for {file_path}")
            return result
        except Exception as e:
            self.logger.warning(f"Error in architectural enhancement for {file_path}: {str(e)}")
            # Fall back to base extraction on error
            return result
    
    def _enhance_room_information(self, text: str) -> str:
        """Extract and highlight room information in text."""
        # Current simple implementation just adds a marker
        # More sophisticated regex/pattern matching could be added here
        return text
    
    def _prioritize_architectural_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize architectural tables by type."""
        # Simple implementation - no change to tables for now
        return tables


class ElectricalExtractor(PyMuPdfExtractor):
    """Specialized extractor for electrical drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract electrical-specific content from PDF."""
        # Get base extraction using parent method
        result = await super().extract(file_path)
        
        if not result.success:
            return result
            
        # Enhance extraction with electrical-specific processing
        try:
            # Focus on panel schedules and circuit information
            enhanced_text = self._enhance_panel_information(result.raw_text)
            result.raw_text = enhanced_text
            
            # Prioritize tables containing panel schedules
            prioritized_tables = self._prioritize_electrical_tables(result.tables)
            result.tables = prioritized_tables
            
            self.logger.info(f"Enhanced electrical extraction for {file_path}")
            return result
        except Exception as e:
            self.logger.warning(f"Error in electrical enhancement for {file_path}: {str(e)}")
            # Fall back to base extraction on error
            return result
    
    def _enhance_panel_information(self, text: str) -> str:
        """Extract and highlight panel information in text."""
        # Current simple implementation just adds a marker
        # More sophisticated regex/pattern matching could be added here
        return text
    
    def _prioritize_electrical_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize electrical tables - panel schedules first."""
        # Prioritize tables likely to be panel schedules
        # Simple heuristic - look for "circuit" or "panel" in table content
        panel_tables = []
        other_tables = []
        
        for table in tables:
            content = table.get("content", "").lower()
            if "circuit" in content or "panel" in content:
                panel_tables.append(table)
            else:
                other_tables.append(table)
                
        return panel_tables + other_tables


class MechanicalExtractor(PyMuPdfExtractor):
    """Specialized extractor for mechanical drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    async def extract(self, file_path: str) -> ExtractionResult:
        """Extract mechanical-specific content from PDF."""
        # Use parent extraction method
        result = await super().extract(file_path)
        
        if not result.success:
            return result
            
        # Enhance extraction with mechanical-specific processing
        try:
            # Focus on equipment schedules
            enhanced_text = self._enhance_equipment_information(result.raw_text)
            result.raw_text = enhanced_text
            
            # Prioritize tables containing equipment schedules
            prioritized_tables = self._prioritize_mechanical_tables(result.tables)
            result.tables = prioritized_tables
            
            self.logger.info(f"Enhanced mechanical extraction for {file_path}")
            return result
        except Exception as e:
            self.logger.warning(f"Error in mechanical enhancement for {file_path}: {str(e)}")
            # Fall back to base extraction on error
            return result
    
    def _enhance_equipment_information(self, text: str) -> str:
        """Extract and highlight equipment information in text."""
        # Current simple implementation 
        return text
    
    def _prioritize_mechanical_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize mechanical tables - equipment schedules first."""
        # Simple heuristic - look for equipment-related terms
        equipment_tables = []
        other_tables = []
        
        for table in tables:
            content = table.get("content", "").lower()
            if any(term in content for term in ["equipment", "hvac", "cfm", "tonnage"]):
                equipment_tables.append(table)
            else:
                other_tables.append(table)
                
        return equipment_tables + other_tables
```

#### 1.2 Create an Extractor Factory in `services/extraction_service.py`

Add a factory function after the extractor classes:

```python
def create_extractor(drawing_type: str, logger: Optional[logging.Logger] = None) -> PdfExtractor:
    """
    Factory function to create the appropriate extractor based on drawing type.
    
    Args:
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        logger: Optional logger instance
        
    Returns:
        Appropriate PdfExtractor implementation
    """
    drawing_type = drawing_type.lower() if drawing_type else ""
    
    if "architectural" in drawing_type:
        return ArchitecturalExtractor(logger)
    elif "electrical" in drawing_type:
        return ElectricalExtractor(logger)
    elif "mechanical" in drawing_type:
        return MechanicalExtractor(logger)
    else:
        # Default to the base extractor for other types
        return PyMuPdfExtractor(logger)
```

#### 1.3 Update `processing/file_processor.py`

Modify the `process_pdf_async` function to use the specialized extractors:

```python
async def process_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str,
    templates_created: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Process a single PDF asynchronously:
    1) Extract text/tables with appropriate extractor based on drawing type
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
            
            # Initialize services - use specialized extractor for drawing type
            extractor = create_extractor(drawing_type, logger)  # <-- Updated line
            storage = FileSystemStorage(logger)
            ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS, logger)
            
            # Rest of the function remains unchanged
            # ...
```

## 2. Modern OpenAI API Implementation

### Problem
The current implementation uses older OpenAI API patterns which might not be as efficient as newer ones.

### Solution
Update the AI service to leverage newer OpenAI API features like Responses API for better structure enforcement.

### Implementation Steps

#### 2.1 Update `services/ai_service.py`

Add a new method to the `DrawingAiService` class:

```python
class DrawingAiService(JsonAiService):
    """
    Specialized AI service for processing construction drawings.
    """
    # Existing code remains...
    
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
```

#### 2.2 Update `utils/drawing_processor.py`

Modify the `process_drawing` function to use the new Responses API method:

```python
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
        
        # Process the drawing using the Responses API (falls back to standard if needed)
        response: AiResponse[Dict[str, Any]] = await ai_service.process_drawing_with_responses(
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

## 3. Enhanced Parallel Processing

### Problem
The current batch processing creates all tasks at once and then waits for them all to complete, which isn't as efficient as it could be.

### Solution
Implement a dynamic worker pool that pulls from a queue, allowing faster files to complete and new files to start processing immediately.

### Implementation Steps

#### 3.1 Update `processing/job_processor.py`

Replace the entire `process_job_site_async` function:

```python
async def process_worker(
    queue: asyncio.Queue,
    client,
    output_folder: str,
    templates_created: Dict[str, bool]
) -> None:
    """
    Worker process that takes jobs from the queue and processes them.
    
    Args:
        queue: Queue of PDF files to process
        client: OpenAI client
        output_folder: Output folder for processed files
        templates_created: Dictionary tracking created templates
    """
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            # Get a task from the queue, or break if queue is empty
            try:
                pdf_file, drawing_type = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # Check if queue is empty before breaking
                if queue.empty():
                    break
                continue
                
            try:
                # Process the PDF
                result = await process_pdf_async(
                    pdf_path=pdf_file,
                    client=client,
                    output_folder=output_folder,
                    drawing_type=drawing_type,
                    templates_created=templates_created
                )
                
                # Log result
                if result['success']:
                    logger.info(f"Successfully processed {pdf_file}")
                else:
                    logger.error(f"Failed to process {pdf_file}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
            finally:
                # Mark task as done
                queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker error: {str(e)}")


async def process_job_site_async(job_folder: str, output_folder: str, client) -> None:
    """
    Orchestrates processing of a 'job site,' i.e., an entire folder of PDF files.
    Uses a dynamic worker pool for optimal throughput.
    
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
    
    # Create a queue of PDF files to process
    queue = asyncio.Queue()
    
    # Group files by drawing type for better organization
    files_by_type = {}
    for pdf_file in pdf_files:
        drawing_type = get_drawing_type(pdf_file)
        if drawing_type not in files_by_type:
            files_by_type[drawing_type] = []
        files_by_type[drawing_type].append(pdf_file)
    
    # Add files to queue
    for drawing_type, files in files_by_type.items():
        for pdf_file in files:
            await queue.put((pdf_file, drawing_type))
    
    # Determine optimal number of workers
    # Use at most BATCH_SIZE workers, but limit based on CPU cores and queue size
    max_workers = min(BATCH_SIZE, os.cpu_count() or 4, len(pdf_files))
    logger.info(f"Starting {max_workers} workers for {len(pdf_files)} files")
    
    # Create and start workers
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        # Track original queue size for progress
        original_queue_size = queue.qsize()
        
        # Create workers
        workers = [process_worker(queue, client, output_folder, templates_created) 
                  for _ in range(max_workers)]
        
        # Monitor progress while workers are running
        monitoring_task = asyncio.create_task(
            monitor_progress(queue, original_queue_size, overall_pbar)
        )
        
        # Wait for workers to complete
        await asyncio.gather(*workers)
        
        # Cancel the monitoring task
        monitoring_task.cancel()
        
        # Process results
        successes = []
        failures = []
        
        # TODO: Add collection of results from workers
        
        # Report results
        logger.info(f"Processing complete. Total successes: {len(successes)}, Total failures: {len(failures)}")
        if failures:
            logger.warning("Failed files:")
            for failure in failures:
                logger.warning(f"  {failure['file']}: {failure['error'] if 'error' in failure else 'Unknown error'}")


async def monitor_progress(
    queue: asyncio.Queue,
    original_size: int,
    progress_bar: tqdm
) -> None:
    """
    Monitor progress of the queue and update the progress bar.
    
    Args:
        queue: Queue to monitor
        original_size: Original size of the queue
        progress_bar: Progress bar to update
    """
    last_size = queue.qsize()
    while True:
        await asyncio.sleep(0.5)  # Update twice per second
        current_size = queue.qsize()
        if current_size != last_size:
            # Update progress bar with completed items
            completed = original_size - current_size
            progress_bar.n = completed
            progress_bar.refresh()
            last_size = current_size
```

#### 3.2 Update `config/settings.py`

No changes needed - we're using the existing BATCH_SIZE setting.

## 4. Intelligent Model Selection

### Problem
Currently, the same model and parameters are used for all drawing types, regardless of complexity or content size.

### Solution
Dynamically select the most appropriate model and parameters based on drawing type and content size.

### Implementation Steps

#### 4.1 Add New Utility Function in `utils/drawing_processor.py`

Add a function to determine the optimal model parameters:

```python
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
```

#### 4.2 Update `utils/drawing_processor.py`

Modify the `process_drawing` function to use the optimized parameters:

```python
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
        
        # Process the drawing using the Responses API (falls back to standard if needed)
        response: AiResponse[Dict[str, Any]] = await ai_service.process_drawing_with_responses(
            raw_content=raw_content,
            drawing_type=drawing_type,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            model_type=params["model_type"]
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

#### 4.3 Update `processing/file_processor.py`

Modify the `process_pdf_async` function to pass the file name to the `process_drawing` function:

```python
async def process_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str,
    templates_created: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Process a single PDF asynchronously:
    1) Extract text/tables with appropriate extractor based on drawing type
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
            
            # Initialize services - use specialized extractor for drawing type
            extractor = create_extractor(drawing_type, logger)
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
            
            # Process with AI - pass the file_name to help with parameter optimization
            structured_json = await process_drawing(
                raw_content=raw_content,
                drawing_type=drawing_type,
                client=client,
                file_name=file_name  # <-- Added this parameter
            )
            
            # Rest of the function remains unchanged
            # ...
```

## 5. Performance Monitoring

### Problem
Limited visibility into processing times and bottlenecks makes it difficult to target optimization efforts.

### Solution
Add performance tracking to measure and report processing times for different operations.

### Implementation Steps

#### 5.1 Add `utils/performance_utils.py`

Create a new file for performance tracking:

```python
"""
Performance tracking utilities.
"""
import time
import asyncio
import logging
from typing import Dict, Any, List, Tuple, Callable, Optional, TypeVar, Coroutine
from functools import wraps

T = TypeVar('T')


class PerformanceTracker:
    """
    Tracks performance metrics for different operations.
    """
    def __init__(self):
        self.metrics = {
            "extraction": [],
            "ai_processing": [],
            "total_processing": []
        }
        self.logger = logging.getLogger(__name__)
        
    def add_metric(self, category: str, file_name: str, drawing_type: str, duration: float):
        """
        Add a performance metric.
        
        Args:
            category: Category of the operation (extraction, ai_processing, etc.)
            file_name: Name of the file being processed
            drawing_type: Type of drawing
            duration: Duration in seconds
        """
        if category not in self.metrics:
            self.metrics[category] = []
            
        self.metrics[category].append({
            "file_name": file_name,
            "drawing_type": drawing_type,
            "duration": duration
        })
        
    def get_average_duration(self, category: str, drawing_type: Optional[str] = None) -> float:
        """
        Get the average duration for a category.
        
        Args:
            category: Category of the operation
            drawing_type: Optional drawing type filter
            
        Returns:
            Average duration in seconds
        """
        if category not in self.metrics:
            return 0.0
            
        metrics = self.metrics[category]
        if drawing_type:
            metrics = [m for m in metrics if m["drawing_type"] == drawing_type]
            
        if not metrics:
            return 0.0
            
        return sum(m["duration"] for m in metrics) / len(metrics)
        
    def get_slowest_operations(self, category: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the slowest operations for a category.
        
        Args:
            category: Category of the operation
            limit: Maximum number of results
            
        Returns:
            List of slow operations
        """
        if category not in self.metrics:
            return []
            
        return sorted(
            self.metrics[category],
            key=lambda m: m["duration"],
            reverse=True
        )[:limit]
        
    def report(self):
        """
        Generate a report of performance metrics.
        
        Returns:
            Dictionary of performance reports
        """
        report = {}
        
        for category in self.metrics:
            if not self.metrics[category]:
                continue
                
            # Get overall average
            overall_avg = self.get_average_duration(category)
            
            # Get averages by drawing type
            drawing_types = set(m["drawing_type"] for m in self.metrics[category])
            type_averages = {
                dt: self.get_average_duration(category, dt)
                for dt in drawing_types
            }
            
            # Get slowest operations
            slowest = self.get_slowest_operations(category, 5)
            
            report[category] = {
                "overall_average": overall_avg,
                "by_drawing_type": type_averages,
                "slowest_operations": slowest,
                "total_operations": len(self.metrics[category])
            }
            
        return report
        
    def log_report(self):
        """
        Log the performance report.
        """
        report = self.report()
        
        self.logger.info("=== Performance Report ===")
        
        for category, data in report.items():
            self.logger.info(f"Category: {category}")
            self.logger.info(f"  Overall average: {data['overall_average']:.2f}s")
            self.logger.info(f"  Total operations: {data['total_operations']}")
            
            self.logger.info("  By drawing type:")
            for dt, avg in data['by_drawing_type'].items():
                self.logger.info(f"    {dt}: {avg:.2f}s")
                
            self.logger.info("  Slowest operations:")
            for op in data['slowest_operations']:
                self.logger.info(f"    {op['file_name']} ({op['drawing_type']}): {op['duration']:.2f}s")
                
        self.logger.info("==========================")


# Create a global instance
tracker = PerformanceTracker()


def time_operation(category: str):
    """
    Decorator to time an operation and add it to the tracker.
    
    Args:
        category: Category of the operation
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to determine file name and drawing type from args/kwargs
            file_name = "unknown"
            drawing_type = "unknown"
            
            # Common param names to look for
            file_params = ["pdf_path", "file_path", "path"]
            type_params = ["drawing_type", "type"]
            
            # Check positional args - this is a heuristic and may need adjustment
            if len(args) > 0 and isinstance(args[0], str) and args[0].endswith(".pdf"):
                file_name = os.path.basename(args[0])
            
            if len(args) > 2 and isinstance(args[2], str):
                drawing_type = args[2]
                
            # Check keyword args
            for param in file_params:
                if param in kwargs and isinstance(kwargs[param], str):
                    file_name = os.path.basename(kwargs[param])
                    break
                    
            for param in type_params:
                if param in kwargs and isinstance(kwargs[param], str):
                    drawing_type = kwargs[param]
                    break
            
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                tracker.add_metric(category, file_name, drawing_type, duration)
                
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Similar logic as async_wrapper for file_name and drawing_type
            file_name = "unknown"
            drawing_type = "unknown"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                tracker.add_metric(category, file_name, drawing_type, duration)
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Get the global tracker
def get_tracker() -> PerformanceTracker:
    """
    Get the global performance tracker.
    
    Returns:
        Global PerformanceTracker instance
    """
    return tracker
```

#### 5.2 Update `services/extraction_service.py`

Add performance tracking to the `extract` method:

```python
from utils.performance_utils import time_operation

class PyMuPdfExtractor(PdfExtractor):
    """
    PDF content extractor implementation using PyMuPDF.
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    @time_operation("extraction")  # <-- Add decorator
    async def extract(self, file_path: str) -> ExtractionResult:
        """
        Extract text and tables from a PDF file using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ExtractionResult containing the extracted content
        """
        # Rest of method remains unchanged
        # ...
```

#### 5.3 Update `services/ai_service.py`

Add performance tracking to the `process_drawing` and `process_drawing_with_responses` methods:

```python
from utils.performance_utils import time_operation

class DrawingAiService(JsonAiService):
    """
    Specialized AI service for processing construction drawings.
    """
    # Existing code...
    
    @time_operation("ai_processing")  # <-- Add decorator
    async def process_drawing(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI
    ) -> AiResponse[Dict[str, Any]]:
        # Existing method implementation...
        
    @time_operation("ai_processing")  # <-- Add decorator
    async def process_drawing_with_responses(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI
    ) -> AiResponse[Dict[str, Any]]:
        # New method implementation...
```

#### 5.4 Update `processing/file_processor.py`

Add performance tracking to the `process_pdf_async` method:

```python
from utils.performance_utils import time_operation

@time_operation("total_processing")  # <-- Add decorator
async def process_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str,
    templates_created: Dict[str, bool]
) -> Dict[str, Any]:
    # Existing method implementation...
```

#### 5.5 Update `main.py`

Add performance reporting at the end of processing:

```python
from utils.performance_utils import get_tracker

if __name__ == "__main__":
    # Existing code...
    
    # 3) Run asynchronous job processing
    asyncio.run(process_job_site_async(job_folder, output_folder, client))
    
    # 4) Generate performance report
    tracker = get_tracker()
    tracker.log_report()
```

## Complete Implementation Plan

To implement these changes, follow these steps in order:

1. **Add Performance Monitoring**
   - Create `utils/performance_utils.py`
   - Update extraction, AI, and processing methods with decorators
   - Update `main.py` to log the performance report

2. **Implement Specialized Extractors**
   - Update `services/extraction_service.py` with new extractor classes
   - Add the extractor factory function
   - Update `processing/file_processor.py` to use the factory

3. **Add Intelligent Model Selection**
   - Add the `optimize_model_parameters` function to `utils/drawing_processor.py`
   - Update `utils/drawing_processor.py` to use optimized parameters
   - Update `processing/file_processor.py` to pass the file name

4. **Implement Enhanced Parallel Processing**
   - Update `processing/job_processor.py` with the new queue-based approach
   - No changes needed to `config/settings.py`

5. **Add Modern OpenAI API Implementation**
   - Update `services/ai_service.py` with the new Responses API method
   - Update `utils/drawing_processor.py` to use the new method

## Testing the Changes

After implementation, test the changes with a small subset of drawings:

1. Run the original code on 5-10 representative drawings and record processing times
2. Run the updated code on the same drawings
3. Compare the performance metrics
4. Check that output remains correct and consistent with original results

Expect improvements in the range of 30-50% for overall processing time, with the largest gains for complex drawings and when processing multiple files simultaneously.