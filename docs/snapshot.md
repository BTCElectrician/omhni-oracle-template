<file_map>
/Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template
├── config
│   └── settings.py
├── processing
│   ├── __init__.py
│   ├── file_processor.py
│   ├── job_processor.py
│   └── panel_schedule_processor.py
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
├── utils
│   ├── __init__.py
│   ├── constants.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   ├── pdf_processor.py
│   └── performance_utils.py
├── .env.example
├── main.py
└── requirements.txt

</file_map>

<file_contents>
File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/requirements.txt
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
openai~=1.66.3  # For Responses API support
tiktoken~=0.9.0

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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/main.py
```py
import os
import sys
import asyncio
import logging
import time

from openai import AsyncOpenAI
from config.settings import OPENAI_API_KEY, get_all_settings
from utils.logging_utils import setup_logging
from processing.job_processor import process_job_site_async
from utils.performance_utils import get_tracker

async def main_async():
    """
    Main async function to handle processing with better error handling.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_folder> [output_folder]")
        return 1
    
    job_folder = sys.argv[1]
    output_folder = sys.argv[2] if len(sys.argv) > 2 else os.path.join(job_folder, "output")
    
    if not os.path.exists(job_folder):
        print(f"Error: Input folder '{job_folder}' does not exist.")
        return 1
    
    # 1) Set up logging
    setup_logging(output_folder)
    logging.info(f"Processing files from: {job_folder}")
    logging.info(f"Output will be saved to: {output_folder}")
    logging.info(f"Application settings: {get_all_settings()}")
    
    try:
        # 2) Create OpenAI Client (v1.66.3)
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        # 3) Record start time
        start_time = time.time()
        
        # 4) Run asynchronous job processing
        await process_job_site_async(job_folder, output_folder, client)
        
        # 5) Calculate total processing time
        total_time = time.time() - start_time
        logging.info(f"Total processing time: {total_time:.2f} seconds")
        
        # 6) Generate performance report
        tracker = get_tracker()
        tracker.log_report()
        
        return 0
    except Exception as e:
        logging.error(f"Unhandled exception in main process: {str(e)}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main_async())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1) 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/config/settings.py
```py
"""
Application settings loaded from environment variables.
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any

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

# Processing Mode Configuration
USE_SIMPLIFIED_PROCESSING = os.getenv('USE_SIMPLIFIED_PROCESSING', 'false').lower() == 'true'

# Template Configuration
TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')

# Additional configuration settings
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

def get_all_settings() -> Dict[str, Any]:
    return {
        "OPENAI_API_KEY": "***REDACTED***" if OPENAI_API_KEY else None,
        "LOG_LEVEL": LOG_LEVEL,
        "BATCH_SIZE": BATCH_SIZE,
        "API_RATE_LIMIT": API_RATE_LIMIT,
        "TIME_WINDOW": TIME_WINDOW,
        "TEMPLATE_DIR": TEMPLATE_DIR,
        "DEBUG_MODE": DEBUG_MODE,
        "USE_SIMPLIFIED_PROCESSING": USE_SIMPLIFIED_PROCESSING
    }

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/__init__.py
```py
# Services package initialization 
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/extraction_service.py
```py
"""
Extraction service interface and implementations for PDF content extraction.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import asyncio
import os

import pymupdf as fitz

from utils.performance_utils import time_operation


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

    @time_operation("extraction")
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
                
                # Try block-based extraction first
                try:
                    blocks = page.get_text("blocks")
                    for block in blocks:
                        if block[6] == 0:  # Text block (type 0)
                            page_text += block[4] + "\n"
                except Exception as e:
                    self.logger.warning(f"Block extraction error on page {i+1}: {str(e)}")
                    # Fall back to regular text extraction
                    try:
                        page_text += page.get_text() + "\n\n"
                    except Exception as e2:
                        self.logger.warning(f"Error extracting text from page {i+1}: {str(e2)}")
                        page_text += "[Error extracting text from this page]\n\n"
                
                # Add to overall text
                raw_text += page_text
                
                # Extract tables safely
                try:
                    # Find tables on the page
                    table_finder = page.find_tables()
                    if table_finder and hasattr(table_finder, "tables"):
                        for j, table in enumerate(table_finder.tables):
                            try:
                                table_markdown = table.to_markdown()
                                tables.append({
                                    "page": i+1,
                                    "table_index": j,
                                    "content": table_markdown
                                })
                            except Exception as e:
                                self.logger.warning(f"Error converting table {j} on page {i+1}: {str(e)}")
                                tables.append({
                                    "page": i+1,
                                    "table_index": j,
                                    "content": f"[Error extracting table {j} from page {i+1}]"
                                })
                except Exception as e:
                    self.logger.warning(f"Error finding tables on page {i+1}: {str(e)}")
            
            return raw_text, tables, metadata

    async def save_page_as_image(self, file_path: str, page_num: int, output_path: str, dpi: int = 300) -> str:
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
        try:
            if not os.path.exists(file_path):
                self.logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            # Use run_in_executor to move CPU-bound work off the main thread
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._save_page_as_image, file_path, page_num, output_path, dpi
            )
            
            return result
        except Exception as e:
            self.logger.error(f"Error saving page as image: {str(e)}")
            raise
            
    def _save_page_as_image(self, file_path: str, page_num: int, output_path: str, dpi: int = 300) -> str:
        """
        Internal method to save a PDF page as an image.
        This method runs in a separate thread.
        
        Args:
            file_path: Path to the PDF file
            page_num: Page number to extract (0-based)
            output_path: Path to save the image
            dpi: DPI for the rendered image
            
        Returns:
            Path to the saved image
        """
        with fitz.open(file_path) as doc:
            if page_num < 0 or page_num >= len(doc):
                raise IndexError(f"Page number {page_num} out of range (0-{len(doc)-1})")
                
            page = doc[page_num]
            pixmap = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
            pixmap.save(output_path)
            
            self.logger.info(f"Saved page {page_num} as image: {output_path}")
            return output_path


class ArchitecturalExtractor(PyMuPdfExtractor):
    """Specialized extractor for architectural drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    @time_operation("extraction")
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
        # Add a marker for room information
        if "room" in text.lower() or "space" in text.lower():
            text = "ROOM INFORMATION DETECTED:\n" + text
        return text
    
    def _prioritize_architectural_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize architectural tables by type."""
        # Prioritize tables likely to be room schedules
        room_tables = []
        other_tables = []
        
        for table in tables:
            content = table.get("content", "").lower()
            if "room" in content or "space" in content or "finish" in content:
                room_tables.append(table)
            else:
                other_tables.append(table)
                
        return room_tables + other_tables


class ElectricalExtractor(PyMuPdfExtractor):
    """Specialized extractor for electrical drawings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__(logger)
    
    @time_operation("extraction")
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
        # Add a marker for panel information
        if "panel" in text.lower() or "circuit" in text.lower():
            text = "PANEL INFORMATION DETECTED:\n" + text
        return text
    
    def _prioritize_electrical_tables(self, tables: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize electrical tables - panel schedules first."""
        # Prioritize tables likely to be panel schedules
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
    
    @time_operation("extraction")
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
        # Add a marker for equipment information
        if "equipment" in text.lower() or "hvac" in text.lower() or "cfm" in text.lower():
            text = "EQUIPMENT INFORMATION DETECTED:\n" + text
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/storage_service.py
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/a_rooms_template.json
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/__init__.py
```py
# Processing package initialization

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/.env.example
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/room_templates.py
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/constants.py
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
    'Kitchen': ['K', 'KD'],
    'Specifications': ['SPEC', 'SP']
}

def get_drawing_type(filename: str) -> str:
    """
    Detect the drawing type by examining the first 1-2 letters of the filename.
    """
    basename = os.path.basename(filename).upper()
    
    # Check by prefixes FIRST
    prefix = basename.split('.')[0][:2].upper()
    for dtype, prefixes in DRAWING_TYPES.items():
        if any(prefix.startswith(p.upper()) for p in prefixes):
            return dtype
            
    # Only use Specifications as a fallback if we couldn't determine by prefix
    if "SPEC" in basename or "SPECIFICATION" in basename:
        return "Specifications"
        
    return 'General'

def get_drawing_subtype(filename: str) -> str:
    """
    Detect the drawing subtype based on keywords in the filename.
    """
    filename_lower = filename.lower()
    if "panel schedule" in filename_lower or "electrical schedule" in filename_lower:
        return "electrical_panel_schedule"
    elif "mechanical schedule" in filename_lower:
        return "mechanical_schedule"
    elif "plumbing schedule" in filename_lower:
        return "plumbing_schedule"
    elif "wall types" in filename_lower or "partition types" in filename_lower:
        return "architectural_schedule"
    else:
        return "default"
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/performance_utils.py
```py
"""
Performance tracking utilities.
"""
import time
import asyncio
import logging
import os
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
        # Define common parameter names here so they're available to both wrappers
        file_params = ["pdf_path", "file_path", "path"]
        type_params = ["drawing_type", "type"]
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Try to determine file name and drawing type from args/kwargs
            file_name = "unknown"
            drawing_type = "unknown"
            
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
            
            # Check positional args
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/__init__.py
```py
# Processing package initialization

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/templates/e_rooms_template.json
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/logging_utils.py
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/file_utils.py
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/job_processor.py
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
from utils.constants import get_drawing_type
from processing.file_processor import process_pdf_async
from services.extraction_service import ExtractionResult
from config.settings import BATCH_SIZE


async def process_worker(
    queue: asyncio.Queue,
    client,
    output_folder: str,
    templates_created: Dict[str, bool],
    results: List[Dict[str, Any]],
    worker_id: int,
    semaphore: asyncio.Semaphore
) -> None:
    """
    Enhanced worker process that takes jobs from the queue and processes them.
    Uses a semaphore to limit concurrent API calls.
    
    Args:
        queue: Queue of PDF files to process
        client: OpenAI client
        output_folder: Output folder for processed files
        templates_created: Dictionary tracking created templates
        results: List to collect processing results
        worker_id: Unique identifier for this worker
        semaphore: Semaphore to limit concurrent API calls
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Worker {worker_id} started")
    
    while True:
        try:
            # Get a task from the queue, or break if queue is empty
            try:
                pdf_file, drawing_type = await asyncio.wait_for(queue.get(), timeout=1.0)
                logger.info(f"Worker {worker_id} processing {os.path.basename(pdf_file)}")
            except asyncio.TimeoutError:
                # Check if queue is empty before breaking
                if queue.empty():
                    logger.info(f"Worker {worker_id} finishing - queue empty")
                    break
                continue
                
            try:
                # Process the PDF with timeout protection and semaphore
                try:
                    async with semaphore:
                        logger.info(f"Worker {worker_id} acquired semaphore for {os.path.basename(pdf_file)}")
                        result = await asyncio.wait_for(
                            process_pdf_async(
                                pdf_path=pdf_file,
                                client=client,
                                output_folder=output_folder,
                                drawing_type=drawing_type,
                                templates_created=templates_created
                            ),
                            timeout=600  # 10-minute timeout per file
                        )
                except asyncio.TimeoutError:
                    logger.error(f"Timeout processing {pdf_file} after 10 minutes")
                    result = {
                        "success": False,
                        "error": "Processing timed out after 10 minutes",
                        "file": pdf_file
                    }
                
                # Add result to results list
                results.append(result)
                
                # Log result
                if result['success']:
                    logger.info(f"Worker {worker_id} successfully processed {pdf_file}")
                else:
                    logger.error(f"Worker {worker_id} failed to process {pdf_file}: {result['error']}")
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error processing {pdf_file}: {str(e)}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "file": pdf_file
                })
            finally:
                # Mark task as done
                queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {str(e)}")
            # Continue to next item rather than breaking


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
    try:
        while True:
            await asyncio.sleep(0.5)  # Update twice per second
            current_size = queue.qsize()
            if current_size != last_size:
                # Update progress bar with completed items
                completed = original_size - current_size
                progress_bar.n = completed
                progress_bar.refresh()
                last_size = current_size
    except asyncio.CancelledError:
        # This is expected when the task is cancelled
        pass
    except Exception as e:
        logging.error(f"Monitor error: {str(e)}")


async def process_job_site_async(job_folder: str, output_folder: str, client) -> None:
    """
    Orchestrates processing of a 'job site,' i.e., an entire folder of PDF files.
    Uses prioritized queue processing with file size sorting and concurrency control.
    
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
    
    # Group and prioritize files by drawing type
    files_by_type = {}
    for pdf_file in pdf_files:
        drawing_type = get_drawing_type(pdf_file)
        if drawing_type not in files_by_type:
            files_by_type[drawing_type] = []
        files_by_type[drawing_type].append(pdf_file)
    
    # Define processing priority order
    priority_order = [
        'Architectural',  # Process architectural drawings first
        'Electrical',     # Then electrical
        'Mechanical',     # Then mechanical
        'Plumbing',       # Then plumbing
        'General'         # Other drawings last
    ]
    
    # Add files to queue in priority order, sorting by file size within each group
    for drawing_type in priority_order:
        if drawing_type in files_by_type:
            # Sort files by size (smallest first)
            files = sorted(files_by_type[drawing_type], key=lambda x: os.path.getsize(x))
            logger.info(f"Queueing {len(files)} {drawing_type} drawings (sorted by size)")
            for pdf_file in files:
                await queue.put((pdf_file, drawing_type))
    
    # Add any remaining file types not explicitly prioritized (also sorted by size)
    for drawing_type, files in files_by_type.items():
        if drawing_type not in priority_order:
            # Sort files by size (smallest first)
            files = sorted(files, key=lambda x: os.path.getsize(x))
            logger.info(f"Queueing {len(files)} {drawing_type} drawings (sorted by size)")
            for pdf_file in files:
                await queue.put((pdf_file, drawing_type))
    
    # Create a semaphore to limit concurrent API calls to 5
    semaphore = asyncio.Semaphore(5)
    logger.info("Using semaphore to limit concurrent API calls to 5")
    
    # Determine optimal number of workers
    max_workers = min(BATCH_SIZE, os.cpu_count() or 4, len(pdf_files))
    logger.info(f"Starting {max_workers} workers for {len(pdf_files)} files")
    
    # Shared list to collect results
    all_results = []
    
    # Create and start workers
    with tqdm(total=len(pdf_files), desc="Overall Progress") as overall_pbar:
        # Track original queue size for progress
        original_queue_size = queue.qsize()
        
        # Create workers with IDs
        workers = []
        for i in range(max_workers):
            worker = asyncio.create_task(
                process_worker(queue, client, output_folder, templates_created, all_results, i+1, semaphore)
            )
            workers.append(worker)
        
        # Monitor progress while workers are running
        monitoring_task = asyncio.create_task(
            monitor_progress(queue, original_queue_size, overall_pbar)
        )
        
        # Wait for all tasks to be processed
        await queue.join()
        
        # Cancel worker tasks
        for worker in workers:
            worker.cancel()
        
        # Cancel the monitoring task
        monitoring_task.cancel()
        
        # Wait for all tasks to be cancelled
        await asyncio.gather(*workers, monitoring_task, return_exceptions=True)
        
        # Summarize results
        successes = [r for r in all_results if r.get('success', False)]
        failures = [r for r in all_results if not r.get('success', False)]
        
        logger.info(f"Processing complete. Total successes: {len(successes)}, Total failures: {len(failures)}")
        if failures:
            logger.warning("Failed files:")
            for failure in failures:
                logger.warning(f"  {failure['file']}: {failure.get('error', 'Unknown error')}")

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/__init__.py
```py
# Processing package initialization

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/panel_schedule_processor.py
```py
import os
import json
import logging
from typing import Dict, Any, List
from tqdm.asyncio import tqdm

from services.extraction_service import PyMuPdfExtractor
from services.storage_service import FileSystemStorage
from services.ai_service import DrawingAiService, AiRequest, ModelType

# If you have a performance decorator, you can add it here if desired
# from utils.performance_utils import time_operation

def split_text_into_chunks(text: str, chunk_size: int = None) -> List[str]:
    """
    Returns the full text as a single chunk with no splitting.
    
    Args:
        text: The text to process
        chunk_size: Ignored parameter (kept for backwards compatibility)
        
    Returns:
        List with a single item containing the full text
    """
    return [text]

def normalize_panel_data_fields(panel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unify synonyms in GPT's JSON:
      - 'description', 'loadType' => 'load_name'
      - 'ocp', 'amperage', 'breaker_size' => 'trip'
      - 'circuit_no', 'circuit_number' => 'circuit'
    """
    circuits = panel_data.get("circuits", [])
    new_circuits = []

    for cdict in circuits:
        c = dict(cdict)
        # load_name synonyms
        if "description" in c and "load_name" not in c:
            c["load_name"] = c.pop("description")
        if "loadType" in c and "load_name" not in c:
            c["load_name"] = c.pop("loadType")

        # trip synonyms
        if "ocp" in c and "trip" not in c:
            c["trip"] = c.pop("ocp")
        if "breaker_size" in c and "trip" not in c:
            c["trip"] = c.pop("breaker_size")
        if "amperage" in c and "trip" not in c:
            c["trip"] = c.pop("amperage")
        if "amp" in c and "trip" not in c:
            c["trip"] = c.pop("amp")

        # circuit synonyms
        if "circuit_no" in c and "circuit" not in c:
            c["circuit"] = c.pop("circuit_no")
        if "circuit_number" in c and "circuit" not in c:
            c["circuit"] = c.pop("circuit_number")

        new_circuits.append(c)

    panel_data["circuits"] = new_circuits
    return panel_data

async def process_panel_schedule_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str
) -> Dict[str, Any]:
    """
    Specialized function for panel schedules:
    1) Extract with PyMuPDF
    2) If any tables found, parse them chunk-by-chunk
    3) If no tables, fallback to raw text chunking
    4) Merge partial results & synonyms
    5) Save final JSON to output_folder/drawing_type
    """
    logger = logging.getLogger(__name__)
    file_name = os.path.basename(pdf_path)

    extractor = PyMuPdfExtractor(logger=logger)
    storage = FileSystemStorage(logger=logger)

    with tqdm(total=100, desc=f"[PanelSchedules] {file_name}", leave=False) as pbar:
        try:
            extraction_result = await extractor.extract(pdf_path)
            pbar.update(10)
            if not extraction_result.success:
                err = f"Extraction failed: {extraction_result.error}"
                logger.error(err)
                return {"success": False, "error": err, "file": pdf_path}

            tables = extraction_result.tables
            raw_text = extraction_result.raw_text

            if tables:
                logger.info(f"Found {len(tables)} table(s) in {file_name}. Using table-based parsing.")
                panels_data = await _parse_tables_chunked(tables, client, logger)
            else:
                logger.warning(f"No tables found in {file_name}—fallback to raw text approach.")
                panels_data = await _fallback_raw_text(raw_text, client, logger)

            pbar.update(60)
            if not panels_data:
                logger.warning(f"No panel data extracted from {file_name}.")
                return {"success": True, "file": pdf_path, "panel_schedule": True, "data": []}

            # Now we place the final JSON in output_folder/drawing_type
            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            base_name = os.path.splitext(file_name)[0]
            output_filename = f"{base_name}_panel_schedules.json"
            output_path = os.path.join(type_folder, output_filename)

            await storage.save_json(panels_data, output_path)
            pbar.update(30)

            logger.info(f"Saved panel schedules to {output_path}")
            pbar.update(10)

            return {
                "success": True,
                "file": output_path,
                "panel_schedule": True,
                "data": panels_data
            }

        except Exception as ex:
            pbar.update(100)
            logger.error(f"Unhandled error processing panel schedule {pdf_path}: {str(ex)}")
            return {"success": False, "error": str(ex), "file": pdf_path}

async def _parse_tables_chunked(tables: List[Dict[str, Any]], client, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Process each table's markdown as a complete unit - no chunking.
    Returns a list of panel objects (one per table).
    """
    from services.ai_service import AiRequest

    ai_service = DrawingAiService(client, drawing_instructions={}, logger=logger)
    all_panels = []

    system_prompt = """
You are an advanced electrical-engineering assistant. I'm giving you a table from a panel schedule in Markdown form.
Synonyms:
  - 'Load', 'Load Type', 'Description' => 'load_name'
  - 'Trip', 'OCP', 'Breaker Size', 'Amperage', 'Amp' => 'trip'
  - 'Circuit No', 'Circuit Number' => 'circuit'
Return valid JSON like:
{
  "panel_name": "...",
  "panel_metadata": {},
  "circuits": [
    { "circuit": "...", "load_name": "...", "trip": "...", "poles": "...", ... }
  ]
}
Do not skip any circuit rows. If missing data, leave blank.
    """.strip()

    for i, tbl_info in enumerate(tables):
        table_md = tbl_info["content"]
        if not table_md.strip():
            logger.debug(f"Skipping empty table {i}.")
            continue

        # Process the entire table as a single unit - NO CHUNKING
        user_text = f"FULL TABLE:\n{table_md}"

        request = AiRequest(
            content=user_text,
            model_type=ModelType.GPT_4O_MINI,
            temperature=0.2,
            max_tokens=3000,
            system_message=system_prompt
        )

        response = await ai_service.process(request)
        if not response.success or not response.content:
            logger.warning(f"GPT parse error on table {i}: {response.error}")
            continue

        try:
            panel_json = json.loads(response.content)
            # Normalize synonyms
            panel_json = normalize_panel_data_fields(panel_json)
            # If we found circuits or a panel name, add it
            if panel_json.get("panel_name") or panel_json.get("circuits"):
                all_panels.append(panel_json)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error (table {i}): {str(e)}")

    return all_panels

async def _fallback_raw_text(raw_text: str, client, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    If no tables found, process the entire raw_text as a single unit.
    Return a list of one or more panels if discovered.
    """
    from services.ai_service import AiRequest

    ai_service = DrawingAiService(client, drawing_instructions={}, logger=logger)

    fallback_prompt = """
You are an electrical-engineering assistant. I have raw text from a panel schedule PDF.
Columns might be unclear. Return JSON like:
{
  "panel_name": "...",
  "panel_metadata": {},
  "circuits": [
    { "circuit": "...", "load_name": "...", "trip": "...", "poles": "...", ... }
  ]
}
Do not skip lines that look like circuit data.
    """.strip()

    # Process the entire content as a single unit - NO CHUNKING
    user_text = f"FULL RAW TEXT:\n{raw_text}"

    request = AiRequest(
        content=user_text,
        model_type=ModelType.GPT_4O_MINI,
        temperature=0.2,
        max_tokens=3000,
        system_message=fallback_prompt
    )

    response = await ai_service.process(request)
    if not response.success or not response.content:
        logger.warning(f"GPT parse error in fallback processing: {response.error}")
        return []

    try:
        fallback_data = json.loads(response.content)
        # Normalize synonyms
        fallback_data = normalize_panel_data_fields(fallback_data)

        # If no circuits or panel name found, return empty list
        if not fallback_data.get("panel_name") and not fallback_data.get("circuits"):
            return []

        return [fallback_data]
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in fallback processing: {str(e)}")
        return []

```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/utils/pdf_processor.py
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

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/services/ai_service.py
```py
import json
import logging
from enum import Enum
from typing import Dict, Any, Optional, TypeVar, Generic, List
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from utils.performance_utils import time_operation

# Drawing type-specific instructions with main types and subtypes
DRAWING_INSTRUCTIONS = {
    # Main drawing types
    "Electrical": """
    You are an electrical drawing expert extracting structured information. Focus on:
    
    1. All panel schedules - capture complete information about:
       - Panel metadata (name, voltage, phases, rating, location)
       - All circuits with numbers, trip sizes, poles, load descriptions
       - Any panel notes or specifications
    
    2. All equipment schedules with:
       - Complete electrical characteristics (voltage, phase, current ratings)
       - Connection types and mounting specifications
       - Part numbers and manufacturers when available
    
    3. Installation details:
       - Circuit assignments and home run information
       - Mounting heights and special requirements
       - Keyed notes relevant to electrical items
       
    Structure all schedule information into consistent field names (e.g., use 'load_name' for descriptions, 
    'circuit' for circuit numbers, 'trip' for breaker sizes).
    
    IMPORTANT: Ensure ALL circuits, equipment items, and notes are captured in your output. Missing information 
    can cause installation errors.
    """,
    
    "Mechanical": """
    Capture ALL mechanical equipment information with extreme detail and precision:
    
    1. Complete equipment schedules with:
       - Model numbers, manufacturers, and dimensions
       - Capacities (CFM, BTU, tonnage)
       - Electrical requirements (voltage, phase, FLA)
       - Mounting details and clearance requirements
    
    2. HVAC specifications:
       - Airflow volumes and static pressures
       - Control requirements and sequence of operations
       - Ductwork specifications and sizing
    
    3. Installation instructions:
       - Mounting heights and clearances
       - Connection requirements (electrical, plumbing, controls)
       - Testing and balancing requirements
       
    IMPORTANT: ALL mechanical information must be captured with exact values - don't round or approximate.
    Engineers rely on these precise specifications for proper system operation.
    """,
    
    "Plumbing": """
    You are extracting detailed plumbing information. Capture:
    
    1. Complete fixture schedules with:
       - Manufacturer, model, and connection types
       - Flow rates and pressure requirements
       - Mounting details and rough-in dimensions
    
    2. Equipment specifications:
       - Water heaters (capacity, recovery rate, electrical)
       - Pumps (GPM, head pressure, electrical requirements)
       - Special systems (medical gas, vacuum, grease interceptors)
    
    3. System details:
       - Pipe sizing and materials
       - Flow requirements and fixture counts
       - Installation notes and special requirements
       
    IMPORTANT: Capture ALL plumbing elements with their complete specifications. 
    Missing or incomplete information can lead to system failures.
    """,
    
    "Architectural": """
    Extract and structure the following information with PRECISE detail:
    
    1. Room information:
       Create a comprehensive 'rooms' array with objects for EACH room, including:
       - 'number': Room number as string (EXACTLY as shown)
       - 'name': Complete room name
       - 'finish': All ceiling finishes
       - 'height': Ceiling height (with units)
       - 'electrical_info': Any electrical specifications
       - 'architectural_info': Additional architectural details
       - 'wall_types': Wall construction for each wall (north/south/east/west)
    
    2. Complete door and window schedules:
       - Door/window numbers, types, sizes, and materials
       - Hardware specifications and fire ratings
       - Frame types and special requirements
    
    3. Wall type details:
       - Create a 'wall_types' array with complete construction details
       - Include ALL layers, thicknesses, and special requirements
       - Document fire and sound ratings
    
    4. Architectural notes:
       - Capture ALL general notes and keyed notes
       - Include ALL finish schedule information
       
    CRITICAL: EVERY room must be captured. Missing rooms can cause major coordination issues.
    For rooms with minimal information, include what's available and note any missing details.
    """,
    
    "Specifications": """
    PRESERVE THE COMPLETE CONTENT of each specification section with its EXACT structure and details.
    
    For each specification section:
    
    1. Create objects in the 'specifications' array with:
       - 'section_title': The complete section number and title (e.g., "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS")
       - 'content': The FULL TEXT of the entire section, including ALL parts, subsections, and items
    
    2. Maintain the exact hierarchical structure:
       - Preserve all SECTION > PART > SUBSECTION relationships
       - Keep all numbering and lettering of items (1., 1.1, A., etc.)
       - Include all indentation and formatting
    
    3. Include EVERY paragraph, table, list, and requirement:
       - Do not summarize or truncate any content
       - Maintain all technical details, values, and specifications
       - Preserve all notes, warnings, and special instructions
    
    CRITICAL: The entire text must be preserved exactly as written. Even minor omissions or changes 
    can alter contract requirements and cause legal issues.
    """,
    
    "General": """
    Extract ALL relevant content and organize into a comprehensive, structured JSON:
    
    1. Identify the document type and organize data accordingly:
       - For schedules: Create arrays of consistently structured objects
       - For specifications: Preserve the complete text with hierarchical structure
       - For drawings: Document all annotations, dimensions, and references
    
    2. Capture EVERY piece of information:
       - Include ALL notes, annotations, and references
       - Document ALL equipment, fixtures, and components
       - Preserve ALL technical specifications and requirements
    
    3. Maintain relationships between elements:
       - Link components to their locations (rooms, areas)
       - Connect items to their technical specifications
       - Reference related notes and details
    
    Structure everything into a clear, consistent JSON format that preserves ALL the original information.
    """,
    
    # Electrical subtypes
    "Electrical_PanelSchedule": """
    You are an expert electrical engineer analyzing panel schedules. Your goal is to produce a precisely structured JSON representation with COMPLETE information.
    
    Extract and structure the following information with PERFECT accuracy:
    
    1. Panel metadata (create a 'panel' object):
       - 'name': Panel name/ID (EXACTLY as written)
       - 'location': Physical location of panel
       - 'voltage': Full voltage specification (e.g., "120/208V Wye", "277/480V")
       - 'phases': Number of phases (1 or 3) and wires (e.g., "3 Phase 4 Wire")
       - 'amperage': Main amperage rating
       - 'main_breaker': Main breaker size if present
       - 'aic_rating': AIC/interrupting rating
       - 'feed': Source information (fed from)
       - Any additional metadata present (enclosure type, mounting, etc.)
       
    2. Circuit information (create a 'circuits' array with objects for EACH circuit):
       - 'circuit': Circuit number or range EXACTLY as shown (e.g., "1", "2-4-6", "3-5")
       - 'trip': Breaker/trip size with units (e.g., "20A", "70 A")
       - 'poles': Number of poles (1, 2, or 3)
       - 'load_name': Complete description of the connected load
       - 'equipment_ref': Reference to equipment ID if available
       - 'room_id': Connected room(s) if specified
       - Any additional circuit information present
       
    3. Additional information:
       - 'panel_totals': Connected load, demand factors, and calculated loads
       - 'notes': Any notes specific to the panel
       
    CRITICAL: EVERY circuit must be documented EXACTLY as shown on the schedule. Missing circuits, incorrect numbering, or incomplete information can cause dangerous installation errors.
    """,
    
    "Electrical_Lighting": """
    You are extracting complete lighting fixture and control information.
    
    1. Create a comprehensive 'lighting_fixtures' array with details for EACH fixture:
       - 'type_mark': Fixture type designation (EXACTLY as shown)
       - 'description': Complete fixture description
       - 'manufacturer': Manufacturer name
       - 'product_number': Model or catalog number
       - 'lamp_type': Complete lamp specification (e.g., "LED, 35W, 3500K")
       - 'mounting': Mounting type and height
       - 'voltage': Operating voltage
       - 'wattage': Power consumption
       - 'dimensions': Complete fixture dimensions
       - 'count': Quantity of fixtures when specified
       - 'location': Installation locations
       - 'dimmable': Dimming capability and type
       - 'remarks': Any special notes or requirements
       
    2. Document all lighting controls with specific details:
       - Switch types and functions
       - Sensors (occupancy, vacancy, daylight)
       - Dimming systems and protocols
       - Control zones and relationships
       
    3. Document circuit assignments:
       - Panel and circuit numbers
       - Connected areas and zones
       - Load calculations
       
    IMPORTANT: Capture EVERY fixture type and ALL specifications. Missing or incorrect information
    can lead to incompatible installations and lighting failure.
    """,
    
    "Electrical_Power": """
    Extract ALL power distribution and equipment connection information with complete detail:
    
    1. Document all outlets and receptacles in an organized array:
       - Type (standard, GFCI, special purpose, isolated ground)
       - Voltage and amperage ratings
       - Mounting height and orientation
       - Circuit assignment (panel and circuit number)
       - Room location and mounting surface
       - NEMA configuration
       
    2. Create a structured array of equipment connections:
       - Equipment type and designation
       - Power requirements (voltage, phase, amperage)
       - Connection method (hardwired, cord-and-plug)
       - Circuit assignment
       - Disconnecting means
       
    3. Detail specialized power systems:
       - UPS connections and specifications
       - Emergency or standby power
       - Isolated power systems
       - Specialty voltage requirements
       
    4. Document all keyed notes related to power:
       - Special installation requirements
       - Code compliance notes
       - Coordination requirements
       
    IMPORTANT: ALL power elements must be captured with their EXACT specifications.
    Electrical inspectors will verify these details during installation.
    """,
    
    "Electrical_FireAlarm": """
    Extract complete fire alarm system information with precise detail:
    
    1. Document ALL devices in a structured array:
       - Device type (smoke detector, heat detector, pull station, etc.)
       - Model number and manufacturer
       - Mounting height and location
       - Zone/circuit assignment
       - Addressable or conventional designation
       
    2. Identify all control equipment:
       - Fire alarm control panel specifications
       - Power supplies and battery calculations
       - Remote annunciators
       - Auxiliary control functions
       
    3. Capture ALL wiring specifications:
       - Circuit types (initiating, notification, signaling)
       - Wire types, sizes, and ratings
       - Survivability requirements
       
    4. Document interface requirements:
       - Sprinkler system monitoring
       - Elevator recall functions
       - HVAC shutdown
       - Door holder/closer release
       
    CRITICAL: Fire alarm systems are life-safety systems subject to strict code enforcement.
    ALL components and functions must be documented exactly as specified to ensure proper operation.
    """,
    
    "Electrical_Technology": """
    Extract ALL low voltage systems information with complete technical detail:
    
    1. Document data/telecom infrastructure in structured arrays:
       - Outlet types and locations
       - Cable specifications (category, shielding)
       - Mounting heights and orientations
       - Pathway types and sizes
       - Equipment rooms, racks, and cabinets
       
    2. Identify security systems with specific details:
       - Camera types, models, and coverage areas
       - Access control devices and door hardware
       - Intrusion detection sensors and zones
       - Control equipment and monitoring requirements
       
    3. Document audiovisual systems:
       - Display types, sizes, and mounting details
       - Audio equipment and speaker layout
       - Control systems and interfaces
       - Signal routing and processing
       
    4. Capture specialty systems:
       - Nurse call or emergency communication
       - Distributed antenna systems (DAS)
       - Paging and intercom
       - Radio and wireless systems
       
    IMPORTANT: Technology systems require precise documentation to ensure proper integration.
    ALL components, connections, and configurations must be captured as specified.
    """,
    
    # Architectural subtypes
    "Architectural_FloorPlan": """
    Extract COMPLETE floor plan information with precise room-by-room detail:
    
    1. Create a comprehensive 'rooms' array with objects for EACH room, capturing:
       - 'number': Room number EXACTLY as shown (including prefixes/suffixes)
       - 'name': Complete room name
       - 'dimensions': Length, width, and area when available
       - 'adjacent_rooms': List of connecting room numbers
       - 'wall_types': Wall construction for each room boundary (north/south/east/west)
       - 'door_numbers': Door numbers providing access to the room
       - 'window_numbers': Window numbers in the room
       
    2. Document circulation paths with specific details:
       - Corridor widths and clearances
       - Stair dimensions and configurations
       - Elevator locations and sizes
       - Exit paths and egress requirements
       
    3. Identify area designations and zoning:
       - Fire-rated separations and occupancy boundaries
       - Smoke compartments
       - Security zones
       - Department or functional areas
       
    CRITICAL: EVERY room must be documented with ALL available information.
    Missing rooms or incomplete details can cause serious coordination issues across all disciplines.
    When room information is unclear or incomplete, note this in the output.
    """,
    
    "Architectural_ReflectedCeiling": """
    Extract ALL ceiling information with complete room-by-room detail:
    
    1. Create a comprehensive 'rooms' array with ceiling-specific objects for EACH room:
       - 'number': Room number EXACTLY as shown
       - 'name': Complete room name
       - 'ceiling_type': Material and system (e.g., "2x2 ACT", "GWB")
       - 'ceiling_height': Height above finished floor (with units)
       - 'soffit_heights': Heights of any soffits or bulkheads
       - 'slope': Ceiling slope information if applicable
       - 'fixtures': Array of ceiling-mounted elements (lights, diffusers, sprinklers)
       
    2. Document ceiling transitions with specific details:
       - Height changes between areas
       - Bulkhead and soffit dimensions
       - Special ceiling features (clouds, islands)
       
    3. Identify ceiling-mounted elements:
       - Lighting fixtures (coordinated with electrical)
       - HVAC diffusers and registers
       - Sprinkler heads and fire alarm devices
       - Specialty items (projector mounts, speakers)
       
    IMPORTANT: Ceiling coordination is critical for clash detection.
    EVERY room must have complete ceiling information to prevent conflicts with mechanical, 
    electrical, and plumbing systems during installation.
    """,
    
    "Architectural_Partition": """
    Extract ALL wall and partition information with precise construction details:
    
    1. Create a comprehensive 'wall_types' array with objects for EACH type:
       - 'type': Wall type designation EXACTLY as shown
       - 'description': Complete description of the wall assembly
       - 'details': Object containing:
         - 'stud_type': Material and thickness (steel, wood)
         - 'stud_width': Dimension with units
         - 'stud_spacing': Spacing with units
         - 'layers': Complete description of all layers from exterior to interior
         - 'insulation': Type and R-value
         - 'total_thickness': Overall dimension with units
       - 'fire_rating': Fire resistance rating with duration
       - 'sound_rating': STC or other acoustic rating
       - 'height': Height designation ("to deck", "above ceiling")
       
    2. Document room-to-wall type relationships:
       - For each room, identify wall types used on each boundary (north/south/east/west)
       - Note any special conditions or variations
       
    3. Identify special wall conditions:
       - Seismic considerations
       - Expansion/control joints
       - Bracing requirements
       - Wall transitions
       
    CRITICAL: Wall type details impact all disciplines (architectural, structural, mechanical, electrical).
    EVERY wall type must be fully documented with ALL construction details to ensure proper installation.
    """,
    
    "Architectural_Details": """
    Extract ALL architectural details with complete construction information:
    
    1. Document each architectural detail:
       - Detail number and reference
       - Complete description of the assembly
       - Materials and dimensions
       - Connection methods and fastening
       - Finish requirements
       
    2. Capture specific assembly information:
       - Waterproofing and flashing details
       - Thermal and moisture protection
       - Acoustic treatments
       - Fire and smoke barriers
       
    3. Document all annotations and notes:
       - Construction requirements
       - Installation sequence
       - Quality standards
       - Reference standards
       
    IMPORTANT: Architectural details provide critical information for proper construction.
    ALL detail information must be captured exactly as specified to ensure code compliance
    and proper installation.
    """,
    
    "Architectural_Schedules": """
    Extract ALL architectural schedules with complete information for each element:
    
    1. Door schedules with comprehensive detail:
       - Door number EXACTLY as shown
       - Type, size (width, height, thickness)
       - Material and finish
       - Fire rating and label requirements
       - Frame type and material
       - Hardware sets and special requirements
       
    2. Window schedules with specific details:
       - Window number and type
       - Dimensions and configuration
       - Glazing type and performance ratings
       - Frame material and finish
       - Operating requirements
       
    3. Room finish schedules:
       - Room number and name
       - Floor, base, wall, and ceiling finishes
       - Special treatments or requirements
       - Finish transitions
       
    4. Accessory and equipment schedules:
       - Item designations and types
       - Mounting heights and locations
       - Material and finish specifications
       - Quantities and installation notes
       
    CRITICAL: Schedule information is used by multiple trades and disciplines.
    EVERY scheduled item must be completely documented with ALL specifications to ensure
    proper procurement and installation.
    """
}

def detect_drawing_subtype(drawing_type: str, file_name: str) -> str:
    """
    Detect more specific drawing subtype based on drawing type and filename.
    
    Args:
        drawing_type: Main drawing type (Electrical, Architectural, etc.)
        file_name: Name of the file being processed
        
    Returns:
        More specific subtype or the original drawing type if no subtype detected
    """
    if not file_name or not drawing_type:
        return drawing_type
    
    file_name_lower = file_name.lower()
    
    # Electrical subtypes
    if drawing_type == "Electrical":
        # Panel schedules - check for these first as they're most specific
        if any(term in file_name_lower for term in ["panel", "schedule", "panelboard", "circuit", "h1", "l1", "k1", "k1s", "21lp-1", "20h-1"]):
            return "Electrical_PanelSchedule"
        # Lighting fixtures and controls
        elif any(term in file_name_lower for term in ["light", "lighting", "fixture", "lamp", "luminaire", "rcp", "ceiling"]):
            return "Electrical_Lighting"
        # Power distribution
        elif any(term in file_name_lower for term in ["power", "outlet", "receptacle", "equipment", "connect", "riser", "metering"]):
            return "Electrical_Power"
        # Fire alarm systems
        elif any(term in file_name_lower for term in ["fire", "alarm", "fa", "detection", "smoke", "emergency", "evacuation"]):
            return "Electrical_FireAlarm"
        # Low voltage systems
        elif any(term in file_name_lower for term in ["tech", "data", "comm", "security", "av", "low voltage", "telecom", "network"]):
            return "Electrical_Technology"
    
    # Architectural subtypes
    elif drawing_type == "Architectural":
        # Reflected ceiling plans
        if any(term in file_name_lower for term in ["rcp", "ceiling", "reflected"]):
            return "Architectural_ReflectedCeiling"
        # Wall types and partitions
        elif any(term in file_name_lower for term in ["partition", "wall type", "wall-type", "wall", "room wall"]):
            return "Architectural_Partition"
        # Floor plans
        elif any(term in file_name_lower for term in ["floor", "plan", "layout", "room"]):
            return "Architectural_FloorPlan"
        # Door and window schedules
        elif any(term in file_name_lower for term in ["door", "window", "hardware", "schedule"]):
            return "Architectural_Schedules"
        # Architectural details
        elif any(term in file_name_lower for term in ["detail", "section", "elevation", "assembly"]):
            return "Architectural_Details"
    
    # Specification documents
    elif "specification" in drawing_type.lower() or "spec" in file_name_lower:
        return "Specifications"
    
    # If no subtype detected, return the main type
    return drawing_type

class ModelType(Enum):
    """Enumeration of supported AI model types."""
    GPT_4O_MINI = "gpt-4o-mini"  # Updated to remove date-specific version
    GPT_4O = "gpt-4o"  # Updated to remove date-specific version

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
        "temperature": 0.1,  # Reduced default temperature for more consistent output
        "max_tokens": 16000,
    }
    
    # For very long content or complex documents, use more powerful model
    if content_length > 50000 or "specification" in drawing_type.lower():
        params["model_type"] = ModelType.GPT_4O
        # Calculate max_tokens dynamically based on content length
        # Estimate token count as roughly chars/4 for English text
        estimated_input_tokens = min(128000, len(raw_content) // 4)  # Cap at 128k tokens maximum
        # Reserve at least 8000 tokens for output, but don't exceed model context limits
        params["max_tokens"] = max(8000, min(14000, 32000 - estimated_input_tokens))
    
    # Adjust based on drawing type
    if "Electrical" in drawing_type:
        # Panel schedules need lowest temperature for precision
        if "PanelSchedule" in drawing_type:
            params["temperature"] = 0.05
            # Use more capable model for complex panel schedules
            if content_length > 15000:
                params["model_type"] = ModelType.GPT_4O
        # Other electrical drawings need precision too
        else:
            params["temperature"] = 0.1
    
    elif "Architectural" in drawing_type:
        # Room information needs precision but some inference
        if "FloorPlan" in drawing_type or "ReflectedCeiling" in drawing_type:
            params["temperature"] = 0.1
            if content_length > 20000:
                params["model_type"] = ModelType.GPT_4O
    
    # For specifications, use more powerful model and lower temperature
    if "SPECIFICATION" in file_name.upper() or "Specifications" in drawing_type:
        logging.info(f"Processing specification document ({content_length} chars)")
        params["temperature"] = 0.05
        params["model_type"] = ModelType.GPT_4O
        # Use dynamic calculation for specifications as well
        estimated_input_tokens = min(128000, len(raw_content) // 4)
        params["max_tokens"] = max(8000, min(14000, 32000 - estimated_input_tokens))
    
    # Safety check: ensure max_tokens is within reasonable limits
    params["max_tokens"] = min(params["max_tokens"], 16000)
    
    logging.info(f"Using model {params['model_type'].value} with temperature {params['temperature']} and max_tokens {params['max_tokens']}")
    
    return params

T = TypeVar('T')

class AiRequest:
    """
    Class to hold AI API request parameters.
    """
    def __init__(
        self,
        content: str,
        model_type: 'ModelType' = None,
        temperature: float = 0.2,
        max_tokens: int = 3000,
        system_message: str = ""
    ):
        """
        Initialize an AiRequest.
        
        Args:
            content: Content to send to the API
            model_type: Model type to use
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            system_message: System message to use
        """
        self.content = content
        self.model_type = model_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_message = system_message

class AiResponse(Generic[T]):
    """
    Generic class to hold AI API response data or errors.
    """
    def __init__(self, success: bool = True, content: str = "", parsed_content: Optional[T] = None, error: str = ""):
        """
        Initialize an AiResponse.
        
        Args:
            success: Whether the API call was successful
            content: Raw content from the API
            parsed_content: Optional parsed content (of generic type T)
            error: Error message if the call failed
        """
        self.success = success
        self.content = content
        self.parsed_content = parsed_content
        self.error = error

class DrawingAiService:
    """
    Specialized AI service for processing construction drawings.
    """
    def __init__(self, client: AsyncOpenAI, drawing_instructions: Dict[str, str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the DrawingAiService.

        Args:
            client: AsyncOpenAI client instance
            drawing_instructions: Optional dictionary of drawing type-specific instructions
            logger: Optional logger instance
        """
        self.client = client
        self.drawing_instructions = drawing_instructions or DRAWING_INSTRUCTIONS
        self.logger = logger or logging.getLogger(__name__)

    def _get_default_system_message(self, drawing_type: str) -> str:
        """
        Get the default system message for the given drawing type.
        
        Args:
            drawing_type: Type of drawing (Architectural, Electrical, etc.) or subtype
            
        Returns:
            System message string
        """
        # Try to get instructions for the specific subtype first
        drawing_instruction = self.drawing_instructions.get(
            drawing_type,
            # If subtype not found and it contains an underscore, try the main type
            self.drawing_instructions.get(
                drawing_type.split('_')[0] if '_' in drawing_type else "",
                # Fall back to general instructions
                self.drawing_instructions.get("General", "")
            )
        )
        
        return f"""
        You are a construction drawing expert tasked with extracting complete, detailed information from the provided content.
        Your job is to structure this information into a comprehensive, well-organized JSON object with these sections:
        
        - 'metadata': Include drawing number, title, date, revision, and any other identifying information.
        - 'schedules': Array of schedules with type and data. Use consistent field names for similar data across different schedules.
        - 'notes': Array of notes and annotations found in the drawing.
        - 'specifications': Array of specification sections with 'section_title' and 'content' fields.
        - 'rooms': For architectural drawings, include an array of rooms with detailed information.
        - Additional sections as appropriate for the specific drawing type.
        
        {drawing_instruction}
        
        CRITICAL REQUIREMENTS:
        1. Extract ALL information from the drawing - nothing should be omitted
        2. Structure data consistently with descriptive field names
        3. For unclear or ambiguous information, include it with a note about the uncertainty
        4. Ensure the output is valid JSON - no malformed structures or syntax errors
        5. For architectural drawings, ALWAYS include a 'rooms' array when room information is present
        
        Accurate and complete extraction is essential - construction decisions will be based on this data.
        """

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process(self, request: AiRequest) -> AiResponse[Dict[str, Any]]:
        """
        Process an AI request.
        
        Args:
            request: AiRequest object containing parameters
            
        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing content of length {len(request.content)}")
            
            response = await self.client.chat.completions.create(
                model=request.model_type.value,
                messages=[
                    {"role": "system", "content": request.system_message},
                    {"role": "user", "content": request.content}
                ],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...")  # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error processing drawing: {str(e)}")
            return AiResponse(success=False, error=str(e))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(Exception)
    )
    @time_operation("ai_processing")
    async def process_drawing_with_responses(
        self,
        raw_content: str,
        drawing_type: str,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        model_type: ModelType = ModelType.GPT_4O_MINI,
        system_message: Optional[str] = None,
        example_output: Optional[str] = None
    ) -> AiResponse:
        """
        Process a drawing using the OpenAI API.
        
        Args:
            raw_content: Complete raw content from the drawing - NO TRUNCATION
            drawing_type: Type of drawing
            temperature: Temperature parameter
            max_tokens: Maximum tokens to generate
            model_type: Model type to use
            system_message: Optional system message
            example_output: Optional example output for few-shot learning
            
        Returns:
            AiResponse with parsed content or error
        """
        try:
            self.logger.info(f"Processing {drawing_type} drawing with {len(raw_content)} characters")
            
            messages = [
                {"role": "system", "content": system_message or self._get_default_system_message(drawing_type)}
            ]
            
            # Add example output if provided (few-shot learning)
            if example_output:
                messages.append({"role": "user", "content": "Please process this drawing content and convert it to structured JSON:"})
                messages.append({"role": "assistant", "content": example_output})
                messages.append({"role": "user", "content": "Now process this new drawing content in the same format:"})
            
            # Add the actual content to process
            messages.append({"role": "user", "content": raw_content})
            
            response = await self.client.chat.completions.create(
                model=model_type.value,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}  # Ensure JSON response
            )
            
            content = response.choices[0].message.content
            
            try:
                parsed_content = json.loads(content)
                return AiResponse(success=True, content=content, parsed_content=parsed_content)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON decoding error: {str(e)}")
                self.logger.error(f"Raw content received: {content[:500]}...")  # Log the first 500 chars for debugging
                return AiResponse(success=False, error=f"JSON decoding error: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error processing drawing: {str(e)}")
            return AiResponse(success=False, error=str(e))

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
        system_message: Optional[str] = None,
        example_output: Optional[str] = None
    ) -> str:
        """
        Process a drawing using a specific prompt, ensuring full content is sent to the API.
        
        Args:
            raw_content: Raw content from the drawing
            temperature: Temperature parameter for the AI model
            max_tokens: Maximum tokens to generate
            model_type: AI model type to use
            system_message: Optional custom system message to override default
            example_output: Optional example output for few-shot learning
        
        Returns:
            Processed content as a JSON string

        Raises:
            JSONDecodeError: If the response is not valid JSON
            ValueError: If the JSON structure is invalid
            Exception: For other processing errors
        """
        default_system_message = """
        You are an AI assistant specialized in construction drawings. Extract all relevant information from the provided content and organize it into a structured JSON object with these sections:
        
        - "metadata": An object containing drawing metadata such as "drawing_number", "title", "date", and "revision".
        Include any available information; if a field is missing, omit it.
        
        - "schedules": An array of schedule objects. Each schedule should have a "type" (e.g., "electrical_panel",
        "mechanical") and a "data" array containing objects with standardized field names. For panel schedules,
        use consistent field names like "circuit" for circuit numbers, "trip" for breaker sizes, 
        "load_name" for equipment descriptions, and "poles" for the number of poles.
        
        - "notes": An array of strings containing any notes or instructions found in the drawing.
        
        - "specifications": An array of objects, each with a "section_title" and "content" for specification sections.
        
        - "rooms": For architectural drawings, include an array of rooms with 'number', 'name', 'finish', 'height',
        'electrical_info', and 'architectural_info'.
        
        CRITICAL REQUIREMENTS:
        1. The JSON output MUST include ALL information from the drawing - nothing should be omitted
        2. Structure data consistently with descriptive field names
        3. Panel schedules MUST include EVERY circuit, with correct circuit numbers, trip sizes, and descriptions
        4. For architectural drawings, ALWAYS include a 'rooms' array with ALL rooms
        5. For specifications, preserve the COMPLETE text in the 'content' field
        6. Ensure the output is valid JSON with no syntax errors
        
        Construction decisions will be based on this data, so accuracy and completeness are essential.
        """

        # Use the provided system message or fall back to default
        final_system_message = system_message if system_message else default_system_message
        
        content_length = len(raw_content)
        self.logger.info(f"Processing content of length {content_length} with model {model_type.value}")

        # Check if content is too large and log a warning
        if content_length > 250000 and model_type == ModelType.GPT_4O_MINI:
            self.logger.warning(f"Content length ({content_length} chars) may exceed GPT-4o-mini context window. Switching to GPT-4o.")
            model_type = ModelType.GPT_4O
        
        if content_length > 500000:
            self.logger.warning(f"Content length ({content_length} chars) exceeds GPT-4o context window. Processing may be incomplete.")

        try:
            messages = [
                {"role": "system", "content": final_system_message}
            ]
            
            # Add example output if provided (few-shot learning)
            if example_output:
                messages.append({"role": "user", "content": "Please process this drawing content and convert it to structured JSON:"})
                messages.append({"role": "assistant", "content": example_output})
                messages.append({"role": "user", "content": "Now process this new drawing content in the same format:"})
            
            # Add the actual content to process
            messages.append({"role": "user", "content": raw_content})
            
            # Calculate rough token estimate for logging
            estimated_tokens = content_length // 4
            self.logger.info(f"Estimated input tokens: ~{estimated_tokens}")
            
            try:
                response = await self.client.chat.completions.create(
                    model=model_type.value,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}  # Ensure JSON response
                )
                content = response.choices[0].message.content
                
                # Process usage information if available
                if hasattr(response, 'usage') and response.usage:
                    self.logger.info(f"Token usage - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Total: {response.usage.total_tokens}")
                
                try:
                    # Validate JSON parsing
                    parsed_content = json.loads(content)
                    if not self.validate_json(parsed_content):
                        self.logger.warning("JSON validation failed - missing required keys")
                        # Still return the content, as it might be usable even with missing keys
                    
                    return content
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decoding error: {str(e)}")
                    self.logger.error(f"Raw content received: {content[:500]}...")  # Log the first 500 chars for debugging
                    raise
            except Exception as e:
                if "maximum context length" in str(e).lower() or "token limit" in str(e).lower():
                    self.logger.error(f"Context length exceeded: {str(e)}")
                    raise ValueError(f"Content too large for model context window: {str(e)}")
                else:
                    self.logger.error(f"API error: {str(e)}")
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
        # Basic validation - check for required top-level keys
        required_keys = ["metadata", "schedules", "notes"]
        
        # Specifications validation - check structure and convert if needed
        if "specifications" in json_data:
            specs = json_data["specifications"]
            if isinstance(specs, list) and specs:
                # Convert string arrays to object arrays if needed
                if isinstance(specs[0], str):
                    self.logger.warning("Converting specifications from string array to object array")
                    json_data["specifications"] = [{"section_title": spec, "content": ""} for spec in specs]
        else:
            # Add empty specifications array if missing
            json_data["specifications"] = []
            
        # For architectural drawings, check for rooms array
        if "metadata" in json_data and "drawing_type" in json_data["metadata"]:
            if "architectural" in json_data["metadata"]["drawing_type"].lower() and "rooms" not in json_data:
                self.logger.warning("Architectural drawing missing 'rooms' array")
                return False
                
        return all(key in json_data for key in required_keys)

    async def get_example_output(self, drawing_type: str) -> Optional[str]:
        """
        Retrieve an example output for the given drawing type from a library of examples.
        This enables few-shot learning for better consistency.
        
        Args:
            drawing_type: Type of drawing
            
        Returns:
            Example output as JSON string, or None if no example is available
        """
        # This would typically load from a database or file system
        # For now, we'll return None as a placeholder
        return None

@time_operation("ai_processing")
async def process_drawing(raw_content: str, drawing_type: str, client, file_name: str = "") -> str:
    """
    Use GPT to parse PDF text and table data into structured JSON based on the drawing type.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing (Architectural, Electrical, etc.)
        client: OpenAI client
        file_name: Optional name of the file being processed
        
    Returns:
        Structured JSON as a string
        
    Raises:
        ValueError: If the content is too large for processing
        JSONDecodeError: If the response is not valid JSON
        Exception: For other processing errors
    """
    if not raw_content:
        logging.warning(f"Empty content received for {file_name}. Cannot process.")
        raise ValueError("Cannot process empty content")
        
    # Log details about processing task
    content_length = len(raw_content)
    drawing_type = drawing_type or "Unknown"
    file_name = file_name or "Unknown"
    
    logging.info(f"Starting drawing processing: Type={drawing_type}, File={file_name}, Content length={content_length}")
    
    try:
        # Detect more specific drawing subtype
        subtype = detect_drawing_subtype(drawing_type, file_name)
        logging.info(f"Detected drawing subtype: {subtype}")
        
        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
        
        # Get optimized parameters for this drawing
        params = optimize_model_parameters(subtype, raw_content, file_name)
        
        # Try to get an example output for few-shot learning
        example_output = await ai_service.get_example_output(subtype)
        
        # Check if this is a specification document
        is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"
        
        # Enhanced system message with different emphasis based on document type
        if is_specification:
            system_message = f"""
            You are processing a SPECIFICATION document. Extract all relevant information and organize it into a JSON object.
            
            CRITICAL INSTRUCTIONS:
            - In the 'specifications' array, create objects with 'section_title' and 'content' fields
            - The 'section_title' should contain the section number and name (e.g., "SECTION 16050 - BASIC ELECTRICAL MATERIALS AND METHODS")
            - The 'content' field MUST contain the COMPLETE TEXT of each section, including all parts, subsections, and detailed requirements
            - Preserve the hierarchical structure (SECTION > PART > SUBSECTION)
            - Include all numbered and lettered items, paragraphs, tables, and detailed requirements
            - Do not summarize or truncate - include the ENTIRE text of each section
            
            {DRAWING_INSTRUCTIONS.get("Specifications")}
            
            Ensure the output is valid JSON.
            """
        else:
            # Get the appropriate system message based on detected subtype
            type_instruction = DRAWING_INSTRUCTIONS.get(subtype, DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"]))
            
            system_message = f"""
            You are processing a construction drawing of type: {subtype}
            
            Extract ALL relevant information and organize it into a comprehensive JSON object with the following sections:
            - 'metadata': Object containing drawing number, title, date, and any other identifying information
            - 'schedules': Array of schedules with type and data using consistent field names
            - 'notes': Array of all notes and annotations
            - 'specifications': Array of specification sections with 'section_title' and 'content'
            - Other sections as appropriate for this drawing type
            
            {type_instruction}
            
            CRITICAL: Your output MUST include ALL information from the drawing - nothing should be omitted.
            Use consistent field names for similar data (e.g., "circuit", "trip", "load_name" for panel schedules).
            For architectural drawings, ALWAYS include a 'rooms' array with ALL rooms.
            
            Engineers, contractors, and installers will rely on this data for construction decisions.
            Accuracy and completeness are essential to prevent costly mistakes and safety issues.
            """
        
        # Process the drawing using the most appropriate method
        try:
            response = await ai_service.process_with_prompt(
                raw_content=raw_content,
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                model_type=params["model_type"],
                system_message=system_message,
                example_output=example_output
            )
            
            # Validate JSON structure
            try:
                parsed = json.loads(response)
                logging.info(f"Successfully processed {subtype} drawing ({len(response)} chars output)")
                return response
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response from AI service for {file_name}")
                raise
                
        except ValueError as e:
            if "content too large" in str(e).lower():
                logging.error(f"Content too large for {file_name}: {str(e)}")
                raise ValueError(f"Drawing content exceeds model context limits: {str(e)}")
            else:
                logging.error(f"Value error processing {file_name}: {str(e)}")
                raise
                
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing '{file_name}': {str(e)}")
        raise

@time_operation("ai_processing")
async def process_drawing_with_examples(raw_content: str, drawing_type: str, client, file_name: str = "", example_outputs: List[str] = None) -> str:
    """
    Process a drawing using few-shot learning with example outputs.
    
    Args:
        raw_content: Raw content from the drawing
        drawing_type: Type of drawing
        client: OpenAI client
        file_name: Optional name of the file being processed
        example_outputs: List of example JSON outputs for few-shot learning
        
    Returns:
        Structured JSON as a string
        
    Raises:
        ValueError: If the content is too large for processing
        JSONDecodeError: If the response is not valid JSON
        Exception: For other processing errors
    """
    if not raw_content:
        logging.warning(f"Empty content received for {file_name}. Cannot process.")
        raise ValueError("Cannot process empty content")
    
    # Log details about the processing task
    content_length = len(raw_content)
    drawing_type = drawing_type or "Unknown"
    file_name = file_name or "Unknown"
    
    logging.info(f"Starting drawing processing with examples: Type={drawing_type}, File={file_name}, Content length={content_length}")
    logging.info(f"Number of example outputs provided: {len(example_outputs) if example_outputs else 0}")
    
    try:
        # Detect more specific drawing subtype
        subtype = detect_drawing_subtype(drawing_type, file_name)
        logging.info(f"Detected drawing subtype: {subtype}")
        
        # Create the AI service
        ai_service = DrawingAiService(client, DRAWING_INSTRUCTIONS)
        
        # Get optimized parameters for this drawing
        params = optimize_model_parameters(subtype, raw_content, file_name)
        
        # Check if this is a specification document
        is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"
        
        # Base system message with type-specific instructions
        type_instruction = DRAWING_INSTRUCTIONS.get(subtype, DRAWING_INSTRUCTIONS.get(drawing_type, DRAWING_INSTRUCTIONS["General"]))
        
        system_message = f"""
        You are a construction drawing expert tasked with extracting ALL information from {subtype} drawings.
        
        Your job is to convert raw drawing content into structured JSON following the exact format shown in the examples.
        Pay close attention to field names and structure used in the examples - your output should match this format.
        
        {type_instruction}
        
        CRITICAL REQUIREMENTS:
        1. Extract EVERY piece of information from the drawing - nothing should be omitted
        2. Use the EXACT same field names and structure as shown in the examples
        3. Include ALL circuits in panel schedules, ALL rooms in architectural drawings, etc.
        4. When in doubt about a field name, check the examples first
        5. Ensure your output is valid JSON with no syntax errors
        
        Construction decisions will be based on this data, so accuracy and completeness are essential.
        """
        
        # Process using the examples-based approach
        try:
            if example_outputs and len(example_outputs) > 0:
                # Use first example only as some APIs have token limits
                example = example_outputs[0]
                logging.info("Processing with example-based learning")
                response = await ai_service.process_with_prompt(
                    raw_content=raw_content,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                    model_type=params["model_type"],
                    system_message=system_message,
                    example_output=example
                )
            else:
                # No examples provided, fall back to standard processing
                logging.info("No examples provided, falling back to standard processing")
                response = await process_drawing(raw_content, drawing_type, client, file_name)
            
            # Validate JSON structure
            try:
                parsed = json.loads(response)
                logging.info(f"Successfully processed {subtype} drawing with examples ({len(response)} chars output)")
                return response
            except json.JSONDecodeError:
                logging.error(f"Invalid JSON response from AI service for {file_name}")
                raise
        
        except ValueError as e:
            if "content too large" in str(e).lower():
                logging.error(f"Content too large for {file_name}: {str(e)}")
                raise ValueError(f"Drawing content exceeds model context limits: {str(e)}")
            else:
                logging.error(f"Value error processing {file_name}: {str(e)}")
                raise
        
    except Exception as e:
        logging.error(f"Error processing {drawing_type} drawing '{file_name}' with examples: {str(e)}")
        raise
```

File: /Users/collin/Desktop/Ohmni/Projects/ohmni-oracle-template/processing/file_processor.py
```py
import os
import json
import logging
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
from typing import Dict, Any

from services.extraction_service import create_extractor
from services.ai_service import DrawingAiService, process_drawing
from services.storage_service import FileSystemStorage
from utils.performance_utils import time_operation
from utils.constants import get_drawing_type
from config.settings import USE_SIMPLIFIED_PROCESSING

# Load environment variables
load_dotenv()

@time_operation("total_processing")
async def process_pdf_async(
    pdf_path: str,
    client,
    output_folder: str,
    drawing_type: str,
    templates_created: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Process a single PDF asynchronously:
    1) Extract text and tables from PDF
    2) Process with AI using universal prompt
    3) Save structured JSON output
    """
    file_name = os.path.basename(pdf_path)
    logger = logging.getLogger(__name__)

    with tqdm(total=100, desc=f"Processing {file_name}", leave=False) as pbar:
        try:
            pbar.update(10)
            extractor = create_extractor(drawing_type, logger)
            storage = FileSystemStorage(logger)
            ai_service = DrawingAiService(client, logger)

            extraction_result = await extractor.extract(pdf_path)
            if not extraction_result.success:
                pbar.update(100)
                logger.error(f"Extraction failed for {pdf_path}: {extraction_result.error}")
                return {"success": False, "error": extraction_result.error, "file": pdf_path}

            # Concatenate all extracted content without any truncation
            raw_content = extraction_result.raw_text
            for table in extraction_result.tables:
                raw_content += f"\nTABLE:\n{table['content']}\n"
                
            # Log content length for specification files
            is_specification = "SPECIFICATION" in file_name.upper() or drawing_type.upper() == "SPECIFICATIONS"
            if is_specification:
                logger.info(f"Processing specification document: {file_name} with {len(raw_content)} characters")
            else:
                logger.info(f"Processing {drawing_type} drawing: {file_name} with {len(raw_content)} characters")
                
            pbar.update(20)

            # Always use process_drawing regardless of simplified setting
            structured_json = await process_drawing(raw_content, drawing_type, client, file_name)
            pbar.update(40)

            type_folder = os.path.join(output_folder, drawing_type)
            os.makedirs(type_folder, exist_ok=True)

            try:
                parsed_json = json.loads(structured_json)
            except json.JSONDecodeError as e:
                pbar.update(100)
                logger.error(f"JSON error for {pdf_path}: {str(e)}")
                logger.error(f"Raw API response: {structured_json[:500]}...")  # Log the first 500 chars
                raw_output_path = os.path.join(type_folder, f"{os.path.splitext(file_name)[0]}_raw_response.json")
                await storage.save_text(structured_json, raw_output_path)
                return {"success": False, "error": f"JSON parse failed: {str(e)}", "file": pdf_path}

            output_path = os.path.join(type_folder, f"{os.path.splitext(file_name)[0]}_structured.json")
            await storage.save_json(parsed_json, output_path)
            pbar.update(20)
            logger.info(f"Saved: {output_path}")

            if drawing_type == 'Architectural' and 'rooms' in parsed_json:
                from templates.room_templates import process_architectural_drawing
                result = process_architectural_drawing(parsed_json, pdf_path, type_folder)
                templates_created['floor_plan'] = True
                logger.info(f"Created templates: {result}")

            pbar.update(10)
            return {"success": True, "file": output_path}
        except Exception as e:
            pbar.update(100)
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"success": False, "error": str(e), "file": pdf_path}
```
</file_contents>

