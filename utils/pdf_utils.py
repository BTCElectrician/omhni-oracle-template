import pdfplumber
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber.
    """
    logger.info(f"Starting text extraction for {file_path}")
    try:
        with pdfplumber.open(file_path) as pdf:
            logger.info(f"Successfully opened {file_path}")
            text = ""
            for i, page in enumerate(pdf.pages):
                logger.info(f"Processing page {i+1} of {len(pdf.pages)}")
                page_text = page.extract_text()
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
    Extract images from a PDF file using pdfplumber.
    """
    try:
        images = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                for image in page.images:
                    images.append({
                        'page': i + 1,
                        'bbox': image['bbox'],
                        'width': image['width'],
                        'height': image['height'],
                        'type': image['type']
                    })
        
        logger.info(f"Extracted {len(images)} images from {file_path}")
        return images
    except Exception as e:
        logger.error(f"Error extracting images from {file_path}: {str(e)}")
        raise

def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata from a PDF file using pdfplumber.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            metadata = pdf.metadata
        logger.info(f"Successfully extracted metadata from {file_path}")
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {file_path}: {str(e)}")
        raise
