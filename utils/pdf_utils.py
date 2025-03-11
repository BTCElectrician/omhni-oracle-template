import pymupdf as fitz
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
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
