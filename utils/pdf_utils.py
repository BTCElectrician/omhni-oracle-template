import pymupdf as fitz
import logging
import aiofiles
from typing import List, Dict, Any, Optional, Tuple
import os

logger = logging.getLogger(__name__)

async def extract_text(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text content
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For any other errors during extraction
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
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing image information
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For any other errors during extraction
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
                    
                    # Get transformation matrix and image rectangle
                    # This gives us more precise positioning information
                    image_rect = None
                    for img_info in page.get_image_info():
                        if img_info["xref"] == xref:
                            image_rect = img_info["bbox"]
                            break
                    
                    bbox = image_rect if image_rect else (0, 0, base_image["width"], base_image["height"])
                    
                    images.append({
                        'page': page_index + 1,
                        'index': img_index,
                        'bbox': bbox,
                        'width': base_image["width"],
                        'height': base_image["height"],
                        'type': base_image["ext"],  # Image extension/type
                        'colorspace': base_image.get("colorspace", ""),
                        'xres': base_image.get("xres", 0),
                        'yres': base_image.get("yres", 0)
                    })
        
        logger.info(f"Extracted {len(images)} images from {file_path}")
        return images
    except Exception as e:
        logger.error(f"Error extracting images from {file_path}: {str(e)}")
        raise

async def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """
    Get metadata from a PDF file using PyMuPDF.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary containing PDF metadata
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For any other errors during extraction
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        with fitz.open(file_path) as doc:
            # Enhanced metadata extraction with more fields
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creationDate": doc.metadata.get("creationDate", ""),
                "modDate": doc.metadata.get("modDate", ""),
                "format": "PDF " + doc.metadata.get("format", ""),
                "pageCount": len(doc),
                "encrypted": doc.is_encrypted,
                "fileSize": os.path.getsize(file_path) if os.path.exists(file_path) else 0
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
