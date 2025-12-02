# extract.py
from pdfminer.high_level import extract_text as pdf_extract_text
from PIL import Image
import pytesseract
import io

def from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF file.
    :param file_bytes: PDF file in bytes
    :return: extracted text as string
    """
    try:
        text = pdf_extract_text(io.BytesIO(file_bytes))
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}")


def from_image(file_bytes: bytes) -> str:
    """
    Extract text from an image using OCR.
    :param file_bytes: Image file in bytes
    :return: extracted text as string
    """
    try:
        img = Image.open(io.BytesIO(file_bytes))
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from image: {str(e)}")


def extract_text(file_buffer: io.BytesIO) -> str:
    """
    Determine file type and extract text accordingly.
    Tries PDF first, then image.
    :param file_buffer: file-like object in BytesIO
    :return: extracted text
    """
    data = file_buffer.read()
    file_buffer.seek(0)

    # Attempt PDF extraction
    try:
        text = from_pdf(data)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback to OCR (image)
    try:
        text = from_image(data)
        return text
    except Exception:
        pass

    # If both fail
    return ""
