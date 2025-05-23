import fitz  # PyMuPDF
import pytesseract
from PIL import Image

def parse_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def parse_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text.strip()
