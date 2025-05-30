import logging
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)

import pytesseract
from pytesseract import image_to_string

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

import pdfplumber, fitz, json, os
import base64
from PIL import Image
import io

DATA_DIR = "C:/Users/zaibu/Desktop/FAST/8th_semester/GEN AI/multimodal_rag/Data"
OUT_FILE = "C:/Users/zaibu/Desktop/FAST/8th_semester/GEN AI/multimodal_rag/Data/chunks.jsonl"

def parse_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            # paragraphs
            text = page.extract_text()
            if text:
                yield {"doc": os.path.basename(pdf_path),
                       "type": "text", "page": i, "content": text}

            # tables
            for table in page.extract_tables():
                yield {"doc": os.path.basename(pdf_path),
                       "type": "table", "page": i, "content": table}

def parse_images(pdf_path):
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            # Force into RGB colorspace
            if pix.n != 3:  
                # any non-RGB (grayscale, CMYK, etc) → convert
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # Build a PIL Image from raw samples
            img_pil = Image.frombytes(
                "RGB", (pix.width, pix.height), pix.samples
            )

            # OCR
            text = image_to_string(img_pil)

            # Encode image as PNG via PIL
            buf = io.BytesIO()
            img_pil.save(buf, format="PNG")
            img_png = buf.getvalue()
            buf.close()

            # Base64 for JSON
            b64 = base64.b64encode(img_png).decode("utf-8")

            yield {
                "doc": os.path.basename(pdf_path),
                "type": "image",
                "page": page_index + 1,
                "content": b64,
                "ocr": text
            }

def main():
    with open(OUT_FILE, "w") as fout:
        for fname in os.listdir(DATA_DIR):
            if fname.lower().endswith(".pdf"):
                path = os.path.join(DATA_DIR, fname)
                for chunk in parse_text(path):
                    fout.write(json.dumps(chunk)+"\n")
                for chunk in parse_images(path):
                    fout.write(json.dumps(chunk)+"\n")

if __name__ == "__main__":
    main()
