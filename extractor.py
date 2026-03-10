import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os
from typing import Optional, List

class PDFExtractor:
    """
    A professional-grade PDF text extractor that supports both digital and scanned PDFs.
    """

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the extractor.
        :param tesseract_cmd: Optional path to the Tesseract executable (if not in PATH).
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text(self, pdf_path: str) -> dict:
        """
        Extract text from a PDF, automatically falling back to OCR if no text is found.
        Returns a dictionary with raw_text and extraction_source.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        print(f"[*] Processing {pdf_path}...")
        
        # 1. Try digital extraction
        text = self._extract_digitally(pdf_path)
        source = "pdfplumber (Digital)"
        
        # 2. Check if text is sufficient
        if len(text.strip()) < 50:
            print("[!] Low text density detected. Falling back to OCR...")
            text = self._extract_via_ocr(pdf_path)
            source = "pytesseract (OCR)"
        else:
            print("[+] Successfully extracted text digitally.")
            
        return {"raw_text": text, "extraction_source": source}

    def extract_tables(self, pdf_path: str) -> List[List[List[str]]]:
        """
        Extract tabular data from the PDF.
        """
        all_tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            all_tables.append(table)
        except Exception as e:
            print(f"[!] Table extraction failed: {e}")
        return all_tables

    def _extract_digitally(self, pdf_path: str) -> str:
        extracted_text = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    content = page.extract_text()
                    if content:
                        extracted_text.append(content)
        except Exception as e:
            print(f"[!] Warning during digital extraction: {e}")
            
        return "\n".join(extracted_text)

    def _extract_via_ocr(self, pdf_path: str) -> str:
        """
        Convert PDF pages to images and perform OCR using Tesseract.
        """
        extracted_text = []
        try:
            # Note: poppler must be installed on the system for pdf2image to work
            pages = convert_from_path(pdf_path, dpi=300)
            for i, page in enumerate(pages):
                print(f"    [*] Performing OCR on page {i+1}...")
                content = pytesseract.image_to_string(page)
                if content:
                    extracted_text.append(content)
        except Exception as e:
            print(f"[-] OCR extraction failed: {e}")
            print("[!] Note: Ensure 'Tesseract-OCR' and 'Poppler' are installed and added to PATH.")
            
        return "\n".join(extracted_text)

if __name__ == "__main__":
    # Example usage (for testing)
    # extractor = PDFExtractor()
    # text = extractor.extract_text("sample.pdf")
    # print(text)
    pass
