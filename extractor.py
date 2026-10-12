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
        Extract tabular data from the PDF using spatial geometry and row alignment.
        Decomposes borderless/collapsed multi-line rows into individual product rows
        using word-level bounding boxes and column boundaries, preserving empty cells.
        """
        all_tables = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    found_tables = page.find_tables()
                    words = page.extract_words()
                    
                    if not found_tables:
                        # Fallback to text-based table extraction if no lines found
                        text_tables = page.extract_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 3,
                            "join_tolerance": 3
                        })
                        for tbl in text_tables:
                            if tbl:
                                all_tables.append(tbl)
                        continue

                    for t in found_tables:
                        raw_table = t.extract()
                        if not raw_table or len(raw_table) < 2:
                            continue

                        # Check if any body row is collapsed (multi-line cells across transactional fields)
                        header_idx = -1
                        for i, row in enumerate(raw_table[:15]):
                            row_str = " ".join(str(c).lower() for c in row if c)
                            kw_count = sum(1 for kw in ["batch", "exp", "hsn", "mrp", "qty", "rate", "item", "product", "description"] if kw in row_str)
                            if kw_count >= 2:
                                header_idx = i
                                break

                        has_collapsed_row = False
                        if header_idx != -1:
                            for r_idx in range(header_idx + 1, len(raw_table)):
                                row = raw_table[r_idx]
                                if sum(1 for c in row if c and "\n" in str(c)) >= 2:
                                    has_collapsed_row = True
                                    break

                        if not has_collapsed_row or header_idx >= len(t.rows):
                            all_tables.append(raw_table)
                            continue

                        # Segment collapsed table rows using column boundaries and word bounding boxes
                        header_cells = t.rows[header_idx].cells
                        col_bboxes = []
                        for j, cell in enumerate(header_cells):
                            if cell is not None:
                                col_bboxes.append((j, cell[0], cell[2])) # (col_idx, x0, x1)

                        if not col_bboxes:
                            all_tables.append(raw_table)
                            continue

                        segmented_table = []
                        for i in range(header_idx + 1):
                            segmented_table.append(raw_table[i])

                        for r_idx in range(header_idx + 1, len(raw_table)):
                            row = raw_table[r_idx]
                            t_row = t.rows[r_idx] if r_idx < len(t.rows) else None
                            multi_line = sum(1 for c in row if c and "\n" in str(c))

                            if multi_line >= 2 and t_row is not None:
                                r_bbox = t_row.bbox
                                # Extract words within this row's spatial bounding box
                                r_words = [
                                    w for w in words
                                    if r_bbox[0] - 2.0 <= w['x0'] and w['x1'] <= r_bbox[2] + 2.0
                                    and r_bbox[1] - 1.0 <= w['top'] and w['bottom'] <= r_bbox[3] + 1.0
                                ]

                                # Cluster words into distinct product rows by vertical midpoint
                                row_clusters = []
                                for w in sorted(r_words, key=lambda x: (x['top'] + x['bottom'])/2):
                                    y = (w['top'] + w['bottom'])/2
                                    matched = False
                                    for cluster in row_clusters:
                                        mean_y = sum(cw['top'] + cw['bottom'] for cw in cluster)/(2*len(cluster))
                                        if abs(y - mean_y) < 4.0:
                                            cluster.append(w)
                                            matched = True
                                            break
                                    if not matched:
                                        row_clusters.append([w])

                                # Reconstruct each row cluster into structured columns
                                for cluster in row_clusters:
                                    row_data = ["" for _ in range(len(header_cells))]
                                    for w in cluster:
                                        w_mid_x = (w['x0'] + w['x1']) / 2
                                        assigned_col = -1
                                        for col_idx, cx0, cx1 in col_bboxes:
                                            if cx0 - 2.0 <= w_mid_x <= cx1 + 2.0:
                                                assigned_col = col_idx
                                                break
                                        if assigned_col != -1:
                                            if row_data[assigned_col]:
                                                row_data[assigned_col] += " " + w['text']
                                            else:
                                                row_data[assigned_col] = w['text']
                                    segmented_table.append(row_data)
                            else:
                                segmented_table.append(row)

                        all_tables.append(segmented_table)
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
            pages = convert_from_path(pdf_path, dpi=200)
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
