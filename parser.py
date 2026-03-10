import re
from typing import Dict, Any, List

class InvoiceParser:
    """
    Parses raw text from invoices to extract specific structured data.
    """

    # Define aliases for common column names across different templates
    COLUMN_MAP = {
        "description": [
            "description", "item name", "product", "item description", "particulars", 
            "medicine name", "name of item", "product description", "items", "name",
            "item_name", "product_name"
        ],
        "hsn": ["hsn", "hsn code", "hsn/sac", "commodity code", "hsn/ sac"],
        "batch": ["batch", "batch no", "batch number", "b.no", "lot no", "lot", "batchno"],
        "expiry": [
            "exp", "expiry", "exp date", "expiry date", "exp.", "e.date", "expdt", 
            "exp.date", "valid till"
        ],
        "p_code": ["p.code", "p code", "product code", "item code", "code"],
        "qty": ["qty", "quantity", "qnty", "units"],
        "pack": ["pack", "packing", "pkg", "unit"],
        "mrp": ["mrp", "m.r.p", "maximum retail price"],
        "ptr": ["ptr", "p.t.r", "price to retailer"],
        "pts": ["pts", "p.t.s", "price to stockist"],
        "disc_per": ["disc%", "disc %", "dis%", "discount%", "disc. %", "discount %"],
        "sch_disc": ["sch. disc.%", "sch disc%", "sch.disc.%", "sch disc", "scheme disc"],
        "disc": ["disc", "discount amt", "discount amount", "dis amt", "disc amt"],
        "rate": ["rate", "unit price", "price"],
        "taxable_amt": ["taxable amt", "taxable amount", "taxable value", "taxable", "taxable val", "taxable value", "value", "amount", "total"],
        "free": ["free", "free qty", "bonus", "scheme", "sch"],
        "sgst": ["sgst", "s.g.s.t"],
        "cgst": ["cgst", "c.g.s.t"],
        "igst": ["igst", "i.g.s.t"],
        "mfr": ["mfr", "mfg", "manufacturer", "make", "mfr name", "mfac"],
        "gst": ["gst", "tax%", "gst %", "gst%", "tax %"],
        "gst_amt": ["gst amt", "tax amt", "tax amount", "gst amount", "taxamt", "gstamt"],
        "amount": ["amount", "total", "net amount", "total amt", "grand total", "net payable", "net amount", "line total", "taxable value", "taxable amt"],
        "sno": ["sno", "s.no", "sl no", "sr no", "s no", "bno"],
        "rack": ["rack", "shelf"],
        "sch": ["sch", "scheme", "free", "sch qty", "sch.qty", "sch. qty", "scheme qty"]
    }

    @staticmethod
    def parse(text: str, tables: List[List[List[str]]] = None, mode: str = "admin") -> Dict[str, Any]:
        """
        Execute parsing logic on raw text and tables based on mode ('admin' or 'retailer').
        """
        data = {
            "invoice_metadata": {
                "invoice_number": InvoiceParser._extract_invoice_number(text),
                "date": InvoiceParser._extract_date(text),
                "total_amount": InvoiceParser._extract_total_amount(text, mode),
                "gstin": InvoiceParser._extract_gstin(text),
                "currency": InvoiceParser._extract_currency(text)
            },
            "line_items": []
        }
        
        if tables:
            data["line_items"] = InvoiceParser.parse_line_items(tables, mode)
            
        return data

    @staticmethod
    def _is_column_match(cell_text: str, target_key: str) -> bool:
        """
        Checks if a cell text matches any of the aliases for a target column.
        Handles punctuation, spacing, and partial matches for merged headers.
        """
        if not cell_text: return False
        
        # Normalize: lower, strip dots, spaces, and percent signs
        def normalize(s):
            return re.sub(r'[\.\s%]', '', str(s).lower())
            
        norm_cell = normalize(cell_text)
        aliases = InvoiceParser.COLUMN_MAP.get(target_key, [])
        
        # Fields that are structural and should NOT match as a substring of others
        # (e.g., "qty" should not match "sch qty", "tax" should not match "taxable value")
        structural_fields = {
            "qty", "sch", "rate", "mrp", "sno", "rack", "hsn", "pack", 
            "batch", "expiry", "disc_per", "sch_disc", "disc", "gst", "taxable_amt", "amount"
        }
        
        for alias in aliases:
            norm_alias = normalize(alias)
            if norm_alias == norm_cell:
                return True
            
            # Substring match only for non-structural fields OR if the cell contains the alias as a discrete word
            if target_key not in structural_fields:
                if norm_alias in norm_cell or norm_cell in norm_alias:
                    if len(norm_alias) > 2:
                        return True
        return False

    @staticmethod
    def parse_line_items(tables: List[List[List[str]]], mode: str = "admin") -> List[Dict[str, Any]]:
        """
        Identifies columns dynamically using fuzzy keyword matching.
        """
        # Fields to extract for each mode
        if mode == "admin":
            target_fields = [
                "description", "p_code", "hsn", "batch", "expiry", "qty", 
                "pack", "mrp", "ptr", "pts", "disc_per", "rate", 
                "taxable_amt", "free", "sgst", "cgst", "igst"
            ]
        else: # retailer
            target_fields = [
                "sno", "rack", "mfr", "description", "pack", "hsn", "batch", 
                "expiry", "qty", "sch", "mrp", "rate", "sch_disc", "disc_per", "disc", "gst", "taxable_amt", "amount"
            ]

        extracted_items = []
        
        for table in tables:
            if not table or len(table) < 2:
                continue
            
            # 1. Find the header row and identify column indexes
            header_idx = -1
            col_map = {field: -1 for field in target_fields}
            
            for i, row in enumerate(table):
                row_str = " ".join([str(c) for c in row if c]).lower()
                # Use a balanced set of keywords to find the header row
                if any(k in row_str for k in ["batch", "exp", "hsn", "mrp", "qty"]):
                    header_idx = i
                    for j, cell in enumerate(row):
                        if not cell: continue
                        for field in target_fields:
                            if InvoiceParser._is_column_match(cell, field):
                                col_map[field] = j
                                # Note: We don't break here because one cell might match multiple (unlikely but safe)
                    break
            
            # 2. Extract data if we found the necessary columns (at least description and batch/exp)
            if header_idx != -1 and (col_map["description"] != -1):
                for row in table[header_idx + 1:]:
                    if not any(row): continue
                    
                    # Prepare raw values for each field
                    raw_data = {}
                    for field, idx in col_map.items():
                        if idx != -1 and idx < len(row) and row[idx]:
                            raw_data[field] = str(row[idx]).strip().split('\n')
                        else:
                            raw_data[field] = ["N/A"]

                    # Determine the number of sub-rows (take the maximum lines found in any column)
                    num_lines = max(len(lines) for lines in raw_data.values())
                    
                    for i in range(num_lines):
                        item = {}
                        for field in target_fields:
                            lines = raw_data[field]
                            item[field] = lines[i].strip() if i < len(lines) and lines[i].strip() else "N/A"
                        
                        # Filtering rule: remove if description is N/A or it has mostly N/A values
                        vals = list(item.values())
                        na_count = vals.count("N/A")
                        
                        if item["description"] != "N/A":
                            # Description merging logic:
                            # If most other fields are N/A, it's likely a continuation of the previous item's description
                            is_continuation = na_count >= (len(target_fields) - 2)
                            
                            if is_continuation and extracted_items:
                                prev_desc = extracted_items[-1]["description"]
                                extracted_items[-1]["description"] = f"{prev_desc} {item['description']}".strip()
                            elif not is_continuation:
                                extracted_items.append(item)
                        
        return extracted_items

    @staticmethod
    def _extract_invoice_number(text: str) -> str:
        # Common non-invoice words to ignore
        blacklist = {"copy", "original", "duplicate", "triplicate", "tax", "invoice", "memo", "bill"}
        
        # 1. Look for labeled patterns first (Invoice No: XYZ)
        labeled_patterns = [
            r"(?:Invoice|Inv|Bill|Memo)(?:\s*(?:#|No|Number|Dt))?[:\.\s]+([A-Z0-9\-/]+)",
            r"(?:Invoice|Inv|Bill|Memo)(?:\s*(?:#|No|Number))?\s+([A-Z0-9\-/]+)"
        ]
        
        for pattern in labeled_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                val = match.group(1).strip()
                # Must not be in blacklist, must contain at least one digit if it's not a known prefix
                if val.lower() not in blacklist and any(c.isdigit() for c in val) and len(val) >= 3:
                    return val

        # 2. Look for standalone alphanumeric patterns often used for invoice numbers
        standalone_patterns = [
            r"\b[A-Z]{1,2}\d*[/-]\d+(?:[/-][A-Z0-9]+)*\b", # PJ/123-456 or G2667 (if matched by label later)
            r"\bINV[/-][A-Z0-9/-]+\b",                   # INV-2024-001
        ]

        for pattern in standalone_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                val = match.group(0).strip()
                if val.lower() not in blacklist and any(c.isdigit() for c in val) and len(val) >= 3:
                    return val

        # 3. Fallback: specific "No: " label if it looks like a number
        matches = re.finditer(r"No[:\.\s]+([A-Z0-9\-/]+)", text, re.IGNORECASE)
        for match in matches:
             val = match.group(1).strip()
             if val.lower() not in blacklist and any(c.isdigit() for c in val) and len(val) >= 3:
                 return val

        return "Not Found"

    @staticmethod
    def _extract_date(text: str) -> str:
        # 1. Look for labeled dates first (most reliable)
        date_labels = [r"Inv\.Date", r"Inv Date", r"Date", r"Inv Dt", r"Billing Date"]
        for label in date_labels:
            pattern = rf"{label}\s*[:\s]+(\d{{1,2}}[-/]\d{{1,2}}[-/]\d{{2,4}})"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # 2. Fallback to standalone date pattern
        # Use more restrictive separators and character boundaries to avoid matching invoice numbers like 25/26-0048
        patterns = [
            r"\b\d{1,2}-\d{1,2}-\d{4}\b",  # 16-09-2025
            r"\b\d{1,2}/\d{1,2}/\d{4}\b",  # 16/09/2025
            r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", # fallback
            r"[A-Z][a-z]+\s+\d{1,2},?\s+\d{4}" # October 16, 2025
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # Basic validation: avoid obviously wrong matches picked from alphanumeric strings
                val = match.group(0).strip()
                if "/" in val and "-" in val: # Avoid mixed separators typical of invoice numbers
                    continue
                return val

        return "Not Found"

    @staticmethod
    def _extract_total_amount(text: str, mode: str = "admin") -> str:
        # Prioritize Net Payable/Amount as they are usually the final totals
        # Using more flexible whitespace and colon handling
        patterns = [
            r"(?:Net\s*Payable|Net\s*Amount|Total\s*Payable|Grand\s*Total|Total\s*Amount|Amount\s*Due|Total)\s*[:\s]*([\$£€₹]?\s*[\d,]+\.\d{2})"
        ]
        
        for pattern in patterns:
            # Using finditer to find the last occurrence which is often the final total at the bottom
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                # Take the last match in the document
                return matches[-1].group(1).strip()
                
        return "Not Found"

    @staticmethod
    def _extract_gstin(text: str) -> str:
        # GSTIN format: 15 characters (e.g., 22AAAAA0000A1Z5)
        pattern = r"\b\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}\b"
        match = re.search(pattern, text)
        return match.group(0) if match else "Not Found"

    @staticmethod
    def _extract_currency(text: str) -> str:
        if "₹" in text or "INR" in text:
            return "INR"
        if "$" in text or "USD" in text:
            return "USD"
        if "€" in text or "EUR" in text:
            return "EUR"
        return "Unknown"

if __name__ == "__main__":
    test_text = """
    Invoice NO: INV-2024-001
    Date: 25/02/2026
    Total: $1,500.00
    """
    print(InvoiceParser.parse(test_text))
