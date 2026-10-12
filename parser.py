import re
from typing import Dict, Any, List, Optional, Tuple

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
        "batch": [
            "batch", "batch no", "batch number", "b.no", "lot no", "lot", "batchno",
            "b no", "b. no", "lot.no", "b.num", "batch/lot", "batch/exp", "b.no / exp", "batch / exp"
        ],
        "expiry": [
            "exp", "expiry", "exp date", "expiry date", "exp.", "e.date", "expdt", 
            "exp.date", "valid till", "exp dt", "exp.dt", "e.dt", "edate", "edt",
            "validity", "valid upto", "val till", "use before", "ubd", "batch/exp", "b.no / exp", "batch / exp"
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
        taxable_amt = InvoiceParser._extract_taxable_amount(text)
        net_amt = InvoiceParser._extract_net_amount(text)
        inv_no = InvoiceParser._extract_invoice_number(text)
        inv_date = InvoiceParser._extract_date(text)
        gstin = InvoiceParser._extract_gstin(text)
        dl = InvoiceParser._extract_dl(text)

        data = {
            "invoice_metadata": {
                "invoice_number": inv_no,
                "invoice_no": inv_no,
                "date": inv_date,
                "invoice_date": inv_date,
                "total_amount": InvoiceParser._extract_total_amount(text, mode),
                "taxable_amount": taxable_amt,
                "net_amount": net_amt,
                "gstin": gstin,
                "extracted_gstin": gstin,
                "dl": dl,
                "extracted_dl": dl,
                "currency": InvoiceParser._extract_currency(text)
            },
            "line_items": []
        }
        
        if tables:
            data["line_items"] = InvoiceParser.parse_line_items(tables, mode)
            
        return data

    @staticmethod
    def _parse_and_verify_expiry(val: str) -> Optional[str]:
        """
        Validates that val contains a plausible expiry date and normalizes it to MM/YY.
        Rejects non-dates such as tax rates, price amounts, HSN codes, or invalid months/years.
        """
        if not val or not str(val).strip():
            return None

        cleaned = str(val).strip()
        cleaned_upper = cleaned.upper()

        if cleaned_upper in ["N/A", "NA", "NONE", "NIL", "-", "--", "NULL", "BLANK"]:
            return None

        month_names = {
            "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
            "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
        }
        m_regex = r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)"

        # 1. Named Month with Year: e.g. APR/28, APR-28, APR 2028, APR.28, OCT/26
        m = re.search(r'\b' + m_regex + r'[\./\s-]+(?:20)?(\d{2})\b', cleaned_upper)
        if m:
            m_str = m.group(1)[:3]
            if m_str in month_names:
                month = month_names[m_str]
                yy = int(m.group(2))
                return f"{month:02d}/{yy:02d}"

        # 2. Year with Named Month: e.g. 28-APR, 2028-APR, 28/APR, 26-OCT
        m = re.search(r'\b(?:20)?(\d{2})[\./\s-]+' + m_regex + r'\b', cleaned_upper)
        if m:
            m_str = m.group(2)[:3]
            if m_str in month_names:
                month = month_names[m_str]
                yy = int(m.group(1))
                return f"{month:02d}/{yy:02d}"

        # 3. ISO format: YYYY-MM-DD or YYYY/MM/DD
        m = re.search(r'\b(20\d{2})[-/](\d{1,2})[-/](\d{1,2})\b', cleaned_upper)
        if m:
            yy = int(m.group(1)) % 100
            month = int(m.group(2))
            if 1 <= month <= 12:
                return f"{month:02d}/{yy:02d}"

        # 4. Full date: DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
        m = re.search(r'\b(\d{1,2})[\./-](\d{1,2})[\./-](20\d{2})\b', cleaned_upper)
        if m:
            d, month, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            yy = yyyy % 100
            if 1 <= month <= 12 and 1 <= d <= 31:
                return f"{month:02d}/{yy:02d}"

        # 5. MM/YYYY or MM-YYYY or MM.YYYY (4-digit year)
        m = re.search(r'\b(\d{1,2})[\./\s-]+(20\d{2})\b', cleaned_upper)
        if m:
            month = int(m.group(1))
            yy = int(m.group(2)) % 100
            if 1 <= month <= 12 and 20 <= yy <= 45:
                return f"{month:02d}/{yy:02d}"

        # 6. YYYY/MM or YYYY-MM (4-digit year first)
        m = re.search(r'\b(20\d{2})[\./\s-]+(\d{1,2})\b', cleaned_upper)
        if m:
            yy = int(m.group(1)) % 100
            month = int(m.group(2))
            if 1 <= month <= 12 and 20 <= yy <= 45:
                return f"{month:02d}/{yy:02d}"

        # 7. MM/YY or MM-YY or MM.YY or MM / YY (2-digit year)
        m = re.search(r'\b(\d{1,2})\s*[\./-]\s*(\d{2})\b', cleaned_upper)
        if m:
            p1 = int(m.group(1))
            p2 = int(m.group(2))
            if 1 <= p1 <= 12 and 20 <= p2 <= 45:
                return f"{p1:02d}/{p2:02d}"
            elif 20 <= p1 <= 45 and 1 <= p2 <= 12:
                return f"{p2:02d}/{p1:02d}"

        # 8. Contiguous 4 digits: MMYY (e.g. 0527)
        m = re.search(r'\b(\d{2})(\d{2})\b', cleaned_upper)
        if m:
            p1, p2 = int(m.group(1)), int(m.group(2))
            if 1 <= p1 <= 12 and 24 <= p2 <= 40:
                return f"{p1:02d}/{p2:02d}"

        return None

    @staticmethod
    def _extract_embedded_expiry(text: str) -> Tuple[Optional[str], str]:
        """
        Extracts an embedded expiry date from a combined string (e.g., in batch or description),
        returning (verified_expiry, cleaned_text).
        """
        if not text:
            return None, text
        exp_val = InvoiceParser._parse_and_verify_expiry(text)
        if exp_val:
            cleaned = re.sub(r'(?:EXP(?:IRY)?[:\.\s]*)?\b\d{1,2}\s*[\./-]\s*(?:20)?\d{2}\b', '', text, flags=re.IGNORECASE)
            cleaned = re.sub(r'(?:EXP(?:IRY)?[:\.\s]*)?\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[a-z]*[\./\s-]+(?:20)?\d{2}\b', '', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'[\s/,-]+$', '', cleaned).strip()
            cleaned = re.sub(r'^[\s/,-]+', '', cleaned).strip()
            return exp_val, (cleaned if cleaned else "N/A")
        return None, text

    @staticmethod
    def _is_column_match(cell_text: str, target_key: str) -> bool:
        """
        Checks if a cell text matches any of the aliases for a target column.
        Handles punctuation, spacing, and partial matches for merged headers.
        """
        if not cell_text: return False
        
        # Normalize: lower, strip dots, spaces, slashes, hyphens, and percent signs
        def normalize(s):
            return re.sub(r'[\.\s%/\-_]', '', str(s).lower())
            
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
        Extracts line items from tables, verifying that every product line item's
        expiry date strictly belongs to that specific product row in the original PDF.
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
            
            # 1. Header row identification by scoring matches
            best_header_idx = -1
            best_col_map = None
            best_score = -1

            for i, row in enumerate(table[:15]):
                if not any(row):
                    continue
                col_map = {field: -1 for field in target_fields}
                matched_count = 0
                for j, cell in enumerate(row):
                    if not cell: continue
                    for field in target_fields:
                        if InvoiceParser._is_column_match(cell, field):
                            if col_map[field] == -1:
                                col_map[field] = j
                                matched_count += 1

                row_str = " ".join([str(c).lower() for c in row if c])
                kw_count = sum(1 for kw in ["batch", "exp", "hsn", "mrp", "qty", "rate", "item", "product", "description", "particulars"] if kw in row_str)
                score = matched_count * 2 + kw_count

                if score > best_score and (matched_count >= 2 or kw_count >= 2):
                    best_score = score
                    best_header_idx = i
                    best_col_map = col_map

            if best_header_idx == -1 or not best_col_map:
                continue

            header_idx = best_header_idx
            col_map = best_col_map
            header_row = table[header_idx]

            # Resolve description column if missing from unmapped header cells
            if col_map["description"] == -1:
                used = set(v for v in col_map.values() if v != -1)
                for j, cell in enumerate(header_row):
                    if j not in used and cell:
                        c_clean = str(cell).lower()
                        if any(k in c_clean for k in ["item", "product", "desc", "particulars", "name"]):
                            col_map["description"] = j
                            break

            # Resolve expiry column if missing (check header text for exp / valid / edt / ubd)
            if col_map["expiry"] == -1:
                for j, cell in enumerate(header_row):
                    if cell:
                        c_clean = re.sub(r'[\.\s%/\-_]', '', str(cell).lower())
                        if any(k in c_clean for k in ["exp", "expiry", "val", "valid", "ubd", "edt", "edate"]):
                            col_map["expiry"] = j
                            break

            # Data-driven expiry column fallback: inspect data rows to identify column containing dates
            if col_map["expiry"] == -1:
                col_date_counts = {}
                for row in table[header_idx + 1: header_idx + 15]:
                    for j, cell in enumerate(row):
                        if cell and j != col_map.get("description", -1):
                            exp_val = InvoiceParser._parse_and_verify_expiry(str(cell))
                            if exp_val:
                                col_date_counts[j] = col_date_counts.get(j, 0) + 1
                if col_date_counts:
                    best_exp_col = max(col_date_counts.items(), key=lambda x: x[1])[0]
                    if col_date_counts[best_exp_col] >= 1:
                        col_map["expiry"] = best_exp_col

            # Skip table if description column could not be found
            if col_map["description"] == -1:
                continue

            # 2. Extract line items row by row, strictly preserving row-to-product alignment
            for row in table[header_idx + 1:]:
                if not any(row):
                    continue

                # Skip summary or total rows
                first_cells = " ".join([str(c).lower().strip() for c in row[:3] if c])
                if any(k in first_cells for k in ["total", "grand total", "net payable", "sub total", "taxable value", "round off"]):
                    continue

                # Skip repeated headers
                row_str_check = " ".join([str(c).lower() for c in row if c])
                if "hsn" in row_str_check and "qty" in row_str_check and "batch" in row_str_check:
                    continue

                # Prepare cell lines
                cell_lines = {}
                for field in target_fields:
                    idx = col_map[field]
                    if idx != -1 and idx < len(row) and row[idx] is not None:
                        lines = [line.strip() for line in str(row[idx]).split('\n') if line.strip()]
                        cell_lines[field] = lines if lines else []
                    else:
                        cell_lines[field] = []

                # Check if this row is a collapsed multi-product block
                multi_line_counts = sum(1 for f in ["qty", "rate", "batch", "sno", "mrp", "amount"] if len(cell_lines.get(f, [])) > 1)
                is_collapsed = (multi_line_counts >= 2)

                if is_collapsed:
                    num_items = max(len(cell_lines.get(f, [])) for f in ["qty", "rate", "batch", "sno", "amount"] if cell_lines.get(f, []))
                    desc_lines = cell_lines.get("description", [])
                    exp_lines = cell_lines.get("expiry", [])

                    for k in range(num_items):
                        item = {}
                        for field in target_fields:
                            lines = cell_lines.get(field, [])
                            if k < len(lines) and lines[k]:
                                item[field] = lines[k]
                            else:
                                item[field] = "N/A"

                        # Verify expiry for product k in collapsed block
                        if len(exp_lines) == num_items and k < len(exp_lines):
                            item["expiry"] = InvoiceParser._parse_and_verify_expiry(exp_lines[k]) or "N/A"
                        else:
                            # Mismatched line count: verify if batch has embedded expiry, else preserve N/A
                            batch_val = item.get("batch", "N/A")
                            if batch_val != "N/A":
                                emb_exp, clean_b = InvoiceParser._extract_embedded_expiry(batch_val)
                                if emb_exp:
                                    item["expiry"] = emb_exp
                                    item["batch"] = clean_b
                                else:
                                    item["expiry"] = "N/A"
                            else:
                                item["expiry"] = "N/A"

                        if item.get("description", "N/A") != "N/A":
                            extracted_items.append(item)

                else:
                    # Standard grid row (1 product row in PDF)
                    item = {}
                    for field in target_fields:
                        idx = col_map[field]
                        if idx != -1 and idx < len(row) and row[idx] is not None:
                            val = " ".join(part for part in str(row[idx]).split('\n') if part.strip())
                            item[field] = val if val and val.upper() not in ["N/A", "NA", "NONE", "NIL", "-", "--"] else "N/A"
                        else:
                            item[field] = "N/A"

                    # Check for description continuation row:
                    # If description is present, but ALL transactional fields are N/A:
                    desc_val = item.get("description", "N/A")
                    if desc_val != "N/A":
                        transactional_fields = ["qty", "rate", "mrp", "amount", "batch", "expiry", "sno"]
                        has_trans = any(item.get(f, "N/A") != "N/A" for f in transactional_fields if f in target_fields)
                        if not has_trans and extracted_items:
                            # Append to previous product's description without shifting fields
                            extracted_items[-1]["description"] = f"{extracted_items[-1]['description']} {desc_val}".strip()
                            continue

                    # Verify expiry date strictly belongs to THIS product row
                    raw_exp = item.get("expiry", "N/A")
                    verified_exp = InvoiceParser._parse_and_verify_expiry(raw_exp) if raw_exp != "N/A" else None

                    # Check if batch and expiry share the same column or if batch has embedded expiry
                    batch_val = item.get("batch", "N/A")
                    if batch_val != "N/A":
                        emb_exp, clean_b = InvoiceParser._extract_embedded_expiry(batch_val)
                        if emb_exp:
                            if not verified_exp:
                                verified_exp = emb_exp
                            item["batch"] = clean_b

                    # If expiry column was identical to batch column, and no expiry found yet
                    if col_map.get("batch") == col_map.get("expiry") and col_map.get("expiry") != -1:
                        if not verified_exp:
                            item["expiry"] = "N/A"
                        else:
                            item["expiry"] = verified_exp

                    # If still not found, check description column of THIS SAME ROW
                    if not verified_exp:
                        desc_val = item.get("description", "N/A")
                        if desc_val != "N/A" and "EXP" in desc_val.upper():
                            emb_exp, clean_d = InvoiceParser._extract_embedded_expiry(desc_val)
                            if emb_exp:
                                verified_exp = emb_exp

                    item["expiry"] = verified_exp if verified_exp else "N/A"

                    if item.get("description", "N/A") != "N/A":
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
    def _extract_dl(text: str) -> Optional[str]:
        # Drug License pattern e.g., D.L. No.: KL-TVM-123456 or 20B/21B: 123456
        patterns = [
            r"(?:D\.?\s*L\.?\s*(?:No|Number)?|Drug\s*Lic(?:ense)?\s*(?:No|Number)?|Licence\s*No)[:\.\s]+([A-Z0-9\-\/,\s]+?)(?=\s*(?:GSTIN|PAN|State|Phone|Mob|Email|\n|$))",
            r"\b([A-Z]{2}-[A-Z]{2,4}-\d{5,8})\b",
            r"\b(KL-[A-Z0-9\-]+)\b"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                if len(val) >= 4 and not val.lower().startswith("gst"):
                    return val
        return None

    @staticmethod
    def _extract_taxable_amount(text: str) -> Optional[float]:
        patterns = [
            r"(?:Taxable\s*(?:Amt|Amount|Value)|Total\s*Taxable)\s*[:\s]*([\$£€₹]?\s*[\d,]+\.?\d*)",
            r"(?:Sub\s*Total|Taxable)\s*[:\s]*([\$£€₹]?\s*[\d,]+\.?\d*)"
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                val_str = matches[-1].group(1).strip()
                cleaned = re.sub(r'[^\d.]', '', val_str)
                try:
                    return float(cleaned)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _extract_net_amount(text: str) -> Optional[float]:
        patterns = [
            r"(?:Net\s*Payable|Net\s*Amount|Total\s*Payable|Grand\s*Total|Invoice\s*Total|Total\s*Amount)\s*[:\s]*([\$£€₹]?\s*[\d,]+\.?\d*)"
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if matches:
                val_str = matches[-1].group(1).strip()
                cleaned = re.sub(r'[^\d.]', '', val_str)
                try:
                    return float(cleaned)
                except ValueError:
                    continue
        return None

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
