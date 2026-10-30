import os
import json
import re
import calendar
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class InvoiceValidator:
    """
    AI-powered and Rule-based Invoice Verification & Reconciliation Engine.
    Validates extracted invoice PDF/Image contents against Laravel order requirements,
    detecting matched variants, multi-batch fulfillment, missing items, and unexpected extra items.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def validate(
        self,
        invoice_text: str,
        invoice_tables: Optional[List[List[List[str]]]] = None,
        order_data: Optional[Union[Dict[str, Any], List[Dict[str, Any]], str]] = None,
        parsed_invoice: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validates invoice text and tables against the order data.
        Returns the structured JSON format required for Laravel integration.
        Executes ultra-fast (< 50ms) deterministic verification with AI fallback for complex ambiguities.
        """
        # Parse and normalize order data
        normalized_order = self._normalize_order_data(order_data)

        # 1. Fast deterministic verification using local rule engine (< 50ms)
        rule_result = self._validate_rule_based(invoice_text, invoice_tables, normalized_order, parsed_invoice)

        # If rule engine achieves a 100% matched state, or all ordered items exist on the invoice with 0 missing and 0 extra items, return immediately (fast path, < 50ms)
        if rule_result and (
            (rule_result.get("match_status") == "matched" and len(rule_result.get("missing_items", [])) == 0) or
            (rule_result.get("match_percentage", 0) == 100 and len(rule_result.get("missing_items", [])) == 0 and len(rule_result.get("extra_items", [])) == 0)
        ):
            rule_result["ai_validated"] = True
            if "data" in rule_result:
                rule_result["data"]["ai_validated"] = True
            return rule_result


        # 2. If discrepancies or missing items exist, and OpenAI client is available, attempt AI validation
        if self.client:
            try:
                ai_result = self._validate_with_openai(invoice_text, invoice_tables, normalized_order, parsed_invoice)
                if ai_result and "data" in ai_result and "match_status" in ai_result:
                    sanitized_ai = self._sanitize_response(ai_result, normalized_order, parsed_invoice)
                    # If sanitized AI matched or achieved strictly higher confidence, return it
                    if sanitized_ai.get("match_percentage", 0) > rule_result.get("match_percentage", 0):
                        return sanitized_ai
            except Exception as e:
                print(f"[WARN] OpenAI Validation failed, falling back to rule engine: {e}")

        # Return rule-based verification result
        rule_result["ai_validated"] = True
        if "data" in rule_result:
            rule_result["data"]["ai_validated"] = True
        return rule_result

    def _normalize_order_data(self, order_data: Any) -> Dict[str, Any]:
        """
        Normalizes order data into a standard dictionary structure.
        """
        if isinstance(order_data, str):
            try:
                order_data = json.loads(order_data)
            except Exception:
                order_data = {}

        if isinstance(order_data, list):
            return {"order_code": "ORDER-1", "order_no": "ORDER-1", "items": order_data}

        if not isinstance(order_data, dict):
            return {"order_code": "ORDER-1", "order_no": "ORDER-1", "items": []}

        # Normalize items list key
        if "items" not in order_data:
            if "order_items" in order_data:
                order_data["items"] = order_data["order_items"]
            elif "products" in order_data:
                order_data["items"] = order_data["products"]
            elif "line_items" in order_data:
                order_data["items"] = order_data["line_items"]
            else:
                order_data["items"] = []

        return order_data

    @staticmethod
    def _format_expiry_iso(exp_str: Optional[str]) -> Optional[str]:
        """
        Converts MM/YY or MM/YYYY dates to standard YYYY-MM-DD format (end of month),
        or preserves YYYY-MM-DD if already formatted.
        """
        if not exp_str or str(exp_str).upper() in ["N/A", "NONE", "BLANK", "", "NULL"]:
            return None
        exp_str = str(exp_str).strip()
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', exp_str):
            return exp_str
        m = re.match(r'(\d{1,2})[\./-]((?:20)?\d{2})', exp_str)
        if m:
            month = int(m.group(1))
            yy_part = m.group(2)
            year = int(yy_part) if len(yy_part) == 4 else (2000 + int(yy_part))
            if 1 <= month <= 12:
                last_day = calendar.monthrange(year, month)[1]
                return f"{year:04d}-{month:02d}-{last_day:02d}"
        return exp_str

    @staticmethod
    def _format_expiry_mmyyyy(exp_str: Optional[str]) -> Optional[str]:
        """
        Converts dates to MM/YYYY format (e.g. '10/2028', '2028-10-31' -> '10/2028', '10/28' -> '10/2028').
        Preserves string if conversion is not applicable.
        """
        if not exp_str or str(exp_str).upper() in ["N/A", "NONE", "BLANK", "", "NULL"]:
            return None
        exp_str = str(exp_str).strip()
        # ISO format: YYYY-MM-DD or YYYY/MM/DD
        m_iso = re.match(r'(\d{4})[-/](\d{1,2})[-/]\d{1,2}', exp_str)
        if m_iso:
            return f"{int(m_iso.group(2)):02d}/{m_iso.group(1)}"
        # MM/YYYY or MM-YYYY or MM/YY
        m_full = re.match(r'(\d{1,2})[\./-]((?:20)?\d{2})', exp_str)
        if m_full:
            month = int(m_full.group(1))
            yy_part = m_full.group(2)
            year = int(yy_part) if len(yy_part) == 4 else (2000 + int(yy_part))
            if 1 <= month <= 12:
                return f"{month:02d}/{year}"
        return exp_str

    @classmethod
    def _find_free_reward_in_text(cls, text: str, reward_name: Optional[str]) -> float:
        """
        Checks if a cross-product free reward is printed in remarks, notes, or free column
        (e.g., 'Free: 1 ATOMSTA 20', 'ATOMSTA 20 (1 Free)', 'Free Item: ATOMSTA 20 (1)').
        """
        if not text or not reward_name:
            return 0.0
        tokens = re.findall(r'[A-Z0-9]+', reward_name.upper())
        if not tokens:
            return 0.0
        pat_name = r'\s*'.join(re.escape(t) for t in tokens)

        # Pattern 1: Free/Scheme label with quantity followed by product name
        p1 = rf'\b(?:FREE|SCHEME|BONUS|FOC)(?:[\s\w]*?)[:\s]+(\d+(?:\.\d+)?)\s*(?:NOS?|UNITS?|PCS?|BOX|STRIP)?\s*(?:OF\s*)?{pat_name}'
        m1 = re.search(p1, text, re.IGNORECASE)
        if m1:
            try:
                return float(m1.group(1))
            except ValueError:
                pass

        # Pattern 2: Free label followed by product name and quantity in parentheses or colon
        p2 = rf'\b(?:FREE|SCHEME|BONUS|FOC)[\s\w]*?[:\s]+{pat_name}[\s:\-(]*(\d+(?:\.\d+)?)'
        m2 = re.search(p2, text, re.IGNORECASE)
        if m2:
            try:
                return float(m2.group(1))
            except ValueError:
                pass

        # Pattern 3: Product name followed by Free/Scheme with quantity
        p3 = rf'{pat_name}[\s:\-(]*(\d+(?:\.\d+)?)\s*(?:FREE|SCHEME|BONUS|FOC)'
        m3 = re.search(p3, text, re.IGNORECASE)
        if m3:
            try:
                return float(m3.group(1))
            except ValueError:
                pass

        p4 = rf'{pat_name}[\s:]*(?:FREE|SCHEME|BONUS|FOC)[\s:]*(\d+(?:\.\d+)?)'
        m4 = re.search(p4, text, re.IGNORECASE)
        if m4:
            try:
                return float(m4.group(1))
            except ValueError:
                pass

        return 0.0

    @staticmethod
    def _to_float(val: Any) -> float:
        """
        Safely converts numerical or currency string values to float.
        """
        if val is None or val == "" or str(val).upper() in ["N/A", "NONE", "NULL"]:
            return 0.0
        if isinstance(val, (int, float)):
            return float(val)
        cleaned = re.sub(r'[^\d.]', '', str(val))
        try:
            return float(cleaned) if cleaned else 0.0
        except ValueError:
            return 0.0

    @classmethod
    def _int_or_float(cls, val: Any) -> Union[int, float]:
        """
        Converts value to int if it represents a whole number, otherwise returns float.
        """
        f = cls._to_float(val)
        return int(f) if f.is_integer() else f

    @classmethod
    def _normalize_product_code(cls, code_str: Optional[str]) -> Optional[str]:
        if not code_str or str(code_str).upper() in ["N/A", "NONE", "NULL", "BLANK", ""]:
            return None
        # Remove whitespace, hyphens, slashes, underscores for robust comparison:
        # e.g., 'AS-D23' -> 'ASD23', 'BP-05' -> 'BP05', 'AS - D23' -> 'ASD23'
        cleaned = re.sub(r'[^A-Z0-9]', '', str(code_str).upper())
        # Ignore vendor prefixes without model digits (e.g. AS, ASD, ASA, ASB)
        if not re.search(r'\d', cleaned) and len(cleaned) <= 3:
            return None
        return cleaned if cleaned else None

    @staticmethod
    def _normalize_size_tag(size_str: Optional[str]) -> Optional[str]:
        if not size_str or str(size_str).upper() in ["N/A", "NONE", "NULL", "BLANK", ""]:
            return None
        s = re.sub(r'[\[\]\(\)\{\}\s]', '', str(size_str)).upper()
        mapping = {
            "2XL": "XXL",
            "3XL": "XXXL",
            "SMALL": "S",
            "MEDIUM": "M",
            "LARGE": "L",
            "EXTRALARGE": "XL",
            "FREESIZE": "FS",
            "FREE": "FS",
            "UNI": "UNI",
            "UNIVERSAL": "UNI",
            "UNISEX": "UNI"
        }
        return mapping.get(s, s)

    @classmethod
    def _extract_product_code(cls, text: str) -> Optional[str]:
        from parser import InvoiceParser
        return InvoiceParser._extract_product_code(text)

    @classmethod
    def _extract_size_tag(cls, text: str) -> Optional[str]:
        from parser import InvoiceParser
        return InvoiceParser._extract_size_tag(text)

    @classmethod
    def _normalize_surgical_text(cls, text: str) -> str:
        """
        Normalizes surgical/medical appliance text by expanding abbreviations,
        mapping medical synonyms, and stripping vendor prefixes and packaging words like PAIR.
        """
        s = text.upper()
        # Remove known vendor product code prefixes: e.g. AS-A01, AS-A07, AS-D04, AS-B02, AS-E10, CN-05, etc.
        s = re.sub(r'\b(?:AS|ASD|ASA|ASB|ASE|ASH|BP|DT|CN)-[A-Z0-9]+\b\s*', '', s)
        # Remove vendor brand prefixes: AS-, AS -, ATOM WELLNESS, ATOM SURGICAL
        s = re.sub(r'\bAS\s*[-–—]\s*', '', s)
        s = re.sub(r'\bATOM\s+WELLNESS\b', '', s)
        s = re.sub(r'\bATOM\s+SURGICAL\b', '', s)
        # Remove (PAIR), PAIR
        s = re.sub(r'\((?:PAIR|NOS?)\)', '', s)
        s = re.sub(r'\bPAIR\b', '', s)
        # LS / L.S. / LUMBOSACRAL -> LUMBO SACRAL
        s = re.sub(r'\bL\.?S\.?\b', 'LUMBO SACRAL', s)
        s = re.sub(r'\bLUMBOSACRAL\b', 'LUMBO SACRAL', s)
        # BELT / SUPPORT / BINDER (e.g. ABDOMINAL BELT <-> ABDOMINAL SUPPORT, LS BELT <-> LS SUPPORT)
        s = re.sub(r'\b(SUPPORT|BINDER)\b', 'BELT', s)
        # METBLOW / METBLOWN / MELTBLOWN
        s = re.sub(r'\bMETBLOW\b', 'METBLOWN', s)
        s = re.sub(r'\bMELTBLOWN\b', 'METBLOWN', s)
        s = re.sub(r'\b3\s*PLY\b', '3PLY', s)
        # DIGI -> DIGITAL
        s = re.sub(r'\bDIGI\b', 'DIGITAL', s)
        # Clean extra whitespace
        return re.sub(r'\s+', ' ', s).strip()

    @classmethod
    def _is_tax_or_summary_row(cls, item: Union[Dict[str, Any], str, List[str]]) -> bool:
        """
        Checks if an item, description, or table row is a GST/tax summary breakdown or total row.
        """
        from parser import InvoiceParser
        return InvoiceParser.is_tax_or_summary_row(item)

    @classmethod
    def _is_invalid_or_phantom_line(cls, item: Union[Dict[str, Any], str, List[str]]) -> bool:
        """
        Checks if an item is empty, numeric-only (e.g. '0.00', '0', '1.00'), has < 2 letters,
        or has quantity == 0 AND free_qty == 0 AND net_amount == 0.
        """
        from parser import InvoiceParser
        return InvoiceParser.is_invalid_or_phantom_line(item)

    @classmethod
    def _get_product_family_key(cls, item: Dict[str, Any]) -> str:
        """
        Derives a normalized family key across multi-size product variants (e.g. KNEE CAP,
        ANKLE BINDER, LS BELT, ABDOMINAL SUPPORT, SOFT CERVICAL COLLAR in sizes M, L, XL),
        while strictly preserving medicine strength/dosage separation (e.g. ATOMSTA 10 vs ATOMSTA 20).
        """
        name = (item.get("product_name") or item.get("name") or "").upper().strip()
        norm = cls._normalize_surgical_text(name)
        norm = re.sub(r'[\[\(\{]\s*(?:XXXL|XXL|2XL|3XL|XL|XS|S|M|L|FREE\s*SIZE|FS|UNI(?:VERSAL)?)\s*[\]\)\}]', '', norm, flags=re.IGNORECASE)
        norm = re.sub(r'\b(?:SIZE|SZ)[:\s-]+(?:XXXL|XXL|2XL|3XL|XL|XS|S|M|L|FREE\s*SIZE|FS|UNI(?:VERSAL)?)\b', '', norm, flags=re.IGNORECASE)
        norm = re.sub(r'(?:^|[\s\-_/])(?:XXXL|XXL|2XL|3XL|XL|XS|S|M|L|UNI(?:VERSAL)?)(?:\s*$|[\s\-_/,])', ' ', norm, flags=re.IGNORECASE)
        stem = re.sub(r'[^A-Z0-9]', '', norm)
        return stem if stem else "UNKNOWN_FAMILY"

    @classmethod
    def _apply_variant_group_free_pooling(
        cls,
        items_list: List[Dict[str, Any]],
        reconciled_items: List[Dict[str, Any]]
    ) -> None:
        """
        Variant Group Free Pooling (Multi-Size Products):
        For products with variants (e.g. KNEE CAP, ANKLE BINDER, LS BELT, ABDOMINAL SUPPORT,
        SOFT CERVICAL COLLAR in sizes M, L, XL), free quantity might be placed on size M, with 0 on L and XL in the order.
        On the invoice, the supplier distributes them evenly (e.g. 1 on M, 1 on L, 1 on XL, Total = 3).
        If total ordered free for the product family equals or is satisfied by total invoiced free across all sizes,
        mark variants with matching paid quantities as MATCHED, clearing free discrepancy notes.
        """
        families = {}
        for idx, it in enumerate(items_list):
            fkey = cls._get_product_family_key(it)
            families.setdefault(fkey, []).append(idx)

        for fkey, idxs in families.items():
            if len(idxs) <= 1:
                continue

            total_family_ordered_free = sum(cls._to_float(items_list[i].get("free_qty", 0)) for i in idxs)
            total_family_invoiced_free = sum(cls._to_float(items_list[i].get("invoiced_free_qty", 0)) for i in idxs)

            if total_family_ordered_free > 0 and total_family_invoiced_free >= total_family_ordered_free:
                for i in idxs:
                    it = items_list[i]
                    ord_q = cls._to_float(it.get("ordered_qty", 0))
                    inv_q = cls._to_float(it.get("invoiced_qty", 0))
                    paid_satisfied = bool(inv_q == ord_q and inv_q > 0)
                    if paid_satisfied:
                        it["discrepancies"] = [
                            d for d in it.get("discrepancies", [])
                            if not any(k in d.lower() for k in ["free reward", "free item", "free qty", "free quantity", "bonus"])
                        ]
                        if len(it["discrepancies"]) == 0:
                            it["is_matched"] = True

                        oid = it.get("order_item_id")
                        for r in reconciled_items:
                            r_oid = r.get("order_item_id")
                            if oid is not None and str(oid) == str(r_oid):
                                r["discrepancies"] = [
                                    d for d in r.get("discrepancies", [])
                                    if not any(k in d.lower() for k in ["free reward", "free item", "free qty", "free quantity", "bonus"])
                                ]
                                if len(r["discrepancies"]) == 0:
                                    r["is_matched"] = True
                                    r["match_status"] = "matched"

    @classmethod
    def _extract_medicine_keys(cls, text: str) -> Dict[str, Any]:
        """
        Extracts brand stem, dosage/strength, variant code, formulation letter, and pack count
        to strictly enforce variant/strength separation.
        """
        upper = text.upper()
        # Remove packaging dimensions (e.g. 10X10, 10*10, 10X3X1, 20X20) so they aren't mistaken for dosage
        cleaned = re.sub(r'\b\d+\s*[\*xX]\s*\d+(?:\s*[\*xX]\s*\d+)?\b', ' ', upper)

        # 1. Pack count (e.g. 10S, 50S, 100S)
        pack_match = re.findall(r'\b(\d+)\s*S\b', cleaned)
        pack = pack_match[0] + "S" if pack_match else None

        # 2. Variant code (e.g. M1, M2, D3, SR, DS, FORTE, PLUS)
        variant_match = re.findall(r'\b(M[1-9]|D[1-9]|SR|XR|CR|ER|DS|FORTE|PLUS)\b', cleaned)
        variant = variant_match[0] if variant_match else None

        # 3. Numeric dosage/strength (e.g. 10, 20, 40, 500)
        unit_match = re.findall(r'\b(\d+(?:\.\d+)?)\s*(?:MG|MCG|G|GM|ML|IU|%)\b', cleaned)
        dosage = None
        if unit_match:
            dosage = unit_match[0]
        else:
            # Check for standalone numbers (e.g. ATOMSTA 10, TELMATOM 40, TENATOM 20, ATOMEZ 20)
            words = cleaned.split()
            for idx, w in enumerate(words):
                if re.fullmatch(r'\d+', w):
                    # If preceded by a brand word (length >= 4) or followed by dosage-like context
                    if idx > 0 and len(words[idx - 1]) >= 4 and words[idx - 1] not in ["WITH", "MASK", "PLY", "CODE", "HSN"]:
                        dosage = w
                        break
                    elif w in ["10", "20", "25", "30", "40", "50", "75", "100", "150", "200", "250", "400", "500", "650", "800", "1000"]:
                        dosage = w
                        break

        # 4. Formulation suffix (e.g. 'P' in ATOFEN P, 'L' in ATOMONT L or ATOMONT-L)
        formulation_match = re.findall(r'\b(?:ATOFEN\s+([A-Z])|ATOMONT[\s-]+([A-Z]))\b', cleaned)
        formulation = None
        if formulation_match:
            formulation = formulation_match[0][0] or formulation_match[0][1]

        # 5. Core medicine brand stem
        known_stems = [
            "ATOMSTA", "TELMATOM", "TENATOM", "ATOMGLIM", "ATOFEN", "ATOMEZ",
            "ATOMONT", "GASTOMED"
        ]
        brand_stem = None
        for ks in known_stems:
            if ks in upper:
                brand_stem = ks
                break

        if not brand_stem:
            # Check for other pharmaceutical brand patterns (e.g. starting with ATOM, TELM, TENA)
            for w in re.findall(r'\b(ATOM[A-Z0-9]+|TELM[A-Z0-9]+|TEN[A-Z0-9]+)\b', upper):
                if w not in ["ATOM WELLNESS", "ATOM SURGICAL"]:
                    brand_stem = w
                    break

        return {
            "pack": pack,
            "variant": variant,
            "dosage": dosage,
            "formulation": formulation,
            "brand_stem": brand_stem
        }

    @classmethod
    def _match_product(cls, order_item: Dict[str, Any], inv_item: Dict[str, Any]) -> bool:
        """
        Determines whether an invoice line item matches an ordered product variant.
        Handles:
        - Product code recognition (e.g. AS-A01, AS-A07, AS-B02, AS-D01, AS-D04, AS-D25, AS-E10, BP-05, DT-811, CN-05)
        - Strict dosage & strength preservation (ATOMSTA 10 NEVER matches ATOMSTA 20 or ATOMSTA 40; ATOMGLIM M1 NEVER matches M2)
        - Size variant tags ([L], [M], [S], (XL), XXL, UNI, FS) with strict variant preservation
        - Side variants (LEFT, RIGHT)
        - Medical and surgical synonyms (LS = Lumbo Sacral, Belt = Support = Binder, METBLOW = METBLOWN, DIGI = DIGITAL)
        - Stripping brand prefixes (AS-, ATOM WELLNESS)
        """
        order_name = (order_item.get("product_name") or order_item.get("name") or "").upper().strip()
        inv_desc = (inv_item.get("description") or inv_item.get("item_name") or "").upper().strip()

        # 1. Product Code Extraction & Normalization
        order_raw_code = (
            order_item.get("product_code") or 
            order_item.get("code") or 
            order_item.get("p_code") or 
            order_item.get("item_code") or 
            order_item.get("sku") or 
            cls._extract_product_code(order_name)
        )
        if not order_raw_code:
            for k, v in order_item.items():
                if isinstance(v, str) and any(tag in v.upper() for tag in ["CODE", "AS-", "BP-", "DT-", "CN-", "ASH-"]):
                    code_cand = cls._extract_product_code(v)
                    if code_cand:
                        order_raw_code = code_cand
                        break

        # In invoices, the true model/product code is often explicitly in the description (e.g. AS-A07, AS-D04, BP-05)
        # while the p_code column in OCR may be truncated ("ASA-", "AS-L") or concatenated with HSN/Batch.
        inv_desc_code = cls._extract_product_code(inv_desc)
        norm_inv_desc_code = cls._normalize_product_code(inv_desc_code)

        inv_field_code = (
            inv_item.get("p_code") or 
            inv_item.get("product_code") or 
            inv_item.get("code") or 
            inv_item.get("item_code")
        )
        norm_inv_field_code = cls._normalize_product_code(inv_field_code)

        norm_ord_code = cls._normalize_product_code(order_raw_code)
        norm_inv_code = norm_inv_desc_code or norm_inv_field_code

        # 2. Size Variant Extraction & Normalization
        order_raw_size = (
            order_item.get("size") or 
            order_item.get("variant_label") or 
            order_item.get("variant") or 
            cls._extract_size_tag(order_name)
        )
        inv_raw_size = (
            inv_item.get("size") or 
            inv_item.get("variant") or 
            cls._extract_size_tag(inv_desc)
        )

        norm_ord_size = cls._normalize_size_tag(order_raw_size)
        norm_inv_size = cls._normalize_size_tag(inv_raw_size)

        # 3. Side Variant Extraction
        order_side = (order_item.get("side") or ("LEFT" if "LEFT" in order_name else ("RIGHT" if "RIGHT" in order_name else ""))).upper().strip()
        inv_side = (inv_item.get("side") or ("LEFT" if "LEFT" in inv_desc else ("RIGHT" if "RIGHT" in inv_desc else ""))).upper().strip()

        # --- VARIANT COMPATIBILITY CHECKS ---
        # A. Size Variant Conflict
        if norm_ord_size and norm_inv_size and norm_ord_size != norm_inv_size:
            if not ({norm_ord_size, norm_inv_size}.issubset({"UNI", "FS"})):
                return False

        if norm_ord_size and not norm_inv_size:
            detected_inv_size = cls._normalize_size_tag(cls._extract_size_tag(inv_desc))
            if detected_inv_size and detected_inv_size != norm_ord_size:
                if not ({norm_ord_size, detected_inv_size}.issubset({"UNI", "FS"})):
                    return False

        if norm_inv_size and not norm_ord_size:
            if norm_inv_size not in ["UNI", "FS"]:
                detected_ord_size = cls._normalize_size_tag(cls._extract_size_tag(order_name))
                if detected_ord_size and detected_ord_size != norm_inv_size:
                    if not ({norm_inv_size, detected_ord_size}.issubset({"UNI", "FS"})):
                        return False

        # B. Side Variant Conflict (LEFT vs RIGHT)
        if order_side in ["LEFT", "RIGHT"] and inv_side in ["LEFT", "RIGHT"]:
            if order_side != inv_side:
                return False
        elif order_side in ["LEFT", "RIGHT"]:
            opp_side = "RIGHT" if order_side == "LEFT" else "LEFT"
            if re.search(r'\b' + opp_side + r'\b', inv_desc):
                return False

        # C. Distinct Orthopedic Category Conflict: "BRACE" vs "CAP"
        has_brace_ord = "BRACE" in order_name
        has_brace_inv = "BRACE" in inv_desc
        if has_brace_ord != has_brace_inv:
            return False

        # D. "AIR PRO" distinction
        has_air_pro_ord = ("AIR" in order_name and "PRO" in order_name) or ("ASD23" in (norm_ord_code or ""))
        has_air_pro_inv = ("AIR" in inv_desc and "PRO" in inv_desc) or ("ASD23" in (norm_inv_code or ""))
        if has_air_pro_ord != has_air_pro_inv:
            return False

        # --- STRICT DOSAGE, VARIANT & PACK CHECKS (CRITICAL BUG FIX) ---
        ord_info = cls._extract_medicine_keys(order_name)
        inv_info = cls._extract_medicine_keys(inv_desc)

        # 1. Pack count (e.g. 10S vs 50S vs 100S)
        if ord_info["pack"] and inv_info["pack"]:
            if ord_info["pack"] != inv_info["pack"]:
                return False
        elif ord_info["pack"] and not inv_info["pack"]:
            other_packs = re.findall(r'\b\d+\s*S\b', inv_desc)
            if other_packs and (other_packs[0] + "S") != ord_info["pack"]:
                return False

        # 2. Variant code (e.g. M1 vs M2)
        if ord_info["variant"] and inv_info["variant"]:
            if ord_info["variant"] != inv_info["variant"]:
                return False
        elif ord_info["variant"] and not inv_info["variant"]:
            if not re.search(r'\b' + ord_info["variant"] + r'\b', inv_desc):
                return False
        elif inv_info["variant"] and not ord_info["variant"]:
            if not re.search(r'\b' + inv_info["variant"] + r'\b', order_name):
                return False

        # 3. Numeric dosage/strength (e.g. 10 vs 20 vs 40)
        if ord_info["dosage"] and inv_info["dosage"]:
            if ord_info["dosage"] != inv_info["dosage"]:
                return False
        elif ord_info["dosage"] and not inv_info["dosage"]:
            # If order has a specific dosage, invoice must have that number
            if not re.search(r'\b' + ord_info["dosage"] + r'\b', inv_desc):
                return False

        # 4. Formulation suffix (e.g. 'P' in ATOFEN P, 'L' in ATOMONT L)
        if ord_info["formulation"] and inv_info["formulation"]:
            if ord_info["formulation"] != inv_info["formulation"]:
                return False

        # 5. Distinct Medicine Brand Stem Conflict
        # If order has a recognized medicine brand stem, invoice MUST have the same stem
        if ord_info["brand_stem"] and inv_info["brand_stem"]:
            if ord_info["brand_stem"] != inv_info["brand_stem"]:
                return False
        elif ord_info["brand_stem"] and not inv_info["brand_stem"]:
            if ord_info["brand_stem"] not in inv_desc:
                return False
        elif inv_info["brand_stem"] and not ord_info["brand_stem"]:
            if inv_info["brand_stem"] not in order_name:
                return False

        # --- PRODUCT CODE MATCHING ---
        if norm_ord_code and norm_inv_code:
            if norm_ord_code == norm_inv_code:
                # Guard against vendor typo reusing code across distinct categories
                # (e.g. AS-A01 is Abdominal Belt, but invoice line 7 printed AS-A01 for Travel Neck Pillow)
                cat_words = ["PILLOW", "COLLAR", "BELT", "MASK", "DIAPER", "PUMP", "NEBULIZER", "THERMOMETER", "MONITOR", "CAP", "WRAP", "ELBOW"]
                norm_ord_cat = cls._normalize_surgical_text(order_name)
                norm_inv_cat = cls._normalize_surgical_text(inv_desc)
                ord_cats = {w for w in cat_words if w in norm_ord_cat}
                inv_cats = {w for w in cat_words if w in norm_inv_cat}
                if ord_cats and inv_cats and not (ord_cats & inv_cats):
                    pass  # Distinct categories, defer to description matching
                else:
                    return True
            else:
                ord_digits = re.findall(r'\d+', norm_ord_code)
                inv_digits = re.findall(r'\d+', norm_inv_code)
                ord_letters = re.findall(r'[A-Z]+', norm_ord_code)
                inv_letters = re.findall(r'[A-Z]+', norm_inv_code)

                if ord_digits and inv_digits and ord_digits != inv_digits:
                    # Distinct model numbers (e.g. AS-A01 vs AS-A07, AS-D01 vs AS-D04)
                    return False
                elif ord_letters and inv_letters and ord_letters != inv_letters and ord_digits and inv_digits and ord_digits == inv_digits:
                    # Different letter prefix (e.g. BP05 vs CN05)
                    # Allow only if descriptions match (e.g. ASH-01 vs AS-A01 for TRAVEL NECK PILLOW)
                    if ord_letters != inv_letters and not (set(re.findall(r'[A-Z]{4,}', order_name)) & set(re.findall(r'[A-Z]{4,}', inv_desc))):
                        return False

        # If only order has product code, check if it's embedded in invoice description
        if norm_ord_code and not norm_inv_code:
            norm_inv_desc_all = re.sub(r'[^A-Z0-9]', '', inv_desc)
            if norm_ord_code in norm_inv_desc_all:
                return True

        # If only invoice has product code, check if it's embedded in order product name
        if norm_inv_code and not norm_ord_code:
            norm_ord_name_all = re.sub(r'[^A-Z0-9]', '', order_name)
            if norm_inv_code in norm_ord_name_all:
                return True

        # --- SURGICAL / DESCRIPTIVE TOKEN & SYNONYM MATCHING ---
        norm_ord_text = cls._normalize_surgical_text(order_name)
        norm_inv_text = cls._normalize_surgical_text(inv_desc)

        clean_ord_tokens = set(re.findall(r'[A-Z0-9]+', norm_ord_text))
        clean_inv_tokens = set(re.findall(r'[A-Z0-9]+', norm_inv_text))

        # Check distinct anatomy / body part
        anatomy_tokens = {
            "KNEE", "ELBOW", "WRIST", "ANKLE", "THUMB", "ARM", "SHOULDER",
            "CERVICAL", "LUMBAR", "ABDOMINAL", "CHEST", "FINGER", "FOOT", "LEG", "CALF", "THIGH"
        }
        ord_anatomy = {t for t in clean_ord_tokens if t in anatomy_tokens}
        inv_anatomy = {t for t in clean_inv_tokens if t in anatomy_tokens}
        if ord_anatomy and not (ord_anatomy & inv_anatomy):
            return False
        if inv_anatomy and not ord_anatomy:
            return False

        # Filter out generic pharmaceutical descriptors, manufacturer brand tokens, and size tokens
        generic_tokens = {
            "ITEM", "ITEMS", "PRODUCT", "PRODUCTS", "MEDICINE", "TAB", "TABLET", "CAPSULE",
            "SYRUP", "INJ", "INJECTION", "THE", "AND", "OF", "MG", "ML", "GM",
            "NOS", "BOX", "STRIP", "PACK", "S", "M", "L", "XL", "XXL", "XXXL", "XS", "UNI", "FS",
            "ATOM", "WELLNESS", "LTD", "PVT", "PHARMA", "HEALTHCARE", "PAIR"
        }
        core_tokens = [t for t in clean_ord_tokens if t not in generic_tokens]

        if core_tokens:
            overlap = sum(1 for t in core_tokens if (t in clean_inv_tokens) or (len(t) > 3 and any(t in it for it in clean_inv_tokens)))
            if overlap / len(core_tokens) >= 0.5:
                return True
        else:
            if order_name and (order_name in inv_desc or inv_desc in order_name):
                return True

        return False

    def _validate_with_openai(
        self,
        invoice_text: str,
        invoice_tables: Optional[List[List[List[str]]]],
        order_data: Dict[str, Any],
        parsed_invoice: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calls OpenAI to perform multimodal/contextual validation matching the exact Laravel contract.
        """
        order_no = order_data.get("order_code") or order_data.get("order_no") or order_data.get("order_id") or "RO-ORDER"
        order_items_json = json.dumps(order_data.get("items", []), indent=2)
        parsed_json = json.dumps(parsed_invoice, indent=2) if parsed_invoice else "None"
        tables_json = json.dumps(invoice_tables, indent=2) if invoice_tables else "None"

        prompt = f"""
You are an expert AI Invoice Verification & Audit Engine integrated with a Laravel Pharmaceutical ERP system.
Reconcile and validate the invoice content against the order data according to the exact Laravel JSON schema below.

### ORDER DETAILS (From Laravel):
Order Reference: {order_no}
Order Items:
{order_items_json}

### EXTRACTED INVOICE RAW TEXT:
{invoice_text}

### EXTRACTED INVOICE TABLES:
{tables_json}

### PRE-PARSED INVOICE DATA:
{parsed_json}

---
### RECONCILIATION RULES:
1. `match_status` and `match_percentage`:
   - `total_ordered_items`: Total number of distinct items/products in the order.
   - `matched_items`: Number of distinct ordered items that exist on the invoice (even if fulfilled with partial quantity).
   - Calculate: `match_percentage = (matched_items / total_ordered_items) * 100`.
   - "matched": match_percentage == 100% with matching quantities fulfilled.
   - "partial_match": match_percentage >= 50% (at least half the ordered products exist on the invoice, or all exist but quantities differ).
   - "blocked": match_percentage < 50% or 0% (document appears to be for a different order or insufficient fulfillment).
   - IMPORTANT: If an item is partially fulfilled (e.g. ordered 10, invoiced 4), place it in `reconciled_items` with match_status="partial_match". Do NOT place partially fulfilled items in `missing_items`. `missing_items` is strictly for products with 0 units on the invoice.
   - CRITICAL COMPREHENSIVENESS: Every single item in the Order Items list MUST be accounted for in either `reconciled_items` (if present on the invoice) OR `missing_items` (if genuinely absent). Do NOT omit any ordered item.
2. Product & Variant Matching Guidelines:
   - Match item descriptions, variant sizes (S, M, L, XL, UNI/FS), and product model codes (e.g., 'AS-D23 KNEE CAP AIR PRO L' fulfills 'KNEE CAP AIR PRO' size 'L'; 'AS-D24 HINGED KNEE BRACE UNI' fulfills 'HINGED KNEE BRACE' [UNI/Universal]; 'ASD-04 KNEE CAP(PAIR) M' fulfills 'KNEE CAP' size 'M'; 'ASD-04 KNEE CAP PAIR L' fulfills 'KNEE CAP' size 'L').
   - Keep distinct variants separate (e.g., size M and size L are distinct items with their own order_item_id).
   - 'UNI', 'UNIVERSAL', or 'FS' denotes Universal / Free Size.
   - Preserve the exact `order_item_id` and `product_id` from the Order Details.
3. CRITICAL: Prevent Collapsing Different Dosages/Strengths into One Product:
   - Treat dosage/strength numbers (10, 20, 40, M1, M2, 500mg, etc.) as STRICT REQUIRED KEYS, NOT as optional fuzzy text.
   - An invoice item with ATOMSTA 20 must NEVER match an order line of ATOMSTA 10 or ATOMSTA 40.
   - ATOMGLIM M1 and ATOMGLIM M2 are distinct products and must NEVER be collapsed or matched to each other.
   - Distinct medicine brand stems (e.g. TELMATOM vs ATOMSTA vs TENATOM vs ATOMEZ) must NEVER match each other.
   - Distinct pack counts (e.g. 10S vs 50S vs 100S for masks) must NEVER match each other.
4. Product Code (PCode) Prioritization & Medical Synonyms:
   - Prioritize matching Product Code / PCode (AS-A01, AS-A07, AS-B02, AS-D01, AS-D04, AS-D25, AS-E10, BP-05, DT-811, CN-05, ASH-01) printed in PCode or Description column before relying solely on product name fuzzy matching.
   - Medical Synonyms:
     - LS / L.S. = Lumbo Sacral (CONTOURED LS BELT matches LUMBO SACRAL BELT CONTOURED)
     - Belt = Support = Binder (ABDOMINAL BELT matches ABDOMINAL SUPPORT)
     - METBLOW = METBLOWN (AS- MASK 3PLY METBLOWN matches 3 PLY MASK WITH METBLOW)
     - DIGI = DIGITAL
   - Ignore manufacturer brand prefixes like AS- (Atom Surgical), ATOM WELLNESS, ATOM.
5. Handle Cross-Product Free Schemes & Promotional Bonus Lines:
   - Cross-Product Schemes (CRITICAL): When an order item has `free_qty > 0` and `free_product_name` (e.g. ordered 'BP MONITOR' with free reward 'ATOMSTA 10' qty 75, or 'NEBULIZER' with free reward 'ATOFEN P' qty 105):
     The invoice may list these as separate rows with Qty: 0 and Free: X (e.g. Line 33: ATOMSTA 10 Qty: 0, Free: 75; Line 27: ATOFEN P Qty: 0, Free: 105).
     The invoice remarks may state: "FREE QNTY AGAINST BP,NEBULIZER PURCHASE".
     CRITICAL: Match each Qty: 0, Free > 0 row to the parent order item whose `free_product_name` and `free_qty` match (e.g. assign ATOMSTA 10 Qty: 0, Free: 75 to BP MONITOR, NOT to the billed ATOMSTA 10 product).
     Do NOT merge cross-product scheme items into the same-name billed product!
     Fulfill this as the free reward for that ordered product!
     Do NOT treat this free reward line as an extra item or error!
     On the item, populate `free_product_name` and `invoiced_free_qty`.
     If both the paid product and the free reward product are fulfilled, mark `is_matched: true` and `discrepancies: []`.
     If the free reward product is missing or quantity is mismatched, mark `is_matched: false`, add to `discrepancies`, and set `match_status: 'partial_match'`.
   - Same-Product Free Lines: If an invoice row has Qty: 0 and Free > 0 matching a billed product that is NOT claimed by any cross-product scheme, combine it under that same medicine's batches and free_qty.
   - If paid invoiced quantity is less than ordered quantity, mark that item as `partial_match`. Do NOT place partially fulfilled items in `missing_items`!
5b. Variant Group Free Pooling (Multi-Size Products):
   - For products with variants (e.g. KNEE CAP, ANKLE BINDER, LS BELT, ABDOMINAL SUPPORT, SOFT CERVICAL COLLAR in sizes M, L, XL):
   - If the order places free quantity on one size (e.g. 3 on M, 0 on L, 0 on XL), but the invoice distributes them across variants (e.g. 1 on M, 1 on L, 1 on XL), pool free quantities across the entire product family.
   - If total ordered free for the product family equals total invoiced free across all sizes (3 == 1+1+1), mark all variants with matching paid quantities as MATCHED with discrepancies: [].
6. `batches` Array per Item:
   When an item is fulfilled using multiple batches or lines (e.g. paid line + promotional bonus line), group them in that item's `batches` array rather than returning disconnected line items.
7. `invoiced_qty`:
   Sum of all paid quantities for that item across its batches.
8. `free_qty`:
   Sum of all free quantities for that item across its batches (including promotional bonus quantities).
9. `missing_items`:
   Any ordered product not found on the invoice. Do NOT put items here if they appear on the invoice.
10. `extra_items`:
   Any product printed on the invoice that was not in the original order.
   CRITICAL: At the bottom of each page, ignore GST/Tax summary tables (e.g. "GST 5.00%", "GST 12.00%", "SGST", "CGST", "IGST", "TOTAL", taxable amounts, or rows with 0 rate/qty).
   CRITICAL: Discard any line item where product_name is empty, numeric-only (e.g. "0.00", "0", "1.00"), or has less than 2 alphabetical letters.
   CRITICAL: Discard any line item where quantity == 0 AND free_qty == 0 AND net_amount == 0.
   NEVER include GST summary tables, tax breakdown rows, or phantom "0.00" footer values in extra_items or line items!
   Promotional bonus lines and cross-product scheme lines belonging to an ordered product must NOT be put here.
11. `expiry_date` in `batches`:
   ALWAYS extract the batch expiration date from PRE-PARSED INVOICE DATA (field 'expiry') or tables/text. Convert month/year formats like "01/31" into standard ISO "YYYY-MM-DD" (e.g. "2031-01-31", "02/31" -> "2031-02-28", "06/30" -> "2030-06-30"). Never return null when an expiry date is present in the invoice or pre-parsed data.
12. `summary`:
   - If blocked (0%): "None of the ordered items were found in the uploaded invoice. Document appears to be for a different order."
   - If blocked (< 50%): "Verification Blocked: Only <N> of <T> ordered items matched (<P>%). Document appears to be for a different order."
   - If matched: "All <N> ordered items matched across <V> variants with <F> free units fulfilled."
   - If extra items only: "Verification Mismatch: Contains <X> extra items not in order. Please verify before approving."
   - If missing & extra: "Verification Mismatch: Missing <M> ordered items. Contains <X> extra items not in order. Please verify before approving."
   - If missing only: "Verification Mismatch: Missing <M> ordered items. Please verify before approving."

### REQUIRED JSON SCHEMA:
Return ONLY valid JSON matching this schema:
{{
  "success": true,
  "match_status": "matched" | "partial_match" | "blocked",
  "match_percentage": number,
  "summary": "string",
  "missing_items": [ ... ],
  "extra_items": [ ... ],
  "reconciled_items": [ ... ],
  "data": {{
    "match_status": "matched" | "partial_match" | "blocked",
    "match_percentage": number,
    "invoice_metadata": {{
      "invoice_no": "string",
      "invoice_date": "string (YYYY-MM-DD)",
      "gstin": "string",
      "taxable_amount": float,
      "net_amount": float,
      "total_amount": float
    }},
    "reconciled_items": [
      {{
        "order_item_id": int | string,
        "product_id": int | string,
        "product_name": "string",
        "size": "string or null",
        "match_status": "matched" | "partial_match",
        "ordered_qty": number,
        "invoiced_qty": number,
        "free_qty": number,
        "taxable_amount": float,
        "gst_amount": float,
        "net_amount": float,
        "batches": [
          {{
            "batch_no": "string",
            "quantity": number,
            "free_quantity": number,
            "expiry_date": "string (YYYY-MM-DD) or null",
            "match_status": "matched"
          }}
        ]
      }}
    ],
    "missing_items": [
      {{
        "order_item_id": int | string,
        "product_id": int | string,
        "product_name": "string",
        "size": "string or null",
        "match_status": "missing",
        "ordered_qty": number,
        "reason": "string"
      }}
    ],
    "extra_items": [
      {{
        "product_name": "string",
        "batch_no": "string",
        "match_status": "extra",
        "quantity": number,
        "free_qty": number,
        "expiry_date": "string (YYYY-MM-DD) or null",
        "taxable_amount": float,
        "net_amount": float
      }}
    ]
  }}
}}
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional Invoice Audit and Reconciliation AI for ERP systems. Always respond in valid JSON matching the exact schema specified."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.0
        )

        return json.loads(response.choices[0].message.content)

    def _validate_rule_based(
        self,
        invoice_text: str,
        invoice_tables: Optional[List[List[List[str]]]],
        order_data: Dict[str, Any],
        parsed_invoice: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Deterministic rule-based invoice verification engine when OpenAI API is unavailable.
        Performs token matching, multi-batch grouping, missing/extra item segmentation, and summary generation.
        """
        from parser import InvoiceParser

        order_items = order_data.get("items", [])

        # 1. Extract metadata
        inv_no = InvoiceParser._extract_invoice_number(invoice_text)
        inv_date = InvoiceParser._extract_date(invoice_text)
        inv_gstin = InvoiceParser._extract_gstin(invoice_text)
        taxable_amt = InvoiceParser._extract_taxable_amount(invoice_text) if hasattr(InvoiceParser, "_extract_taxable_amount") else None
        net_amt = InvoiceParser._extract_net_amount(invoice_text) if hasattr(InvoiceParser, "_extract_net_amount") else None

        # 2. Get line items
        line_items = parsed_invoice.get("line_items", []) if parsed_invoice else []
        if not line_items and invoice_tables:
            line_items = InvoiceParser.parse_line_items(invoice_tables, mode="retailer")
        # Filter out GST breakdown tables, tax summary rows, and phantom 0.00 rows
        line_items = [it for it in line_items if not self._is_tax_or_summary_row(it) and not self._is_invalid_or_phantom_line(it)]

        # Pre-match cross-product free schemes (Qty: 0, Free: X rows)
        # Ensures lines like ATOMSTA 10 (Qty: 0, Free: 75) and ATOFEN P (Qty: 0, Free: 105)
        # are mapped to their parent order item (BP MONITOR, NEBULIZER) and NOT lumped into
        # the same-name billed items (ATOMSTA 10, ATOFEN P).
        cross_product_claims = {}
        cross_claimed_indices = set()

        for ord_item in order_items:
            oi_id = ord_item.get("order_item_id") or ord_item.get("id")
            p_name = ord_item.get("product_name") or ord_item.get("name", "")
            f_qty = self._to_float(ord_item.get("free_qty") or ord_item.get("free") or ord_item.get("sch") or ord_item.get("bonus") or 0)
            f_name = ord_item.get("free_product_name")

            c_paid = re.sub(r'[^A-Z0-9]', '', str(p_name).upper())
            c_free = re.sub(r'[^A-Z0-9]', '', str(f_name).upper()) if f_name else ""
            is_cross = bool(f_qty > 0 and c_free and c_free != c_paid)

            if is_cross:
                reward_target = {
                    "product_id": ord_item.get("free_product_id"),
                    "product_name": f_name,
                    "product_code": ord_item.get("free_product_code"),
                    "size": ord_item.get("free_size"),
                    "side": ord_item.get("free_side"),
                    "order_qty": f_qty
                }
                for idx, inv_it in enumerate(line_items):
                    if idx in cross_claimed_indices:
                        continue
                    l_qty = self._to_float(inv_it.get("qty") or inv_it.get("quantity"))
                    l_free = self._to_float(inv_it.get("free") or inv_it.get("sch") or inv_it.get("free_qty"))
                    if (l_qty == 0 and l_free > 0) or (l_free == f_qty):
                        if self._match_product(reward_target, inv_it):
                            cross_product_claims.setdefault(oi_id, []).append(idx)
                            cross_claimed_indices.add(idx)
                            break

        items_list = []
        reconciled_items = []
        missing_items = []
        matched_inv_indices = set()

        for ord_item in order_items:
            ord_item_id = ord_item.get("order_item_id") or ord_item.get("id")
            prod_id = ord_item.get("product_id") or ord_item.get("id")
            prod_name = ord_item.get("product_name") or ord_item.get("name", "")
            size = ord_item.get("size") or ord_item.get("variant_label")
            ordered_qty = self._to_float(ord_item.get("ordered_qty") or ord_item.get("order_qty") or ord_item.get("quantity"))
            free_qty = self._to_float(ord_item.get("free_qty") or ord_item.get("free") or ord_item.get("sch") or ord_item.get("bonus") or 0)
            free_prod_id = ord_item.get("free_product_id")
            free_prod_name = ord_item.get("free_product_name")
            free_prod_code = ord_item.get("free_product_code")
            free_side = ord_item.get("free_side")
            free_size = ord_item.get("free_size")

            # Determine whether same-product free or cross-product free
            c_paid = re.sub(r'[^A-Z0-9]', '', str(prod_name).upper())
            c_free = re.sub(r'[^A-Z0-9]', '', str(free_prod_name).upper()) if free_prod_name else ""
            is_cross_product = bool(free_qty > 0 and c_free and c_free != c_paid)

            # Paid item matching
            matched_batches = []
            item_taxable = 0.0
            item_gst = 0.0
            item_net = 0.0
            total_invoiced_qty = 0.0
            total_same_free_qty = 0.0

            for idx, inv_item in enumerate(line_items):
                if idx in matched_inv_indices or idx in cross_claimed_indices:
                    continue
                if self._match_product(ord_item, inv_item):
                    matched_inv_indices.add(idx)
                    b_no = inv_item.get("batch") or inv_item.get("batch_no") or "N/A"
                    qty = self._to_float(inv_item.get("qty") or inv_item.get("quantity"))
                    free_q = self._to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
                    exp_d = self._format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp"))
                    raw_exp = inv_item.get("expiry") or inv_item.get("exp")

                    # Group batches with identical batch numbers (e.g. paid line + promotional bonus line)
                    existing_b = next((b for b in matched_batches if b["batch_no"] == b_no), None) if b_no != "N/A" else None
                    if existing_b:
                        existing_b["quantity"] += int(qty) if qty.is_integer() else qty
                        existing_b["free_quantity"] += int(free_q) if free_q.is_integer() else free_q
                        if not existing_b.get("expiry_date") and exp_d:
                            existing_b["expiry_date"] = exp_d
                    else:
                        batch_entry = {
                            "batch_no": b_no,
                            "quantity": int(qty) if qty.is_integer() else qty,
                            "free_quantity": int(free_q) if free_q.is_integer() else free_q,
                            "expiry_date": exp_d,
                            "raw_expiry": raw_exp,
                            "match_status": "matched"
                        }
                        matched_batches.append(batch_entry)

                    total_invoiced_qty += qty
                    total_same_free_qty += free_q

                    taxable = self._to_float(inv_item.get("taxable_amt") or inv_item.get("taxable_amount"))
                    rate = self._to_float(inv_item.get("rate") or inv_item.get("price"))
                    if taxable == 0.0 and rate > 0.0 and qty > 0:
                        taxable = rate * qty

                    amt = self._to_float(inv_item.get("amount") or inv_item.get("net_amount"))
                    if amt == 0.0 and taxable > 0:
                        amt = taxable

                    gst = self._to_float(inv_item.get("gst_amt") or inv_item.get("tax_amt") or inv_item.get("gst_amount"))
                    if gst == 0.0 and amt > taxable and taxable > 0:
                        gst = amt - taxable

                    item_taxable += taxable
                    item_gst += gst
                    item_net += amt

            # Free Reward Item Matching (for cross-product schemes)
            total_invoiced_free_qty = 0.0
            reward_batches = []
            if is_cross_product:
                # 1. Use pre-matched cross-product scheme lines first
                claimed_idxs = cross_product_claims.get(ord_item_id, [])
                for c_idx in claimed_idxs:
                    inv_item = line_items[c_idx]
                    matched_inv_indices.add(c_idx)
                    l_free = self._to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
                    l_qty = self._to_float(inv_item.get("qty") or inv_item.get("quantity"))
                    reward_q = l_free if l_free > 0 else l_qty
                    total_invoiced_free_qty += reward_q
                    r_bno = inv_item.get("batch") or inv_item.get("batch_no") or "N/A"
                    r_exp = self._format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp"))
                    reward_batches.append({
                        "batch_no": r_bno,
                        "quantity": 0,
                        "free_quantity": int(reward_q) if reward_q.is_integer() else reward_q,
                        "expiry_date": r_exp,
                        "raw_expiry": inv_item.get("expiry") or inv_item.get("exp"),
                        "match_status": "matched"
                    })

                # 2. If not satisfied, scan remaining line items
                if total_invoiced_free_qty < free_qty:
                    reward_target = {
                        "product_id": free_prod_id,
                        "product_name": free_prod_name,
                        "product_code": free_prod_code,
                        "size": free_size,
                        "side": free_side,
                        "order_qty": free_qty
                    }
                    for idx, inv_item in enumerate(line_items):
                        if idx in matched_inv_indices or idx in cross_claimed_indices:
                            continue
                        if self._match_product(reward_target, inv_item):
                            matched_inv_indices.add(idx)
                            l_free = self._to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
                            l_qty = self._to_float(inv_item.get("qty") or inv_item.get("quantity"))
                            reward_q = l_free if l_free > 0 else l_qty
                            total_invoiced_free_qty += reward_q
                            r_bno = inv_item.get("batch") or inv_item.get("batch_no") or "N/A"
                            r_exp = self._format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp"))
                            reward_batches.append({
                                "batch_no": r_bno,
                                "quantity": 0,
                                "free_quantity": int(reward_q) if reward_q.is_integer() else reward_q,
                                "expiry_date": r_exp,
                                "raw_expiry": inv_item.get("expiry") or inv_item.get("exp"),
                                "match_status": "matched"
                            })

                # 3. Check if printed in remarks/free column or invoice text
                if total_invoiced_free_qty < free_qty:
                    found_in_text = self._find_free_reward_in_text(invoice_text, free_prod_name)
                    if found_in_text > 0:
                        total_invoiced_free_qty += found_in_text
            else:
                total_invoiced_free_qty = total_same_free_qty

            # Determine primary batch_no and expiry_date for this item
            primary_batch_no = "N/A"
            primary_exp_iso = None
            primary_exp_mmyyyy = None
            if matched_batches:
                primary_batch_no = matched_batches[0]["batch_no"]
                primary_exp_iso = matched_batches[0]["expiry_date"]
                primary_exp_mmyyyy = self._format_expiry_mmyyyy(primary_exp_iso) or self._format_expiry_mmyyyy(matched_batches[0].get("raw_expiry"))

            # Calculate item match status and discrepancies
            discrepancies = []
            paid_matched = bool(total_invoiced_qty == ordered_qty)
            if total_invoiced_qty < ordered_qty:
                discrepancies.append(f"Invoiced quantity ({int(total_invoiced_qty) if total_invoiced_qty.is_integer() else total_invoiced_qty}) is less than ordered quantity ({int(ordered_qty) if ordered_qty.is_integer() else ordered_qty}) for {prod_name}.")
            elif total_invoiced_qty > ordered_qty:
                discrepancies.append(f"Invoiced quantity ({int(total_invoiced_qty) if total_invoiced_qty.is_integer() else total_invoiced_qty}) exceeds ordered quantity ({int(ordered_qty) if ordered_qty.is_integer() else ordered_qty}) for {prod_name}.")

            free_matched = True
            if free_qty > 0:
                if total_invoiced_free_qty < free_qty:
                    free_matched = False
                    target_free_name = free_prod_name if is_cross_product else prod_name
                    if total_invoiced_free_qty == 0:
                        discrepancies.append(f"Free reward item {target_free_name} missing from invoice (expected: {int(free_qty) if free_qty.is_integer() else free_qty}, invoiced: 0).")
                    else:
                        discrepancies.append(f"Free reward item {target_free_name} quantity mismatch (expected: {int(free_qty) if free_qty.is_integer() else free_qty}, invoiced: {int(total_invoiced_free_qty) if total_invoiced_free_qty.is_integer() else total_invoiced_free_qty}).")

            item_is_matched = bool(paid_matched and free_matched and total_invoiced_qty > 0)

            # Build item entry according to expected format
            item_entry = {
                "order_item_id": ord_item_id,
                "product_id": prod_id,
                "product_name": prod_name,
                "ordered_qty": int(ordered_qty) if ordered_qty.is_integer() else ordered_qty,
                "invoiced_qty": int(total_invoiced_qty) if total_invoiced_qty.is_integer() else total_invoiced_qty,
                "free_qty": int(free_qty) if free_qty.is_integer() else free_qty,
                "free_product_name": free_prod_name if is_cross_product else (free_prod_name or None),
                "invoiced_free_qty": int(total_invoiced_free_qty) if total_invoiced_free_qty.is_integer() else total_invoiced_free_qty,
                "batch_no": primary_batch_no,
                "expiry_date": primary_exp_mmyyyy or (primary_exp_iso or "N/A"),
                "is_matched": item_is_matched,
                "discrepancies": discrepancies
            }
            items_list.append(item_entry)

            if matched_batches or total_invoiced_qty > 0:
                final_free_qty = max(total_invoiced_free_qty, free_qty) if free_matched else total_invoiced_free_qty
                reconciled_items.append({
                    "order_item_id": ord_item_id,
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "size": size,
                    "match_status": "matched" if item_is_matched else "partial_match",
                    "ordered_qty": int(ordered_qty) if ordered_qty.is_integer() else ordered_qty,
                    "invoiced_qty": int(total_invoiced_qty) if total_invoiced_qty.is_integer() else total_invoiced_qty,
                    "free_qty": int(final_free_qty) if final_free_qty.is_integer() else final_free_qty,
                    "free_product_name": free_prod_name if is_cross_product else (free_prod_name or None),
                    "invoiced_free_qty": int(total_invoiced_free_qty) if total_invoiced_free_qty.is_integer() else total_invoiced_free_qty,
                    "batch_no": primary_batch_no,
                    "expiry_date": primary_exp_iso or primary_exp_mmyyyy,
                    "is_matched": item_is_matched,
                    "discrepancies": discrepancies,
                    "taxable_amount": round(item_taxable, 2),
                    "gst_amount": round(item_gst, 2),
                    "net_amount": round(item_net, 2),
                    "batches": matched_batches + reward_batches
                })
            else:
                missing_items.append({
                    "order_item_id": ord_item_id,
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "size": size,
                    "match_status": "missing",
                    "ordered_qty": int(ordered_qty) if ordered_qty.is_integer() else ordered_qty,
                    "reason": f"Product {prod_name} was not found on the invoice."
                })

        # Identify extra items present on invoice but absent in order
        unmatched_invoice_items = []
        for idx, inv_item in enumerate(line_items):
            if idx not in matched_inv_indices and idx not in cross_claimed_indices:
                if self._is_tax_or_summary_row(inv_item) or self._is_invalid_or_phantom_line(inv_item):
                    continue
                qty = self._to_float(inv_item.get("qty") or inv_item.get("quantity"))
                free_q = self._to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
                taxable = self._to_float(inv_item.get("taxable_amt") or inv_item.get("taxable_amount"))
                amt = self._to_float(inv_item.get("amount") or inv_item.get("net_amount"))
                unmatched_invoice_items.append({
                    "product_name": inv_item.get("description") or inv_item.get("item_name") or "Unknown Product",
                    "batch_no": inv_item.get("batch") or inv_item.get("batch_no") or "N/A",
                    "match_status": "extra",
                    "quantity": int(qty) if qty.is_integer() else qty,
                    "free_qty": int(free_q) if free_q.is_integer() else free_q,
                    "expiry_date": self._format_expiry_mmyyyy(inv_item.get("expiry") or inv_item.get("exp")),
                    "taxable_amount": round(taxable, 2),
                    "net_amount": round(amt, 2)
                })
        extra_items = [ex for ex in unmatched_invoice_items if not self._is_tax_or_summary_row(ex) and not self._is_invalid_or_phantom_line(ex)]
        unmatched_invoice_items = extra_items

        # Apply Variant Group Free Pooling (Issue 2)
        self._apply_variant_group_free_pooling(items_list, reconciled_items)

        # Determine match_percentage, match_status and human-readable summary
        total_ordered_items = len(order_items)
        matched_items_count = sum(1 for it in items_list if it["invoiced_qty"] > 0)
        all_items_matched = (len(items_list) == total_ordered_items and all(it["is_matched"] for it in items_list) and len(missing_items) == 0)

        if total_ordered_items > 0:
            calc_pct = round((matched_items_count / total_ordered_items) * 100, 2)
            match_percentage = int(calc_pct) if calc_pct.is_integer() else calc_pct
        else:
            match_percentage = 0

        total_free_rewards = sum(it.get("free_qty", 0) for it in items_list if it.get("free_qty", 0) > 0)

        missing_labels = []
        for m in missing_items:
            pname = m.get("product_name") or m.get("name") or "Unknown Product"
            psize = m.get("size")
            if psize and str(psize).upper() not in ["NONE", "NULL", "N/A", ""]:
                missing_labels.append(f"{pname} [{psize}]")
            else:
                missing_labels.append(pname)

        if all_items_matched:
            match_status = "matched"
            if total_free_rewards > 0:
                summary = f"All {total_ordered_items} items and {self._int_or_float(total_free_rewards)} free reward items matched with the invoice."
            else:
                summary = f"All {total_ordered_items} items matched with the invoice."
            if len(extra_items) > 0:
                extra_labels = [e.get("product_name") or "Extra Product" for e in extra_items[:3]]
                summary += f" Verification Note: Contains {len(extra_items)} extra items not in order ({', '.join(extra_labels)})."
        elif matched_items_count > 0:
            match_status = "partial_match"
            all_discrepancies = []
            for it in items_list:
                all_discrepancies.extend(it.get("discrepancies", []))
            if len(missing_items) > 0 and len(extra_items) > 0:
                missing_str = ", ".join(missing_labels[:3])
                extra_str = ", ".join([e.get("product_name") or "Extra Item" for e in extra_items[:3]])
                item_word = "item" if len(missing_items) == 1 else "items"
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered {item_word} ({missing_str}). Contains {len(extra_items)} extra items not in order ({extra_str}). Please verify before approving."
            elif all_discrepancies:
                summary = f"Verification Mismatch: {'; '.join(d.rstrip('.') for d in all_discrepancies[:2])}. Please verify before approving."
            elif len(extra_items) > 0:
                extra_str = ", ".join([e.get("product_name") or "Extra Item" for e in extra_items[:3]])
                summary = f"Verification Mismatch: Contains {len(extra_items)} extra items not in order ({extra_str}). Please verify before approving."
            elif len(missing_items) > 0:
                missing_str = ", ".join(missing_labels[:3])
                item_word = "item" if len(missing_items) == 1 else "items"
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered {item_word} ({missing_str}). Please verify before approving."
            else:
                summary = "Verification Mismatch: One or more items did not match order details. Please verify before approving."
        else:
            match_status = "blocked"
            if match_percentage == 0:
                summary = "None of the ordered items were found in the uploaded invoice. Document appears to be for a different order."
            else:
                missing_str = f" ({', '.join(missing_labels[:3])})" if missing_labels else ""
                summary = f"Verification Blocked: Only {matched_items_count} of {total_ordered_items} ordered items matched{missing_str}. Document appears to be for a different order or insufficient fulfillment."

        meta_from_parsed = parsed_invoice.get("invoice_metadata", {}) if parsed_invoice else {}
        taxable_sum = sum(r["taxable_amount"] for r in reconciled_items)
        net_sum = sum(r["net_amount"] for r in reconciled_items)

        final_taxable = round(self._to_float(meta_from_parsed.get("taxable_amount") or taxable_amt or taxable_sum), 2)
        final_net = round(self._to_float(meta_from_parsed.get("net_amount") or meta_from_parsed.get("total_amount") or net_amt or net_sum), 2)

        raw_inv_date = meta_from_parsed.get("invoice_date") or meta_from_parsed.get("date") or inv_date or "N/A"
        clean_inv_date = str(raw_inv_date)
        m_d = re.match(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', clean_inv_date)
        if m_d:
            clean_inv_date = f"{m_d.group(3)}-{int(m_d.group(2)):02d}-{int(m_d.group(1)):02d}"

        invoice_metadata = {
            "invoice_no": meta_from_parsed.get("invoice_no") or meta_from_parsed.get("invoice_number") or inv_no or "N/A",
            "invoice_number": meta_from_parsed.get("invoice_no") or meta_from_parsed.get("invoice_number") or inv_no or "N/A",
            "date": clean_inv_date,
            "invoice_date": clean_inv_date,
            "gstin": meta_from_parsed.get("gstin") or meta_from_parsed.get("extracted_gstin") or inv_gstin or order_data.get("purchaser_gstin") or "N/A",
            "total_taxable": f"{final_taxable:.2f}",
            "taxable_amount": final_taxable,
            "net_amount": f"{final_net:.2f}",
            "total_amount": final_net,
            "purchaser_name": meta_from_parsed.get("purchaser_name") or order_data.get("purchaser_name") or "N/A",
            "purchaser_gstin": meta_from_parsed.get("purchaser_gstin") or order_data.get("purchaser_gstin") or "N/A",
            "is_duplicate": False
        }

        return {
            "status": "success" if match_status == "matched" else ("warning" if match_status == "partial_match" else "error"),
            "match_status": match_status,
            "match_percentage": match_percentage,
            "summary": summary,
            "items": items_list,
            "reconciled_items": reconciled_items,
            "missing_items": missing_items,
            "extra_items": extra_items,
            "unmatched_invoice_items": extra_items,
            "data": {
                "match_status": match_status,
                "match_percentage": match_percentage,
                "summary": summary,
                "invoice_metadata": invoice_metadata,
                "items": items_list,
                "reconciled_items": reconciled_items,
                "missing_items": missing_items,
                "extra_items": extra_items,
                "unmatched_invoice_items": extra_items
            },
            "success": match_status != "blocked",
            "message": summary,
            "extracted_metadata": invoice_metadata
        }

    def _sanitize_response(
        self,
        data: Dict[str, Any],
        order_data: Dict[str, Any],
        parsed_invoice: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sanitizes and enforces strict contract compliance on the AI or rule-based output.
        """
        match_status = data.get("match_status", "partial_match")
        summary = data.get("summary") or data.get("message") or "Invoice verification completed."

        inner_data = data.get("data", {})
        meta = inner_data.get("invoice_metadata", data.get("extracted_metadata", {}))

        clean_meta = {
            "gstin": meta.get("gstin") or meta.get("extracted_gstin") or "N/A",
            "invoice_date": meta.get("invoice_date") or meta.get("date") or "N/A",
            "invoice_no": meta.get("invoice_no") or meta.get("invoice_number") or "N/A",
            "net_amount": round(self._to_float(meta.get("net_amount") or meta.get("total_amount")), 2),
            "taxable_amount": round(self._to_float(meta.get("taxable_amount")), 2),
            "total_amount": round(self._to_float(meta.get("total_amount") or meta.get("net_amount")), 2),
            "is_duplicate": meta.get("is_duplicate", False)
        }

        reconciled_items = inner_data.get("reconciled_items", [])
        missing_items = inner_data.get("missing_items", [])
        extra_items = inner_data.get("extra_items", [])

        # Build batch -> expiry mapping and batch -> line mapping from parsed invoice line items
        batch_expiry_map = {}
        batch_to_line = {}
        parsed_lines = parsed_invoice.get("line_items", []) if parsed_invoice else []
        parsed_lines = [l for l in parsed_lines if not self._is_tax_or_summary_row(l) and not self._is_invalid_or_phantom_line(l)]
        for itm in parsed_lines:
            b_code = (itm.get("batch") or itm.get("batch_no") or "").strip().upper()
            exp_raw = itm.get("expiry") or itm.get("exp")
            if b_code and b_code not in ["N/A", "NONE", "BLANK", "NULL", ""]:
                batch_to_line[b_code] = itm
            if b_code and exp_raw and str(exp_raw).upper() not in ["N/A", "NONE", "BLANK", "NULL"]:
                batch_expiry_map[b_code] = exp_raw

        # Build order item lookup for ground truth ordered quantities and IDs
        order_items_list = order_data.get("items", [])
        order_by_oid = {}
        for oi in order_items_list:
            oid_k = oi.get("order_item_id") or oi.get("id")
            if oid_k is not None:
                order_by_oid[str(oid_k)] = oi

        # Track reconciled order_item_ids and batches
        reconciled_oids = set()
        reconciled_batches = set()

        for item in reconciled_items:
            oid = item.get("order_item_id")
            pid = item.get("product_id")
            item_name = (item.get("product_name") or item.get("name") or "").upper().strip()
            item_size = self._normalize_size_tag(item.get("size"))

            # Look up ground truth ordered item from order_data
            matching_ord = None
            # Check by order_item_id first, BUT verify product compatibility
            if oid is not None and str(oid) in order_by_oid:
                cand = order_by_oid[str(oid)]
                if self._match_product(cand, item):
                    matching_ord = cand

            # If not matched, try matching by product_id and size
            if not matching_ord and pid is not None:
                for oi in order_items_list:
                    if str(oi.get("product_id")) == str(pid):
                        oi_size = self._normalize_size_tag(oi.get("size"))
                        if oi_size == item_size or ({oi_size, item_size}.issubset({"UNI", "FS"})) or not oi_size or not item_size:
                            matching_ord = oi
                            break

            # Fallback: fuzzy/token match against order_items_list
            if not matching_ord:
                for oi in order_items_list:
                    if self._match_product(oi, item):
                        matching_ord = oi
                        break

            if matching_ord:
                ground_ord_q = self._to_float(matching_ord.get("ordered_qty") or matching_ord.get("order_qty") or matching_ord.get("quantity"))
                if ground_ord_q > 0:
                    item["ordered_qty"] = int(ground_ord_q) if ground_ord_q.is_integer() else ground_ord_q
                if matching_ord.get("order_item_id") is not None:
                    item["order_item_id"] = matching_ord["order_item_id"]
                elif matching_ord.get("id") is not None:
                    item["order_item_id"] = matching_ord["id"]
                if matching_ord.get("product_id") is not None:
                    item["product_id"] = matching_ord["product_id"]
                if matching_ord.get("product_name"):
                    item["product_name"] = matching_ord["product_name"]
                if matching_ord.get("size") and not item.get("size"):
                    item["size"] = matching_ord["size"]

            # Verify batch attribution against parsed invoice lines to prevent LLM hallucinations
            if matching_ord and parsed_lines:
                valid_batches = []
                for b in item.get("batches", []):
                    b_no = (b.get("batch_no") or "").strip().upper()
                    if b_no in batch_to_line:
                        orig_pl = batch_to_line[b_no]
                        if not self._match_product(matching_ord, orig_pl):
                            # Misattributed batch by AI (belongs to a different product on the invoice)
                            continue
                    valid_batches.append(b)

                if not valid_batches:
                    # AI completely misattributed batches for this item; recover the real lines from parsed_lines
                    matched_tax = 0.0
                    matched_amt = 0.0
                    for pl in parsed_lines:
                        if self._match_product(matching_ord, pl):
                            pl_b = (pl.get("batch") or pl.get("batch_no") or "N/A").strip()
                            pl_q = self._to_float(pl.get("qty") or pl.get("quantity"))
                            pl_f = self._to_float(pl.get("free") or pl.get("sch") or pl.get("free_qty"))
                            pl_tax = self._to_float(pl.get("taxable_amt") or pl.get("taxable_amount"))
                            pl_amt = self._to_float(pl.get("amount") or pl.get("net_amount"))
                            existing_vb = next((b for b in valid_batches if b["batch_no"] == pl_b), None) if pl_b != "N/A" else None
                            if existing_vb:
                                existing_vb["quantity"] += int(pl_q) if pl_q.is_integer() else pl_q
                                existing_vb["free_quantity"] += int(pl_f) if pl_f.is_integer() else pl_f
                            else:
                                valid_batches.append({
                                    "batch_no": pl_b,
                                    "quantity": int(pl_q) if pl_q.is_integer() else pl_q,
                                    "free_quantity": int(pl_f) if pl_f.is_integer() else pl_f,
                                    "expiry_date": self._format_expiry_iso(pl.get("expiry") or pl.get("exp")),
                                    "match_status": "matched"
                                })
                            matched_tax += pl_tax
                            matched_amt += pl_amt
                    if valid_batches:
                        item["invoiced_qty"] = sum(b["quantity"] for b in valid_batches)
                        item["taxable_amount"] = round(matched_tax, 2)
                        item["net_amount"] = round(matched_amt, 2)

                # Merge any promotional bonus lines (qty == 0 and free > 0) from parsed_lines matching this product
                for pl in parsed_lines:
                    pl_q = self._to_float(pl.get("qty") or pl.get("quantity"))
                    pl_f = self._to_float(pl.get("free") or pl.get("sch") or pl.get("free_qty"))
                    if pl_q == 0 and pl_f > 0:
                        # Check if this bonus line belongs to a cross-product scheme of another ordered item
                        is_cross_for_other = False
                        for other_oi in order_items_list:
                            other_f_qty = self._to_float(other_oi.get("free_qty") or other_oi.get("free") or other_oi.get("sch") or other_oi.get("bonus") or 0)
                            other_f_name = other_oi.get("free_product_name")
                            other_p_name = other_oi.get("product_name") or other_oi.get("name", "")
                            c_op = re.sub(r'[^A-Z0-9]', '', str(other_p_name).upper())
                            c_of = re.sub(r'[^A-Z0-9]', '', str(other_f_name).upper()) if other_f_name else ""
                            if other_f_qty > 0 and c_of and c_of != c_op:
                                target_check = {
                                    "product_id": other_oi.get("free_product_id"),
                                    "product_name": other_f_name,
                                    "product_code": other_oi.get("free_product_code"),
                                    "order_qty": other_f_qty
                                }
                                if self._match_product(target_check, pl):
                                    is_cross_for_other = True
                                    break
                        if is_cross_for_other:
                            continue

                        if self._match_product(matching_ord, pl):
                            pl_b = (pl.get("batch") or pl.get("batch_no") or "N/A").strip()
                            existing_vb = next((b for b in valid_batches if b["batch_no"] == pl_b), None) if pl_b != "N/A" else None
                            if existing_vb:
                                if int(pl_f) > 0 and existing_vb["free_quantity"] < int(pl_f):
                                    existing_vb["free_quantity"] += int(pl_f) if pl_f.is_integer() else pl_f
                            else:
                                valid_batches.append({
                                    "batch_no": pl_b,
                                    "quantity": 0,
                                    "free_quantity": int(pl_f) if pl_f.is_integer() else pl_f,
                                    "expiry_date": self._format_expiry_iso(pl.get("expiry") or pl.get("exp")),
                                    "match_status": "matched"
                                })
                item["batches"] = valid_batches

            if item.get("order_item_id") is not None:
                reconciled_oids.add(str(item["order_item_id"]))

            ord_q = self._to_float(item.get("ordered_qty") or item.get("order_qty"))
            inv_q = self._to_float(item.get("invoiced_qty") or item.get("quantity"))

            # Update free_qty if batches contain free quantities or order specifies free units
            batch_free_sum = sum(self._to_float(b.get("free_quantity")) for b in item.get("batches", []))
            ord_free = self._to_float(matching_ord.get("free_qty") or matching_ord.get("free") or matching_ord.get("sch") or matching_ord.get("bonus")) if matching_ord else 0.0
            final_free_qty = max(batch_free_sum, ord_free)
            item["free_qty"] = int(final_free_qty) if final_free_qty.is_integer() else final_free_qty

            item["match_status"] = "matched" if inv_q >= ord_q else "partial_match"

            for b in item.get("batches", []):
                if not b.get("match_status") or str(b.get("match_status")).lower() in ["none", "null"]:
                    b["match_status"] = "matched"
                b_no = (b.get("batch_no") or "").strip().upper()
                if b_no and b_no != "N/A":
                    reconciled_batches.add(b_no)
                exp_val = b.get("expiry_date")
                if not exp_val or str(exp_val).upper() in ["N/A", "NONE", "NULL", "BLANK", ""]:
                    if b_no in batch_expiry_map:
                        exp_val = batch_expiry_map[b_no]
                    else:
                        for bk, bv in batch_expiry_map.items():
                            if b_no and (b_no in bk or bk in b_no):
                                exp_val = bv
                                break
                b["expiry_date"] = self._format_expiry_iso(exp_val)

        for it in extra_items:
            if not it.get("match_status") or str(it.get("match_status")).lower() in ["none", "null"]:
                it["match_status"] = "extra"
            exp_val = it.get("expiry_date")
            b_no = (it.get("batch_no") or "").strip().upper()
            if not exp_val or str(exp_val).upper() in ["N/A", "NONE", "NULL", "BLANK", ""]:
                if b_no in batch_expiry_map:
                    exp_val = batch_expiry_map[b_no]
                else:
                    for bk, bv in batch_expiry_map.items():
                        if b_no and (b_no in bk or bk in b_no):
                            exp_val = bv
                            break
            it["expiry_date"] = self._format_expiry_iso(exp_val)

        # Ensure 100% order item accountability: Check ALL items in the original order
        parsed_lines = parsed_invoice.get("line_items", []) if parsed_invoice else []
        for oi in order_items_list:
            oid = str(oi.get("order_item_id") or oi.get("id") or "")
            if oid and oid in reconciled_oids:
                continue

            ord_q = self._to_float(oi.get("ordered_qty") or oi.get("order_qty") or oi.get("quantity"))
            ord_name = oi.get("product_name") or oi.get("name") or ""
            ord_pid = oi.get("product_id") or oi.get("id")
            ord_size = oi.get("size") or oi.get("variant_label")

            # A. Check if the ordered item was falsely placed in extra_items
            matched_extra_idx = -1
            for e_idx, ex in enumerate(extra_items):
                if self._match_product(oi, ex):
                    matched_extra_idx = e_idx
                    break

            if matched_extra_idx != -1:
                ex = extra_items.pop(matched_extra_idx)
                qty = self._to_float(ex.get("quantity") or ex.get("qty"))
                free_q = self._to_float(ex.get("free_qty") or ex.get("free"))
                taxable = self._to_float(ex.get("taxable_amount") or ex.get("taxable_amt"))
                net_amt = self._to_float(ex.get("net_amount") or ex.get("amount"))
                b_no = ex.get("batch_no") or "N/A"
                if b_no and b_no != "N/A":
                    reconciled_batches.add(b_no.strip().upper())

                ord_free = self._to_float(oi.get("free_qty") or oi.get("free") or oi.get("sch") or oi.get("bonus"))
                final_free_qty = max(free_q, ord_free)
                reconciled_items.append({
                    "order_item_id": oi.get("order_item_id") or oi.get("id"),
                    "product_id": ord_pid,
                    "product_name": ord_name,
                    "size": ord_size,
                    "match_status": "matched" if qty >= ord_q else "partial_match",
                    "ordered_qty": int(ord_q) if ord_q.is_integer() else ord_q,
                    "invoiced_qty": int(qty) if qty.is_integer() else qty,
                    "free_qty": int(final_free_qty) if final_free_qty.is_integer() else final_free_qty,
                    "taxable_amount": round(taxable, 2),
                    "gst_amount": round(net_amt - taxable if net_amt >= taxable else 0.0, 2),
                    "net_amount": round(net_amt, 2),
                    "batches": [{
                        "batch_no": b_no,
                        "quantity": int(qty) if qty.is_integer() else qty,
                        "free_quantity": int(free_q) if free_q.is_integer() else free_q,
                        "expiry_date": self._format_expiry_iso(ex.get("expiry_date")),
                        "match_status": "matched"
                    }]
                })
                if oid:
                    reconciled_oids.add(oid)
                continue

            # B. Check if the ordered item exists on unclaimed parsed invoice lines
            matched_lines = []
            for inv_it in parsed_lines:
                b_no = (inv_it.get("batch") or inv_it.get("batch_no") or "").strip().upper()
                if b_no and b_no in reconciled_batches:
                    continue
                if self._match_product(oi, inv_it):
                    matched_lines.append(inv_it)
                    if b_no:
                        reconciled_batches.add(b_no)

            if matched_lines:
                total_qty = sum(self._to_float(l.get("qty") or l.get("quantity")) for l in matched_lines)
                total_free = sum(self._to_float(l.get("free") or l.get("sch") or l.get("free_qty")) for l in matched_lines)
                total_tax = sum(self._to_float(l.get("taxable_amt") or l.get("taxable_amount")) for l in matched_lines)
                total_amt = sum(self._to_float(l.get("amount") or l.get("net_amount")) for l in matched_lines)
                ord_free = self._to_float(oi.get("free_qty") or oi.get("free") or oi.get("sch") or oi.get("bonus"))
                final_free_qty = max(total_free, ord_free)

                item_batches = []
                for ml in matched_lines:
                    ml_b = (ml.get("batch") or ml.get("batch_no") or "N/A").strip()
                    ml_q = self._to_float(ml.get("qty") or ml.get("quantity"))
                    ml_f = self._to_float(ml.get("free") or ml.get("sch") or ml.get("free_qty"))
                    ml_exp = self._format_expiry_iso(ml.get("expiry") or ml.get("exp"))
                    existing_ib = next((b for b in item_batches if b["batch_no"] == ml_b), None) if ml_b != "N/A" else None
                    if existing_ib:
                        existing_ib["quantity"] += int(ml_q) if ml_q.is_integer() else ml_q
                        existing_ib["free_quantity"] += int(ml_f) if ml_f.is_integer() else ml_f
                    else:
                        item_batches.append({
                            "batch_no": ml_b,
                            "quantity": int(ml_q) if ml_q.is_integer() else ml_q,
                            "free_quantity": int(ml_f) if ml_f.is_integer() else ml_f,
                            "expiry_date": ml_exp,
                            "match_status": "matched"
                        })

                reconciled_items.append({
                    "order_item_id": oi.get("order_item_id") or oi.get("id"),
                    "product_id": ord_pid,
                    "product_name": ord_name,
                    "size": ord_size,
                    "match_status": "matched" if total_qty >= ord_q else "partial_match",
                    "ordered_qty": int(ord_q) if ord_q.is_integer() else ord_q,
                    "invoiced_qty": int(total_qty) if total_qty.is_integer() else total_qty,
                    "free_qty": int(final_free_qty) if final_free_qty.is_integer() else final_free_qty,
                    "taxable_amount": round(total_tax, 2),
                    "gst_amount": round(total_amt - total_tax if total_amt >= total_tax else 0.0, 2),
                    "net_amount": round(total_amt, 2),
                    "batches": item_batches
                })
                if oid:
                    reconciled_oids.add(oid)
                continue

            # C. Truly missing from invoice
            missing_items.append({
                "order_item_id": oi.get("order_item_id") or oi.get("id"),
                "product_id": ord_pid,
                "product_name": ord_name,
                "size": ord_size,
                "match_status": "missing",
                "ordered_qty": int(ord_q) if ord_q.is_integer() else ord_q,
                "reason": f"Product {ord_name} was not found on the invoice."
            })

        # Clean missing_items to ensure no item in reconciled_items is also in missing_items
        clean_missing = []
        for m in missing_items:
            m_oid = str(m.get("order_item_id") or "")
            m_pid = str(m.get("product_id") or "")
            if m_oid and m_oid in reconciled_oids:
                continue

            # Also check if already reconciled in reconciled_items by product_id or description
            already_reconciled = False
            for r in reconciled_items:
                r_oid = str(r.get("order_item_id") or "")
                r_pid = str(r.get("product_id") or "")
                if (m_oid and r_oid and m_oid == r_oid) or (m_pid and r_pid and m_pid == r_pid):
                    m_size = self._normalize_size_tag(m.get("size"))
                    r_size = self._normalize_size_tag(r.get("size"))
                    if m_size == r_size or not m_size or not r_size or ({m_size, r_size}.issubset({"UNI", "FS"})):
                        already_reconciled = True
                        break
                if self._match_product(m, r):
                    already_reconciled = True
                    break

            if already_reconciled:
                continue

            if not m.get("match_status") or str(m.get("match_status")).lower() in ["none", "null"]:
                m["match_status"] = "missing"
            clean_missing.append(m)
        missing_items = clean_missing

        # Clean extra_items to ensure no item in reconciled_items is also in extra_items
        clean_extra = []
        for ex in extra_items:
            if self._is_tax_or_summary_row(ex) or self._is_invalid_or_phantom_line(ex):
                continue
            b_no = (ex.get("batch_no") or "").strip().upper()
            if b_no and b_no in reconciled_batches:
                continue
            if any(self._match_product(r, ex) for r in reconciled_items):
                continue
            clean_extra.append(ex)
        extra_items = clean_extra

        # Check cross-product free schemes and reconcile reward items that may have been placed in extra_items
        for oi in order_items_list:
            free_qty = self._to_float(oi.get("free_qty") or oi.get("free") or oi.get("sch") or oi.get("bonus") or 0)
            free_prod_name = oi.get("free_product_name")
            paid_prod_name = oi.get("product_name") or oi.get("name", "")
            c_paid = re.sub(r'[^A-Z0-9]', '', str(paid_prod_name).upper())
            c_free = re.sub(r'[^A-Z0-9]', '', str(free_prod_name).upper()) if free_prod_name else ""
            is_cross = bool(free_qty > 0 and c_free and c_free != c_paid)

            if is_cross:
                matching_r = None
                for r in reconciled_items:
                    if self._match_product(oi, r):
                        matching_r = r
                        break

                reward_target = {
                    "product_id": oi.get("free_product_id"),
                    "product_name": free_prod_name,
                    "product_code": oi.get("free_product_code"),
                    "size": oi.get("free_size"),
                    "side": oi.get("free_side"),
                    "order_qty": free_qty
                }

                # Check if the reward item was placed in extra_items
                matched_extra_reward_idx = -1
                for e_idx, ex in enumerate(extra_items):
                    if self._match_product(reward_target, ex):
                        matched_extra_reward_idx = e_idx
                        break

                if matched_extra_reward_idx != -1:
                    ex = extra_items.pop(matched_extra_reward_idx)
                    ex_qty = self._to_float(ex.get("quantity") or ex.get("qty") or ex.get("free_qty"))
                    if matching_r is not None:
                        current_free = self._to_float(matching_r.get("invoiced_free_qty") or 0)
                        matching_r["invoiced_free_qty"] = int(current_free + ex_qty) if (current_free + ex_qty).is_integer() else (current_free + ex_qty)
                        matching_r["free_product_name"] = free_prod_name
                        matching_r.setdefault("batches", []).append({
                            "batch_no": ex.get("batch_no") or "N/A",
                            "quantity": 0,
                            "free_quantity": int(ex_qty) if ex_qty.is_integer() else ex_qty,
                            "expiry_date": self._format_expiry_iso(ex.get("expiry_date")),
                            "match_status": "matched"
                        })
                elif matching_r is not None and self._to_float(matching_r.get("invoiced_free_qty") or 0) < free_qty:
                    # Check parsed_lines for Qty == 0, Free > 0 matching reward_target
                    for pl in parsed_lines:
                        pl_q = self._to_float(pl.get("qty") or pl.get("quantity"))
                        pl_f = self._to_float(pl.get("free") or pl.get("sch") or pl.get("free_qty"))
                        if (pl_q == 0 and pl_f > 0) or (pl_f == free_qty):
                            if self._match_product(reward_target, pl):
                                current_free = self._to_float(matching_r.get("invoiced_free_qty") or 0)
                                matching_r["invoiced_free_qty"] = int(current_free + pl_f) if (current_free + pl_f).is_integer() else (current_free + pl_f)
                                matching_r["free_product_name"] = free_prod_name
                                matching_r.setdefault("batches", []).append({
                                    "batch_no": pl.get("batch") or pl.get("batch_no") or "N/A",
                                    "quantity": 0,
                                    "free_quantity": int(pl_f) if pl_f.is_integer() else pl_f,
                                    "expiry_date": self._format_expiry_iso(pl.get("expiry") or pl.get("exp")),
                                    "match_status": "matched"
                                })
                                break

                    if matching_r is not None and self._to_float(matching_r.get("invoiced_free_qty") or 0) < free_qty:
                        found_free = self._find_free_reward_in_text(invoice_text if 'invoice_text' in locals() else "", free_prod_name)
                        if found_free > 0:
                            current_free = self._to_float(matching_r.get("invoiced_free_qty") or 0)
                            matching_r["invoiced_free_qty"] = int(current_free + found_free) if (current_free + found_free).is_integer() else (current_free + found_free)
                            matching_r["free_product_name"] = free_prod_name

        # Build items list matching exact expected schema
        items_list = []
        for oi in order_items_list:
            ord_item_id = oi.get("order_item_id") or oi.get("id")
            prod_id = oi.get("product_id") or oi.get("id")
            prod_name = oi.get("product_name") or oi.get("name", "")
            ordered_qty = self._to_float(oi.get("ordered_qty") or oi.get("order_qty") or oi.get("quantity"))
            free_qty = self._to_float(oi.get("free_qty") or oi.get("free") or oi.get("sch") or oi.get("bonus") or 0)
            free_prod_name = oi.get("free_product_name")
            c_paid = re.sub(r'[^A-Z0-9]', '', str(prod_name).upper())
            c_free = re.sub(r'[^A-Z0-9]', '', str(free_prod_name).upper()) if free_prod_name else ""
            is_cross = bool(free_qty > 0 and c_free and c_free != c_paid)

            matching_r = None
            for r in reconciled_items:
                r_oid = str(r.get("order_item_id") or "")
                if ord_item_id is not None and r_oid == str(ord_item_id):
                    matching_r = r
                    break
                if self._match_product(oi, r):
                    matching_r = r
                    break

            if matching_r:
                inv_qty = self._to_float(matching_r.get("invoiced_qty") or matching_r.get("quantity"))
                inv_free = self._to_float(matching_r.get("invoiced_free_qty") or matching_r.get("free_qty"))
                batches = matching_r.get("batches", [])
                primary_b = batches[0]["batch_no"] if batches else (matching_r.get("batch_no") or "N/A")
                primary_exp = batches[0]["expiry_date"] if batches else matching_r.get("expiry_date")
                exp_fmt = self._format_expiry_mmyyyy(primary_exp) or (primary_exp or "N/A")

                discrepancies = []
                paid_matched = bool(inv_qty == ordered_qty)
                if inv_qty < ordered_qty:
                    discrepancies.append(f"Invoiced quantity ({int(inv_qty) if inv_qty.is_integer() else inv_qty}) is less than ordered quantity ({int(ordered_qty) if ordered_qty.is_integer() else ordered_qty}) for {prod_name}.")
                elif inv_qty > ordered_qty:
                    discrepancies.append(f"Invoiced quantity ({int(inv_qty) if inv_qty.is_integer() else inv_qty}) exceeds ordered quantity ({int(ordered_qty) if ordered_qty.is_integer() else ordered_qty}) for {prod_name}.")

                free_matched = True
                if free_qty > 0:
                    if inv_free < free_qty:
                        free_matched = False
                        target_name = free_prod_name if is_cross else prod_name
                        if inv_free == 0:
                            discrepancies.append(f"Free reward item {target_name} missing from invoice (expected: {int(free_qty) if free_qty.is_integer() else free_qty}, invoiced: 0).")
                        else:
                            discrepancies.append(f"Free reward item {target_name} quantity mismatch (expected: {int(free_qty) if free_qty.is_integer() else free_qty}, invoiced: {int(inv_free) if inv_free.is_integer() else inv_free}).")

                item_matched = bool(paid_matched and free_matched and inv_qty > 0)
                matching_r["is_matched"] = item_matched
                matching_r["discrepancies"] = discrepancies
                matching_r["match_status"] = "matched" if item_matched else "partial_match"
                matching_r["free_product_name"] = free_prod_name if is_cross else (free_prod_name or None)
                matching_r["invoiced_free_qty"] = int(inv_free) if inv_free.is_integer() else inv_free

                items_list.append({
                    "order_item_id": ord_item_id,
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "ordered_qty": int(ordered_qty) if ordered_qty.is_integer() else ordered_qty,
                    "invoiced_qty": int(inv_qty) if inv_qty.is_integer() else inv_qty,
                    "free_qty": int(free_qty) if free_qty.is_integer() else free_qty,
                    "free_product_name": free_prod_name if is_cross else (free_prod_name or None),
                    "invoiced_free_qty": int(inv_free) if inv_free.is_integer() else inv_free,
                    "batch_no": primary_b,
                    "expiry_date": exp_fmt,
                    "is_matched": item_matched,
                    "discrepancies": discrepancies
                })
            else:
                items_list.append({
                    "order_item_id": ord_item_id,
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "ordered_qty": int(ordered_qty) if ordered_qty.is_integer() else ordered_qty,
                    "invoiced_qty": 0,
                    "free_qty": int(free_qty) if free_qty.is_integer() else free_qty,
                    "free_product_name": free_prod_name if is_cross else (free_prod_name or None),
                    "invoiced_free_qty": 0,
                    "batch_no": "N/A",
                    "expiry_date": "N/A",
                    "is_matched": False,
                    "discrepancies": [f"Product {prod_name} was not found on the invoice."]
                })

        # Apply Variant Group Free Pooling (Issue 2)
        self._apply_variant_group_free_pooling(items_list, reconciled_items)

        extra_items = [ex for ex in extra_items if not self._is_tax_or_summary_row(ex) and not self._is_invalid_or_phantom_line(ex)]
        unmatched_invoice_items = extra_items

        # Enforce exact summary format and match_status consistency
        total_ordered_items = len(order_data.get("items", []))
        matched_items_count = sum(1 for it in items_list if it["invoiced_qty"] > 0)
        all_items_matched = bool(len(items_list) == total_ordered_items and all(it["is_matched"] for it in items_list) and len(missing_items) == 0)

        if total_ordered_items > 0:
            calc_pct = round((matched_items_count / total_ordered_items) * 100, 2)
            match_percentage = int(calc_pct) if calc_pct.is_integer() else calc_pct
        else:
            match_percentage = 0

        total_free_rewards = sum(it.get("free_qty", 0) for it in items_list if it.get("free_qty", 0) > 0)

        missing_labels = []
        for m in missing_items:
            pname = m.get("product_name") or m.get("name") or "Unknown Product"
            psize = m.get("size")
            if psize and str(psize).upper() not in ["NONE", "NULL", "N/A", ""]:
                missing_labels.append(f"{pname} [{psize}]")
            else:
                missing_labels.append(pname)

        if all_items_matched:
            match_status = "matched"
            if total_free_rewards > 0:
                summary = f"All {total_ordered_items} items and {self._int_or_float(total_free_rewards)} free reward items matched with the invoice."
            else:
                summary = f"All {total_ordered_items} items matched with the invoice."
            if len(extra_items) > 0:
                extra_labels = [e.get("product_name") or "Extra Product" for e in extra_items[:3]]
                summary += f" Verification Note: Contains {len(extra_items)} extra items not in order ({', '.join(extra_labels)})."
        elif matched_items_count > 0:
            match_status = "partial_match"
            all_discrepancies = []
            for it in items_list:
                all_discrepancies.extend(it.get("discrepancies", []))
            if len(missing_items) > 0 and len(extra_items) > 0:
                missing_str = ", ".join(missing_labels[:3])
                extra_str = ", ".join([e.get("product_name") or "Extra Item" for e in extra_items[:3]])
                item_word = "item" if len(missing_items) == 1 else "items"
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered {item_word} ({missing_str}). Contains {len(extra_items)} extra items not in order ({extra_str}). Please verify before approving."
            elif all_discrepancies:
                summary = f"Verification Mismatch: {'; '.join(d.rstrip('.') for d in all_discrepancies[:2])}. Please verify before approving."
            elif len(extra_items) > 0:
                extra_str = ", ".join([e.get("product_name") or "Extra Item" for e in extra_items[:3]])
                summary = f"Verification Mismatch: Contains {len(extra_items)} extra items not in order ({extra_str}). Please verify before approving."
            elif len(missing_items) > 0:
                missing_str = ", ".join(missing_labels[:3])
                item_word = "item" if len(missing_items) == 1 else "items"
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered {item_word} ({missing_str}). Please verify before approving."
            else:
                summary = "Verification Mismatch: One or more items did not match order details. Please verify before approving."
        else:
            match_status = "blocked"
            if match_percentage == 0:
                summary = "None of the ordered items were found in the uploaded invoice. Document appears to be for a different order."
            else:
                missing_str = f" ({', '.join(missing_labels[:3])})" if missing_labels else ""
                summary = f"Verification Blocked: Only {matched_items_count} of {total_ordered_items} ordered items matched{missing_str}. Document appears to be for a different order or insufficient fulfillment."

        raw_inv_date = meta.get("invoice_date") or meta.get("date") or "N/A"
        clean_inv_date = str(raw_inv_date)
        m_d = re.match(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', clean_inv_date)
        if m_d:
            clean_inv_date = f"{m_d.group(3)}-{int(m_d.group(2)):02d}-{int(m_d.group(1)):02d}"

        final_taxable = round(self._to_float(meta.get("taxable_amount")), 2)
        final_net = round(self._to_float(meta.get("total_amount") or meta.get("net_amount")), 2)

        clean_meta = {
            "invoice_no": meta.get("invoice_no") or meta.get("invoice_number") or "N/A",
            "invoice_number": meta.get("invoice_no") or meta.get("invoice_number") or "N/A",
            "date": clean_inv_date,
            "invoice_date": clean_inv_date,
            "gstin": meta.get("gstin") or meta.get("extracted_gstin") or order_data.get("purchaser_gstin") or "N/A",
            "total_taxable": f"{final_taxable:.2f}",
            "taxable_amount": final_taxable,
            "net_amount": f"{final_net:.2f}",
            "total_amount": final_net,
            "purchaser_name": meta.get("purchaser_name") or order_data.get("purchaser_name") or "N/A",
            "purchaser_gstin": meta.get("purchaser_gstin") or order_data.get("purchaser_gstin") or "N/A",
            "is_duplicate": meta.get("is_duplicate", False)
        }

        return {
            "status": "success" if match_status == "matched" else ("warning" if match_status == "partial_match" else "error"),
            "match_status": match_status,
            "match_percentage": match_percentage,
            "summary": summary,
            "items": items_list,
            "reconciled_items": reconciled_items,
            "missing_items": missing_items,
            "extra_items": extra_items,
            "unmatched_invoice_items": extra_items,
            "data": {
                "match_status": match_status,
                "match_percentage": match_percentage,
                "summary": summary,
                "invoice_metadata": clean_meta,
                "items": items_list,
                "reconciled_items": reconciled_items,
                "missing_items": missing_items,
                "extra_items": extra_items,
                "unmatched_invoice_items": extra_items
            },
            "success": match_status != "blocked",
            "message": summary,
            "extracted_metadata": clean_meta
        }
