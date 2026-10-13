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
        """
        # Parse and normalize order data
        normalized_order = self._normalize_order_data(order_data)

        # Attempt AI validation if OpenAI client is available
        if self.client:
            try:
                ai_result = self._validate_with_openai(invoice_text, invoice_tables, normalized_order, parsed_invoice)
                if ai_result and "data" in ai_result and "match_status" in ai_result:
                    return self._sanitize_response(ai_result, normalized_order, parsed_invoice)
            except Exception as e:
                print(f"[WARN] OpenAI Validation failed, falling back to rule engine: {e}")

        # Fallback to local rule-based verification engine
        return self._validate_rule_based(invoice_text, invoice_tables, normalized_order, parsed_invoice)

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

    @staticmethod
    def _normalize_product_code(code_str: Optional[str]) -> Optional[str]:
        if not code_str or str(code_str).upper() in ["N/A", "NONE", "NULL", "BLANK", ""]:
            return None
        # Remove whitespace, hyphens, slashes, underscores for robust comparison:
        # e.g., 'AS-D23' -> 'ASD23', 'BP-05' -> 'BP05', 'AS - D23' -> 'ASD23'
        cleaned = re.sub(r'[^A-Z0-9]', '', str(code_str).upper())
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
            "FREE": "FS"
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
    def _match_product(cls, order_item: Dict[str, Any], inv_item: Dict[str, Any]) -> bool:
        """
        Determines whether an invoice line item matches an ordered product variant.
        Handles:
        - Product code recognition (e.g. AS-D23, BP-05) both in dedicated columns and embedded in descriptions.
        - Size variant tags (e.g. [L], [M], [S], (XL), XXL) with strict variant preservation.
        - Side variants (LEFT, RIGHT).
        - Substring and token matching for product descriptions.
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
        inv_raw_code = (
            inv_item.get("p_code") or 
            inv_item.get("product_code") or 
            inv_item.get("code") or 
            inv_item.get("item_code") or 
            cls._extract_product_code(inv_desc)
        )

        norm_ord_code = cls._normalize_product_code(order_raw_code)
        norm_inv_code = cls._normalize_product_code(inv_raw_code)

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
        # A. Size Variant Conflict: If both specify a size and they differ -> Mismatch!
        if norm_ord_size and norm_inv_size and norm_ord_size != norm_inv_size:
            return False

        # If order specifies size, but invoice didn't explicitly extract one, check if inv_desc has conflicting size
        if norm_ord_size and not norm_inv_size:
            detected_inv_size = cls._extract_size_tag(inv_desc)
            if detected_inv_size and detected_inv_size != norm_ord_size:
                return False

        # If invoice specifies size, but order didn't, check if order_name has conflicting size
        if norm_inv_size and not norm_ord_size:
            detected_ord_size = cls._extract_size_tag(order_name)
            if detected_ord_size and detected_ord_size != norm_inv_size:
                return False

        # B. Side Variant Conflict (LEFT vs RIGHT)
        if order_side in ["LEFT", "RIGHT"] and inv_side in ["LEFT", "RIGHT"]:
            if order_side != inv_side:
                return False
        elif order_side in ["LEFT", "RIGHT"]:
            opp_side = "RIGHT" if order_side == "LEFT" else "LEFT"
            if re.search(r'\b' + opp_side + r'\b', inv_desc):
                return False

        # --- PRODUCT CODE MATCHING ---
        # If BOTH have product codes:
        if norm_ord_code and norm_inv_code:
            if norm_ord_code == norm_inv_code:
                # Matching product codes + matching sizes = definite match!
                return True
            else:
                # Distinct explicit product codes (e.g. AS-D23 vs BP-05) -> Not the same product
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

        # --- DESCRIPTIVE TOKEN & FUZZY MATCHING ---
        clean_ord_tokens = set(re.findall(r'[A-Z0-9]+', order_name))
        clean_inv_tokens = set(re.findall(r'[A-Z0-9]+', inv_desc))

        # Filter out generic pharmaceutical descriptors and size tokens from token set
        generic_tokens = {
            "ITEM", "ITEMS", "PRODUCT", "PRODUCTS", "MEDICINE", "TAB", "TABLET", "CAP", "CAPSULE",
            "SYRUP", "INJ", "INJECTION", "PRO", "AIR", "PAIR", "THE", "AND", "OF", "MG", "ML", "GM",
            "NOS", "BOX", "STRIP", "PACK", "S", "M", "L", "XL", "XXL", "XXXL", "XS"
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
2. `batches` Array per Item:
   When an item is fulfilled using multiple batches (e.g. 12 in ASKCM1225 and 8 in KCM02261), group them in that item's `batches` array rather than returning disconnected line items.
3. `invoiced_qty`:
   Sum of all paid quantities for that item across its batches.
4. `free_qty`:
   Sum of all free quantities for that item across its batches.
5. `missing_items`:
   Any ordered product not found on the invoice. Do NOT put items here if they appear on the invoice.
6. `extra_items`:
   Any product printed on the invoice that was not in the original order.
7. `expiry_date` in `batches`:
   ALWAYS extract the batch expiration date from PRE-PARSED INVOICE DATA (field 'expiry') or tables/text. Convert month/year formats like "01/31" into standard ISO "YYYY-MM-DD" (e.g. "2031-01-31", "02/31" -> "2031-02-28", "06/30" -> "2030-06-30"). Never return null when an expiry date is present in the invoice or pre-parsed data.
8. `summary`:
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

        reconciled_items = []
        missing_items = []
        matched_inv_indices = set()

        for ord_item in order_items:
            ord_item_id = ord_item.get("order_item_id") or ord_item.get("id")
            prod_id = ord_item.get("product_id") or ord_item.get("id")
            prod_name = ord_item.get("product_name") or ord_item.get("name", "")
            size = ord_item.get("size") or ord_item.get("variant_label")
            ordered_qty = self._to_float(ord_item.get("order_qty") or ord_item.get("quantity"))

            matched_batches = []
            item_taxable = 0.0
            item_gst = 0.0
            item_net = 0.0
            total_invoiced_qty = 0.0
            total_free_qty = 0.0

            for idx, inv_item in enumerate(line_items):
                if self._match_product(ord_item, inv_item):
                    matched_inv_indices.add(idx)
                    b_no = inv_item.get("batch") or inv_item.get("batch_no") or "N/A"
                    qty = self._to_float(inv_item.get("qty") or inv_item.get("quantity"))
                    free_q = self._to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
                    exp_d = self._format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp"))

                    batch_entry = {
                        "batch_no": b_no,
                        "quantity": int(qty) if qty.is_integer() else qty,
                        "expiry_date": exp_d
                    }
                    if free_q > 0:
                        batch_entry["free_quantity"] = int(free_q) if free_q.is_integer() else free_q

                    matched_batches.append(batch_entry)
                    total_invoiced_qty += qty
                    total_free_qty += free_q

                    taxable = self._to_float(inv_item.get("taxable_amt") or inv_item.get("taxable_amount"))
                    rate = self._to_float(inv_item.get("rate") or inv_item.get("price"))
                    if taxable == 0.0 and rate > 0.0 and qty > 0:
                        taxable = rate * qty

                    amt = self._to_float(inv_item.get("amount") or inv_item.get("net_amount"))
                    if amt == 0.0 and taxable > 0:
                        amt = taxable * 1.05

                    gst = amt - taxable if amt >= taxable else taxable * 0.05

                    item_taxable += taxable
                    item_gst += gst
                    item_net += amt

            if matched_batches:
                item_status = "matched" if total_invoiced_qty >= ordered_qty else "partial_match"
                reconciled_items.append({
                    "order_item_id": ord_item_id,
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "size": size,
                    "match_status": item_status,
                    "ordered_qty": int(ordered_qty) if ordered_qty.is_integer() else ordered_qty,
                    "invoiced_qty": int(total_invoiced_qty) if total_invoiced_qty.is_integer() else total_invoiced_qty,
                    "free_qty": int(total_free_qty) if total_free_qty.is_integer() else total_free_qty,
                    "taxable_amount": round(item_taxable, 2),
                    "gst_amount": round(item_gst, 2),
                    "net_amount": round(item_net, 2),
                    "batches": matched_batches
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
        extra_items = []
        for idx, inv_item in enumerate(line_items):
            if idx not in matched_inv_indices:
                qty = self._to_float(inv_item.get("qty") or inv_item.get("quantity"))
                free_q = self._to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
                taxable = self._to_float(inv_item.get("taxable_amt") or inv_item.get("taxable_amount"))
                amt = self._to_float(inv_item.get("amount") or inv_item.get("net_amount"))
                extra_items.append({
                    "product_name": inv_item.get("description") or inv_item.get("item_name") or "Unknown Product",
                    "batch_no": inv_item.get("batch") or inv_item.get("batch_no") or "N/A",
                    "match_status": "extra",
                    "quantity": int(qty) if qty.is_integer() else qty,
                    "free_qty": int(free_q) if free_q.is_integer() else free_q,
                    "expiry_date": self._format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp")),
                    "taxable_amount": round(taxable, 2),
                    "net_amount": round(amt, 2)
                })

        # Determine match_percentage, match_status and human-readable summary
        total_ordered_items = len(order_items)
        matched_items_count = len(reconciled_items)

        if total_ordered_items > 0:
            calc_pct = round((matched_items_count / total_ordered_items) * 100, 2)
            match_percentage = int(calc_pct) if calc_pct.is_integer() else calc_pct
        else:
            match_percentage = 0

        # Check if quantities match for all reconciled items
        all_quantities_fulfilled = True
        for item in reconciled_items:
            ord_q = self._to_float(item.get("ordered_qty") or item.get("order_qty"))
            inv_q = self._to_float(item.get("invoiced_qty") or item.get("quantity"))
            if inv_q < ord_q:
                all_quantities_fulfilled = False
                break

        # Match status classification
        # "matched": match_percentage == 100% with matching quantities.
        # "partial_match": match_percentage >= 50% (at least half the ordered products exist on the invoice).
        # "blocked" / "unmatched": match_percentage < 50% or 0%.
        if match_percentage == 100 and all_quantities_fulfilled and len(missing_items) == 0:
            match_status = "matched"
            total_ord_qty = sum(r["ordered_qty"] for r in reconciled_items)
            total_free = sum(r["free_qty"] for r in reconciled_items)
            if len(extra_items) > 0:
                summary = f"All {int(total_ord_qty)} ordered items matched across {len(reconciled_items)} variants with {int(total_free)} free units fulfilled. Verification Note: Contains {len(extra_items)} extra items not in order."
            else:
                summary = f"All {int(total_ord_qty)} ordered items matched across {len(reconciled_items)} variants with {int(total_free)} free units fulfilled."
        elif match_percentage >= 50:
            match_status = "partial_match"
            if len(missing_items) > 0 and len(extra_items) > 0:
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items. Contains {len(extra_items)} extra items not in order. Please verify before approving."
            elif len(extra_items) > 0:
                summary = f"Verification Mismatch: Contains {len(extra_items)} extra items not in order. Please verify before approving."
            elif len(missing_items) > 0:
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items ({match_percentage}% items matched). Please verify before approving."
            else:
                summary = f"Verification Mismatch: Invoiced quantity differs from ordered quantity ({match_percentage}% items matched). Please verify before approving."
        else:
            match_status = "blocked"
            if match_percentage == 0:
                summary = "None of the ordered items were found in the uploaded invoice. Document appears to be for a different order."
            else:
                summary = f"Verification Blocked: Only {matched_items_count} of {total_ordered_items} ordered items matched ({match_percentage}%). Document appears to be for a different order or insufficient fulfillment."

        meta_from_parsed = parsed_invoice.get("invoice_metadata", {}) if parsed_invoice else {}
        taxable_sum = sum(r["taxable_amount"] for r in reconciled_items)
        net_sum = sum(r["net_amount"] for r in reconciled_items)

        final_taxable = round(self._to_float(meta_from_parsed.get("taxable_amount") or taxable_amt or taxable_sum), 2)
        final_net = round(self._to_float(meta_from_parsed.get("net_amount") or meta_from_parsed.get("total_amount") or net_amt or net_sum), 2)

        invoice_metadata = {
            "invoice_no": meta_from_parsed.get("invoice_no") or meta_from_parsed.get("invoice_number") or inv_no or "N/A",
            "invoice_date": meta_from_parsed.get("invoice_date") or meta_from_parsed.get("date") or inv_date or "N/A",
            "gstin": meta_from_parsed.get("gstin") or meta_from_parsed.get("extracted_gstin") or inv_gstin or "N/A",
            "taxable_amount": final_taxable,
            "net_amount": final_net,
            "total_amount": final_net
        }

        return {
            "success": match_status != "blocked",
            "match_status": match_status,
            "match_percentage": match_percentage,
            "summary": summary,
            "missing_items": missing_items,
            "extra_items": extra_items,
            "reconciled_items": reconciled_items,
            "data": {
                "match_status": match_status,
                "match_percentage": match_percentage,
                "summary": summary,
                "invoice_metadata": invoice_metadata,
                "reconciled_items": reconciled_items,
                "missing_items": missing_items,
                "extra_items": extra_items
            },
            # Backward-compatibility aliases
            "status": "success" if match_status == "matched" else ("warning" if match_status == "partial_match" else "error"),
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
            "invoice_no": meta.get("invoice_no") or meta.get("invoice_number") or "N/A",
            "invoice_date": meta.get("invoice_date") or meta.get("date") or "N/A",
            "gstin": meta.get("gstin") or meta.get("extracted_gstin") or "N/A",
            "taxable_amount": round(self._to_float(meta.get("taxable_amount")), 2),
            "net_amount": round(self._to_float(meta.get("net_amount") or meta.get("total_amount")), 2),
            "total_amount": round(self._to_float(meta.get("total_amount") or meta.get("net_amount")), 2)
        }

        reconciled_items = inner_data.get("reconciled_items", [])
        missing_items = inner_data.get("missing_items", [])
        extra_items = inner_data.get("extra_items", [])

        # Build batch -> expiry mapping from parsed invoice line items
        batch_expiry_map = {}
        if parsed_invoice:
            for itm in parsed_invoice.get("line_items", []):
                b_code = (itm.get("batch") or itm.get("batch_no") or "").strip().upper()
                exp_raw = itm.get("expiry") or itm.get("exp")
                if b_code and exp_raw and str(exp_raw).upper() not in ["N/A", "NONE", "BLANK", "NULL"]:
                    batch_expiry_map[b_code] = exp_raw

        # Build order item lookup for ground truth ordered quantities and IDs
        order_items_list = order_data.get("items", [])
        order_lookup = {}
        for oi in order_items_list:
            if oi.get("order_item_id") is not None:
                order_lookup[str(oi["order_item_id"])] = oi
            if oi.get("id") is not None:
                order_lookup[str(oi["id"])] = oi
            if oi.get("product_id") is not None:
                order_lookup[str(oi["product_id"])] = oi
            name = (oi.get("product_name") or oi.get("name") or "").upper().strip()
            if name:
                order_lookup[name] = oi

        # Ensure batches inside reconciled_items have correct fields and populated expiry
        reconciled_pids = set()
        for item in reconciled_items:
            pid = item.get("product_id")
            if pid is not None:
                reconciled_pids.add(str(pid))
            oid = item.get("order_item_id")
            if oid is not None:
                reconciled_pids.add(str(oid))

            # Look up ground truth ordered quantity from order_data
            matching_ord = (
                order_lookup.get(str(oid)) or 
                order_lookup.get(str(pid)) or 
                order_lookup.get((item.get("product_name") or "").upper().strip())
            )
            if matching_ord:
                ground_ord_q = self._to_float(matching_ord.get("ordered_qty") or matching_ord.get("order_qty") or matching_ord.get("quantity"))
                if ground_ord_q > 0:
                    item["ordered_qty"] = int(ground_ord_q) if ground_ord_q.is_integer() else ground_ord_q
                if item.get("order_item_id") is None and matching_ord.get("order_item_id") is not None:
                    item["order_item_id"] = matching_ord["order_item_id"]
                if item.get("product_id") is None and matching_ord.get("product_id") is not None:
                    item["product_id"] = matching_ord["product_id"]

            ord_q = self._to_float(item.get("ordered_qty") or item.get("order_qty"))
            inv_q = self._to_float(item.get("invoiced_qty") or item.get("quantity"))
            if not item.get("match_status") or str(item.get("match_status")).lower() in ["none", "null"]:
                item["match_status"] = "matched" if inv_q >= ord_q else "partial_match"

            for b in item.get("batches", []):
                if not b.get("match_status") or str(b.get("match_status")).lower() in ["none", "null"]:
                    b["match_status"] = "matched"
                exp_val = b.get("expiry_date")
                b_no = (b.get("batch_no") or "").strip().upper()
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

        # Cross-validate missing_items against parsed invoice line items
        # If an item was erroneously marked missing but actually exists on invoice, recover it!
        if parsed_invoice and parsed_invoice.get("line_items"):
            parsed_lines = parsed_invoice.get("line_items", [])
            clean_missing = []
            for m in missing_items:
                m_pid = str(m.get("product_id", ""))
                m_oid = str(m.get("order_item_id", ""))
                if (m_pid and m_pid in reconciled_pids) or (m_oid and m_oid in reconciled_pids):
                    continue

                matched_line = None
                for inv_it in parsed_lines:
                    if self._match_product(m, inv_it):
                        matched_line = inv_it
                        break

                if matched_line:
                    matching_ord = (
                        order_lookup.get(m_oid) or 
                        order_lookup.get(m_pid) or 
                        order_lookup.get((m.get("product_name") or "").upper().strip()) or 
                        m
                    )
                    ord_q = self._to_float(matching_ord.get("ordered_qty") or matching_ord.get("quantity") or matching_ord.get("order_qty"))
                    qty = self._to_float(matched_line.get("qty") or matched_line.get("quantity"))
                    free_q = self._to_float(matched_line.get("free") or matched_line.get("sch") or matched_line.get("free_qty"))
                    exp_d = self._format_expiry_iso(matched_line.get("expiry") or matched_line.get("exp"))
                    taxable = self._to_float(matched_line.get("taxable_amt") or matched_line.get("taxable_amount"))
                    amt = self._to_float(matched_line.get("amount") or matched_line.get("net_amount"))
                    reconciled_items.append({
                        "order_item_id": m.get("order_item_id") or matching_ord.get("order_item_id"),
                        "product_id": m.get("product_id") or matching_ord.get("product_id"),
                        "product_name": m.get("product_name") or matching_ord.get("product_name"),
                        "size": m.get("size") or matching_ord.get("size"),
                        "match_status": "matched" if qty >= ord_q else "partial_match",
                        "ordered_qty": int(ord_q) if ord_q.is_integer() else ord_q,
                        "invoiced_qty": int(qty) if qty.is_integer() else qty,
                        "free_qty": int(free_q) if free_q.is_integer() else free_q,
                        "taxable_amount": round(taxable, 2),
                        "gst_amount": round(amt - taxable if amt >= taxable else 0.0, 2),
                        "net_amount": round(amt, 2),
                        "batches": [{
                            "batch_no": matched_line.get("batch") or matched_line.get("batch_no") or "N/A",
                            "quantity": int(qty) if qty.is_integer() else qty,
                            "free_quantity": int(free_q) if free_q.is_integer() else free_q,
                            "expiry_date": exp_d,
                            "match_status": "matched"
                        }]
                    })
                    reconciled_pids.add(str(m.get("order_item_id", "")))
                    reconciled_pids.add(str(m.get("product_id", "")))
                else:
                    if not m.get("match_status") or str(m.get("match_status")).lower() in ["none", "null"]:
                        m["match_status"] = "missing"
                    clean_missing.append(m)
            missing_items = clean_missing
        else:
            clean_missing = []
            for m in missing_items:
                m_pid = str(m.get("product_id", ""))
                m_oid = str(m.get("order_item_id", ""))
                if (m_pid and m_pid in reconciled_pids) or (m_oid and m_oid in reconciled_pids):
                    continue
                if not m.get("match_status") or str(m.get("match_status")).lower() in ["none", "null"]:
                    m["match_status"] = "missing"
                clean_missing.append(m)
            missing_items = clean_missing

        # Enforce exact summary format and match_status consistency
        total_ordered_items = len(order_data.get("items", []))
        matched_items_count = len(reconciled_items)

        if total_ordered_items > 0:
            calc_pct = round((matched_items_count / total_ordered_items) * 100, 2)
            match_percentage = int(calc_pct) if calc_pct.is_integer() else calc_pct
        else:
            match_percentage = 0

        # Check if quantities match for all reconciled items
        all_quantities_fulfilled = True
        for item in reconciled_items:
            ord_q = self._to_float(item.get("ordered_qty") or item.get("order_qty"))
            inv_q = self._to_float(item.get("invoiced_qty") or item.get("quantity"))
            if inv_q < ord_q:
                all_quantities_fulfilled = False
                break

        if match_percentage == 100 and all_quantities_fulfilled and len(missing_items) == 0:
            match_status = "matched"
            total_ord = sum(self._to_float(i.get("ordered_qty") or i.get("order_qty")) for i in reconciled_items)
            total_free = sum(self._to_float(i.get("free_qty")) for i in reconciled_items)
            if len(extra_items) > 0:
                summary = f"All {int(total_ord)} ordered items matched across {len(reconciled_items)} variants with {int(total_free)} free units fulfilled. Verification Note: Contains {len(extra_items)} extra items not in order."
            else:
                summary = f"All {int(total_ord)} ordered items matched across {len(reconciled_items)} variants with {int(total_free)} free units fulfilled."
        elif match_percentage >= 50:
            match_status = "partial_match"
            if len(missing_items) > 0 and len(extra_items) > 0:
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items. Contains {len(extra_items)} extra items not in order. Please verify before approving."
            elif len(extra_items) > 0:
                summary = f"Verification Mismatch: Contains {len(extra_items)} extra items not in order. Please verify before approving."
            elif len(missing_items) > 0:
                summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items ({match_percentage}% items matched). Please verify before approving."
            else:
                summary = f"Verification Mismatch: Invoiced quantity differs from ordered quantity ({match_percentage}% items matched). Please verify before approving."
        else:
            match_status = "blocked"
            if match_percentage == 0:
                summary = "None of the ordered items were found in the uploaded invoice. Document appears to be for a different order."
            else:
                summary = f"Verification Blocked: Only {matched_items_count} of {total_ordered_items} ordered items matched ({match_percentage}%). Document appears to be for a different order or insufficient fulfillment."

        return {
            "success": match_status != "blocked",
            "match_status": match_status,
            "match_percentage": match_percentage,
            "summary": summary,
            "missing_items": missing_items,
            "extra_items": extra_items,
            "reconciled_items": reconciled_items,
            "data": {
                "match_status": match_status,
                "match_percentage": match_percentage,
                "summary": summary,
                "invoice_metadata": clean_meta,
                "reconciled_items": reconciled_items,
                "missing_items": missing_items,
                "extra_items": extra_items
            },
            # Backward-compatibility aliases
            "status": "success" if match_status == "matched" else ("warning" if match_status == "partial_match" else "error"),
            "message": summary,
            "extracted_metadata": clean_meta
        }
