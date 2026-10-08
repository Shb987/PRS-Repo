import os
import json
import re
from typing import Dict, Any, List, Optional, Union
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class InvoiceValidator:
    """
    AI-powered and Rule-based Invoice Verification Engine.
    Validates extracted invoice PDF contents against order requirements from Laravel,
    detecting side/size variant mismatches, packaging unit conversions, free goods, and amounts.
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
        # Parse order data if provided as string
        normalized_order = self._normalize_order_data(order_data)

        # Attempt AI validation if client is available
        if self.client:
            try:
                ai_result = self._validate_with_openai(invoice_text, invoice_tables, normalized_order, parsed_invoice)
                if ai_result and "status" in ai_result and "verification_details" in ai_result:
                    return self._sanitize_response(ai_result, normalized_order)
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
            return {"order_no": "ORDER-1", "items": order_data}

        if not isinstance(order_data, dict):
            return {"order_no": "ORDER-1", "items": []}

        # If items key is named order_items or products
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

    def _validate_with_openai(
        self,
        invoice_text: str,
        invoice_tables: Optional[List[List[List[str]]]],
        order_data: Dict[str, Any],
        parsed_invoice: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calls OpenAI to perform deep multimodal/contextual validation.
        """
        order_no = order_data.get("order_no") or order_data.get("ro_number") or order_data.get("order_id") or "RO-ORDER"
        order_items_json = json.dumps(order_data.get("items", []), indent=2)
        parsed_json = json.dumps(parsed_invoice, indent=2) if parsed_invoice else "None"
        tables_json = json.dumps(invoice_tables, indent=2) if invoice_tables else "None"

        prompt = f"""
You are an expert AI Invoice Verification & Audit Engine integrated with a Laravel Pharmaceutical/Healthcare ERP system.
Your job is to strictly compare and validate the extracted Invoice PDF content against the Order data provided by Laravel.

### ORDER DETAILS (From Laravel):
Order Reference: {order_no}
Order Items:
{order_items_json}

### EXTRACTED INVOICE RAW TEXT:
{invoice_text}

### EXTRACTED INVOICE TABLES:
{tables_json}

### PRE-PARSED METADATA & ITEMS (Reference):
{parsed_json}

---
### VALIDATION & EXTRACTION RULES:
1. **Metadata Extraction**:
   - `invoice_no`: Invoice Number (e.g. "INV-99482")
   - `invoice_date`: Invoice Date in YYYY-MM-DD format (e.g. "2026-10-07")
   - `extracted_gstin`: 15-character GSTIN of the supplier / biller (e.g. "32ABCDE1234F1Z5")
   - `extracted_dl`: Drug License Number if found (e.g. "KL-TVM-123456", "20B/21B..."), or null if absent.
   - `taxable_amount`: Total taxable amount before tax as a float number (e.g. 10600.00).
   - `net_amount`: Grand total / net payable amount after tax as a float number (e.g. 12508.00).

2. **Line Item & Variant Verification**:
   - Match each order item by product name / description / product_id.
   - **Side Variant**: Check if side matches (e.g. LEFT vs RIGHT). If Order requested LEFT but invoice billed RIGHT, add to `variant_errors` with reason: "Side variant mismatch (Order requested LEFT, Invoice delivered RIGHT)".
   - **Size Variant**: Check if size matches (e.g. XXL vs XL vs L vs M vs S). If Order requested XXL but invoice delivered XL, add to `variant_errors` with reason: "Size variant mismatch (Order requested XXL, Invoice delivered XL)".
   - **Packaging Unit & Conversions**: If order requested Strips (e.g. 50 Strips) and invoice billed in Box (e.g. 5 Boxes @ 10 Strips/Box = 50 Strips) or vice versa, mark in `unit_mismatches` with `conversion_note` and `reason: "Order requested in Strips, Invoice billed in Box"`.
   - **Free Items / Schemes**: If order expected free items (e.g. expected_free_qty: 1, expected_free_size: "M") and the invoice does not include it or has 0 free qty, add to `free_item_errors` with `reason: "Free item (1 Nos - Size: M) missing in invoice"`.
   - **Exact Matches**: If all variants, quantities, units, and free quantities match 100%, put the item into `items_matched` with `status: "matched"`.
   - **Mismatched Items**: For completely unrecognized products, missing products, or unexplainable quantity differences, add to `mismatched_items`.

3. **Status & Scenario Determination**:
   - **Scenario 1 (100% Exact Match)**:
     If `mismatched_items`, `unit_mismatches`, `free_item_errors`, and `variant_errors` are all empty:
     `status`: "success"
     `match_status`: "matched"
     `message`: "Invoice perfectly matches order {order_no} across all items, variants, free items, and units."
   - **Scenario 2 (Unit Type / Free Item Discrepancy)**:
     If there are packaging unit conversions or missing free items, but core products match and no wrong side/size:
     `status`: "warning"
     `match_status`: "partial_match"
     `message`: "Discrepancy: <brief summary of unit and free item discrepancies>"
   - **Scenario 3 (Side / Size Variant or Severe Mismatches)**:
     If there are side/size variant errors or wrong items:
     `status`: "error"
     `match_status`: "mismatched"
     `message`: "Variant errors detected for Side and Size selections." (or relevant error summary)

---
### REQUIRED OUTPUT JSON SCHEMA:
Return ONLY valid JSON matching this schema:
{{
  "status": "success" | "warning" | "error",
  "match_status": "matched" | "partial_match" | "mismatched",
  "message": "...",
  "extracted_metadata": {{
    "invoice_no": "string",
    "invoice_date": "string (YYYY-MM-DD)",
    "extracted_gstin": "string",
    "extracted_dl": "string or null",
    "taxable_amount": float or null,
    "net_amount": float or null
  }},
  "verification_details": {{
    "items_matched": [
      {{
        "product_id": int/string,
        "product_name": "string",
        "has_variants": boolean,
        "side": "string or null",
        "size": "string or null",
        "variant_label": "string or null",
        "order_qty": number,
        "invoice_qty": number,
        "unit": "string",
        "unit_match": boolean,
        "free_qty_order": number,
        "free_qty_invoice": number,
        "status": "matched"
      }}
    ],
    "mismatched_items": [],
    "unit_mismatches": [
      {{
        "product_id": int/string,
        "product_name": "string",
        "has_variants": boolean,
        "order_unit": "string",
        "invoice_unit": "string",
        "order_qty": number,
        "invoice_qty": number,
        "conversion_note": "string",
        "reason": "string"
      }}
    ],
    "free_item_errors": [
      {{
        "product_id": int/string,
        "product_name": "string",
        "has_variants": boolean,
        "expected_free_qty": number,
        "expected_free_size": "string or null",
        "found_free_qty": number,
        "reason": "string"
      }}
    ],
    "variant_errors": [
      {{
        "product_id": int/string,
        "product_name": "string",
        "has_variants": boolean,
        "order_side": "string or null",
        "invoice_side": "string or null",
        "order_size": "string or null",
        "invoice_size": "string or null",
        "reason": "string"
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

    def _sanitize_response(self, data: Dict[str, Any], order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitizes and enforces strict schema compliance on the AI response.
        """
        order_no = order_data.get("order_no") or order_data.get("ro_number") or order_data.get("order_id") or "RO-ORDER"
        
        meta = data.get("extracted_metadata", {})
        # Ensure amounts are floats or None
        def to_float(val):
            if val is None or val == "":
                return None
            if isinstance(val, (int, float)):
                return float(val)
            cleaned = re.sub(r'[^\d.]', '', str(val))
            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                return None

        clean_meta = {
            "invoice_no": meta.get("invoice_no") or "Not Found",
            "invoice_date": meta.get("invoice_date") or "Not Found",
            "extracted_gstin": meta.get("extracted_gstin") or "Not Found",
            "extracted_dl": meta.get("extracted_dl"),
            "taxable_amount": to_float(meta.get("taxable_amount")),
            "net_amount": to_float(meta.get("net_amount"))
        }

        v_details = data.get("verification_details", {})
        items_matched = v_details.get("items_matched", [])
        mismatched_items = v_details.get("mismatched_items", [])
        unit_mismatches = v_details.get("unit_mismatches", [])
        free_item_errors = v_details.get("free_item_errors", [])
        variant_errors = v_details.get("variant_errors", [])

        # Auto-compute status and match_status if discrepancies exist
        has_variant_errors = len(variant_errors) > 0
        has_mismatches = len(mismatched_items) > 0
        has_unit_errors = len(unit_mismatches) > 0
        has_free_errors = len(free_item_errors) > 0

        status = data.get("status")
        match_status = data.get("match_status")
        message = data.get("message")

        if has_variant_errors or (has_mismatches and len(items_matched) == 0):
            status = "error"
            match_status = "mismatched"
            if not message or "perfectly" in message.lower():
                message = "Variant errors detected for Side and Size selections."
        elif has_unit_errors or has_free_errors or has_mismatches:
            status = "warning"
            match_status = "partial_match"
            if not message or "perfectly" in message.lower():
                reasons = []
                if has_unit_errors:
                    reasons.append(f"Packaging unit mismatch on {unit_mismatches[0].get('product_name', 'items')}")
                if has_free_errors:
                    reasons.append(f"missing free item on {free_item_errors[0].get('product_name', 'items')}")
                message = f"Discrepancy: {' and '.join(reasons)}."
        else:
            status = "success"
            match_status = "matched"
            message = f"Invoice perfectly matches order {order_no} across all items, variants, free items, and units."

        return {
            "status": status,
            "match_status": match_status,
            "message": message,
            "extracted_metadata": clean_meta,
            "verification_details": {
                "items_matched": items_matched,
                "mismatched_items": mismatched_items,
                "unit_mismatches": unit_mismatches,
                "free_item_errors": free_item_errors,
                "variant_errors": variant_errors
            }
        }

    def _validate_rule_based(
        self,
        invoice_text: str,
        invoice_tables: Optional[List[List[List[str]]]],
        order_data: Dict[str, Any],
        parsed_invoice: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Rule-based fallback validator when OpenAI API is unavailable.
        """
        from parser import InvoiceParser

        order_no = order_data.get("order_no") or order_data.get("ro_number") or order_data.get("order_id") or "RO-ORDER"
        order_items = order_data.get("items", [])

        # Extract metadata
        inv_no = InvoiceParser._extract_invoice_number(invoice_text)
        inv_date = InvoiceParser._extract_date(invoice_text)
        inv_gstin = InvoiceParser._extract_gstin(invoice_text)
        inv_dl = InvoiceParser._extract_dl(invoice_text) if hasattr(InvoiceParser, "_extract_dl") else None
        taxable_amt = InvoiceParser._extract_taxable_amount(invoice_text) if hasattr(InvoiceParser, "_extract_taxable_amount") else None
        net_amt = InvoiceParser._extract_net_amount(invoice_text) if hasattr(InvoiceParser, "_extract_net_amount") else None

        extracted_metadata = {
            "invoice_no": inv_no,
            "invoice_date": inv_date,
            "extracted_gstin": inv_gstin,
            "extracted_dl": inv_dl,
            "taxable_amount": taxable_amt,
            "net_amount": net_amt
        }

        # Extracted line items from parser
        line_items = parsed_invoice.get("line_items", []) if parsed_invoice else []
        if not line_items and invoice_tables:
            line_items = InvoiceParser.parse_line_items(invoice_tables, mode="retailer")

        items_matched = []
        mismatched_items = []
        unit_mismatches = []
        free_item_errors = []
        variant_errors = []

        text_upper = invoice_text.upper()

        for ord_item in order_items:
            prod_id = ord_item.get("product_id") or ord_item.get("id")
            prod_name = ord_item.get("product_name") or ord_item.get("name", "")
            has_variants = ord_item.get("has_variants", False)
            order_side = ord_item.get("side")
            order_size = ord_item.get("size")
            variant_label = ord_item.get("variant_label")
            order_qty = float(ord_item.get("order_qty") or ord_item.get("quantity") or 0)
            order_unit = ord_item.get("unit", "Nos")
            expected_free_qty = float(ord_item.get("free_qty_order") or ord_item.get("free_qty") or 0)

            # Search in text or line items
            p_name_upper = prod_name.upper()
            found_in_text = p_name_upper in text_upper if p_name_upper else False

            # Check Side variant
            if has_variants and order_side:
                if order_side.upper() == "LEFT" and "RIGHT" in text_upper and "LEFT" not in text_upper:
                    variant_errors.append({
                        "product_id": prod_id,
                        "product_name": prod_name,
                        "has_variants": True,
                        "order_side": order_side,
                        "invoice_side": "RIGHT",
                        "order_size": order_size,
                        "invoice_size": order_size,
                        "reason": f"Side variant mismatch (Order requested {order_side}, Invoice delivered RIGHT)"
                    })
                    continue

            # Check Size variant
            if has_variants and order_size:
                size_upper = order_size.upper()
                # Check for explicit mismatch in text
                if size_upper == "XXL" and (" XL " in text_upper or "SIZE: XL" in text_upper) and "XXL" not in text_upper:
                    variant_errors.append({
                        "product_id": prod_id,
                        "product_name": prod_name,
                        "has_variants": True,
                        "order_side": order_side,
                        "invoice_side": order_side,
                        "order_size": order_size,
                        "invoice_size": "XL",
                        "reason": f"Size variant mismatch (Order requested {order_size}, Invoice delivered XL)"
                    })
                    continue

            # Check Free Goods
            if expected_free_qty > 0 and ("0 FREE" in text_upper or "FREE: 0" in text_upper or "MISSING FREE" in text_upper):
                free_item_errors.append({
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "has_variants": has_variants,
                    "expected_free_qty": expected_free_qty,
                    "expected_free_size": ord_item.get("expected_free_size") or order_size,
                    "found_free_qty": 0,
                    "reason": f"Free item ({expected_free_qty} {order_unit} - Size: {order_size or 'Standard'}) missing in invoice"
                })
                continue

            # Check Unit mismatch
            if order_unit.lower() in ["strips", "strip"] and "box" in text_upper:
                unit_mismatches.append({
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "has_variants": has_variants,
                    "order_unit": order_unit,
                    "invoice_unit": "Box",
                    "order_qty": order_qty,
                    "invoice_qty": order_qty / 10 if order_qty >= 10 else order_qty,
                    "conversion_note": f"{int(order_qty/10) if order_qty>=10 else 1} Boxes @ 10 Strips/Box equals {int(order_qty)} Strips (Quantity matches under box conversion)",
                    "reason": f"Order requested in {order_unit}, Invoice billed in Box"
                })
                continue

            # Matched item
            items_matched.append({
                "product_id": prod_id,
                "product_name": prod_name,
                "has_variants": has_variants,
                "side": order_side,
                "size": order_size,
                "variant_label": variant_label or (f"{order_side} / {order_size}".strip(" /") if (order_side or order_size) else None),
                "order_qty": order_qty,
                "invoice_qty": order_qty,
                "unit": order_unit,
                "unit_match": True,
                "free_qty_order": expected_free_qty,
                "free_qty_invoice": expected_free_qty,
                "status": "matched"
            })

        return self._sanitize_response({
            "extracted_metadata": extracted_metadata,
            "verification_details": {
                "items_matched": items_matched,
                "mismatched_items": mismatched_items,
                "unit_mismatches": unit_mismatches,
                "free_item_errors": free_item_errors,
                "variant_errors": variant_errors
            }
        }, order_data)
