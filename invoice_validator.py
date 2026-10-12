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
    def _match_product(order_item: Dict[str, Any], inv_item: Dict[str, Any]) -> bool:
        """
        Determines whether an invoice line item matches an ordered product variant.
        Handles core token matching, size variants (S, M, L, XL, XXL), and side variants (LEFT, RIGHT).
        """
        order_name = (order_item.get("product_name") or order_item.get("name") or "").upper().strip()
        inv_desc = (inv_item.get("description") or inv_item.get("item_name") or "").upper().strip()

        order_size = (order_item.get("size") or order_item.get("variant_label") or "").upper().strip()
        order_side = (order_item.get("side") or "").upper().strip()

        clean_ord_tokens = set(re.findall(r'[A-Z0-9]+', order_name))
        clean_inv_tokens = set(re.findall(r'[A-Z0-9]+', inv_desc))

        # Filter out generic pharmaceutical descriptors
        generic_tokens = {
            "ITEM", "ITEMS", "PRODUCT", "PRODUCTS", "MEDICINE", "TAB", "TABLET", "CAP", "CAPSULE",
            "SYRUP", "INJ", "INJECTION", "PRO", "AIR", "PAIR", "THE", "AND", "OF", "MG", "ML", "GM"
        }
        core_tokens = [t for t in clean_ord_tokens if t not in generic_tokens]

        if core_tokens:
            overlap = sum(1 for t in core_tokens if (t in clean_inv_tokens) or (len(t) > 3 and any(t in it for it in clean_inv_tokens)))
            if overlap / len(core_tokens) < 0.6:
                return False
        else:
            if order_name not in inv_desc and inv_desc not in order_name:
                return False

        # Side variant validation
        if order_side in ["LEFT", "RIGHT"]:
            if order_side not in clean_inv_tokens and order_side not in inv_desc:
                opp_side = "RIGHT" if order_side == "LEFT" else "LEFT"
                if opp_side in clean_inv_tokens or opp_side in inv_desc:
                    return False

        # Size variant validation
        if order_size in ["S", "M", "L", "XL", "XXL", "XXXL"]:
            inv_size_tokens = set(re.findall(r'\b(S|M|L|XL|XXL|XXXL)\b', inv_desc))
            if inv_size_tokens:
                if order_size not in inv_size_tokens:
                    return False
            else:
                if not re.search(r'\b' + re.escape(order_size) + r'\b', inv_desc):
                    if not inv_desc.endswith(" " + order_size) and not inv_desc.endswith("(" + order_size + ")"):
                        return False

        return True

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
1. `match_status`:
   - "matched": All items and required quantities are fulfilled.
   - "partial_match": Some items have partial quantities, substitutions, missing items, or extra non-ordered items.
   - "mismatched": Completely incorrect invoice (wrong customer, unrelated products).
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
   - If matched: "All <N> ordered items matched across <V> variants with <F> free units fulfilled."
   - If extra items only: "Verification Mismatch: Contains <X> extra items not in order. Please verify before approving."
   - If missing & extra: "Verification Mismatch: Missing <M> ordered items. Contains <X> extra items not in order. Please verify before approving."
   - If missing only: "Verification Mismatch: Missing <M> ordered items. Please verify before approving."

### REQUIRED JSON SCHEMA:
Return ONLY valid JSON matching this schema:
{{
  "success": true,
  "match_status": "matched" | "partial_match" | "mismatched",
  "summary": "string",
  "data": {{
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
            "expiry_date": "string (YYYY-MM-DD) or null"
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
        "ordered_qty": number,
        "reason": "string"
      }}
    ],
    "extra_items": [
      {{
        "product_name": "string",
        "batch_no": "string",
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
                reconciled_items.append({
                    "order_item_id": ord_item_id,
                    "product_id": prod_id,
                    "product_name": prod_name,
                    "size": size,
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
                    "quantity": int(qty) if qty.is_integer() else qty,
                    "free_qty": int(free_q) if free_q.is_integer() else free_q,
                    "expiry_date": self._format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp")),
                    "taxable_amount": round(taxable, 2),
                    "net_amount": round(amt, 2)
                })

        # Determine match status and human-readable summary
        total_ord_qty = sum(self._to_float(o.get("order_qty") or o.get("quantity")) for o in order_items)
        total_fulfilled = sum(r["invoiced_qty"] for r in reconciled_items)
        total_free = sum(r["free_qty"] for r in reconciled_items)

        if len(reconciled_items) == 0:
            match_status = "mismatched"
            summary = "Completely incorrect invoice. No ordered items matched."
        elif len(missing_items) == 0 and len(extra_items) == 0 and total_fulfilled >= total_ord_qty:
            match_status = "matched"
            summary = f"All {int(total_ord_qty)} ordered items matched across {len(reconciled_items)} variants with {int(total_free)} free units fulfilled."
        elif len(missing_items) > 0 and len(extra_items) > 0:
            match_status = "partial_match"
            summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items. Contains {len(extra_items)} extra items not in order. Please verify before approving."
        elif len(extra_items) > 0:
            match_status = "partial_match"
            summary = f"Verification Mismatch: Contains {len(extra_items)} extra items not in order. Please verify before approving."
        elif len(missing_items) > 0:
            match_status = "partial_match"
            summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items. Please verify before approving."
        else:
            match_status = "partial_match"
            summary = f"Verification Mismatch: Invoiced quantity ({int(total_fulfilled)}) differs from ordered quantity ({int(total_ord_qty)})."

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
            "success": True,
            "match_status": match_status,
            "summary": summary,
            "data": {
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

        # Ensure batches inside reconciled_items have correct fields and populated expiry
        reconciled_pids = set()
        for item in reconciled_items:
            pid = item.get("product_id")
            if pid is not None:
                reconciled_pids.add(str(pid))
            oid = item.get("order_item_id")
            if oid is not None:
                reconciled_pids.add(str(oid))

            for b in item.get("batches", []):
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

        # Exclude from missing_items any item that was actually fulfilled in reconciled_items
        clean_missing = []
        for m in missing_items:
            m_pid = str(m.get("product_id", ""))
            m_oid = str(m.get("order_item_id", ""))
            if (m_pid and m_pid in reconciled_pids) or (m_oid and m_oid in reconciled_pids):
                continue
            clean_missing.append(m)
        missing_items = clean_missing

        # Enforce exact summary format and match_status consistency
        if len(reconciled_items) == 0:
            match_status = "mismatched"
            summary = "Completely incorrect invoice. No ordered items matched."
        elif len(missing_items) == 0 and len(extra_items) == 0:
            match_status = "matched"
            total_ord = sum(self._to_float(i.get("ordered_qty") or i.get("order_qty")) for i in reconciled_items)
            total_free = sum(self._to_float(i.get("free_qty")) for i in reconciled_items)
            summary = f"All {int(total_ord)} ordered items matched across {len(reconciled_items)} variants with {int(total_free)} free units fulfilled."
        elif len(missing_items) > 0 and len(extra_items) > 0:
            match_status = "partial_match"
            summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items. Contains {len(extra_items)} extra items not in order. Please verify before approving."
        elif len(extra_items) > 0:
            match_status = "partial_match"
            summary = f"Verification Mismatch: Contains {len(extra_items)} extra items not in order. Please verify before approving."
        elif len(missing_items) > 0:
            match_status = "partial_match"
            summary = f"Verification Mismatch: Missing {len(missing_items)} ordered items. Please verify before approving."
        else:
            match_status = "partial_match"
            summary = "Verification Mismatch: Quantities differ from ordered. Please verify before approving."

        return {
            "success": True,
            "match_status": match_status,
            "summary": summary,
            "data": {
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
