import json
import re
import calendar

def format_expiry_iso(exp_str):
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

def to_float(val):
    if val is None or val == "" or str(val).upper() in ["N/A", "NONE"]:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    cleaned = re.sub(r'[^\d.]', '', str(val))
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0

def match_product(order_item, inv_item):
    order_name = (order_item.get("product_name") or order_item.get("name") or "").upper().strip()
    inv_desc = (inv_item.get("description") or inv_item.get("item_name") or "").upper().strip()
    
    order_size = (order_item.get("size") or order_item.get("variant_label") or "").upper().strip()
    order_side = (order_item.get("side") or "").upper().strip()
    
    clean_ord_tokens = set(re.findall(r'[A-Z0-9]+', order_name))
    clean_inv_tokens = set(re.findall(r'[A-Z0-9]+', inv_desc))
    
    core_tokens = [t for t in clean_ord_tokens if t not in ["ITEM", "ITEMS", "PRODUCT", "PRODUCTS", "MEDICINE", "TAB", "TABLET", "CAP", "CAPSULE", "SYRUP", "INJ", "INJECTION", "PRO", "AIR", "PAIR", "THE", "AND", "OF"]]
    if core_tokens:
        overlap = sum(1 for t in core_tokens if (t in clean_inv_tokens) or (len(t) > 3 and any(t in it for it in clean_inv_tokens)))
        if overlap / len(core_tokens) < 0.6:
            return False
    else:
        if order_name not in inv_desc and inv_desc not in order_name:
            return False

    if order_side in ["LEFT", "RIGHT"]:
        if order_side not in clean_inv_tokens and order_side not in inv_desc:
            opp_side = "RIGHT" if order_side == "LEFT" else "LEFT"
            if opp_side in clean_inv_tokens or opp_side in inv_desc:
                return False

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

def reconcile_invoice(order_payload, invoice_data):
    order_items = order_payload.get("items", [])
    inv_items = invoice_data.get("line_items", [])
    inv_meta = invoice_data.get("invoice_metadata", {})
    
    reconciled_items = []
    missing_items = []
    matched_inv_indices = set()
    
    for ord_item in order_items:
        ord_item_id = ord_item.get("order_item_id")
        prod_id = ord_item.get("product_id")
        prod_name = ord_item.get("product_name") or ord_item.get("name")
        size = ord_item.get("size") or ord_item.get("variant_label")
        side = ord_item.get("side")
        ordered_qty = to_float(ord_item.get("order_qty") or ord_item.get("quantity"))
        
        # Find all matching invoice line items
        matched_batches = []
        item_taxable = 0.0
        item_gst = 0.0
        item_net = 0.0
        total_invoiced_qty = 0.0
        total_free_qty = 0.0
        
        for idx, inv_item in enumerate(inv_items):
            if match_product(ord_item, inv_item):
                matched_inv_indices.add(idx)
                b_no = inv_item.get("batch") or inv_item.get("batch_no") or "N/A"
                qty = to_float(inv_item.get("qty") or inv_item.get("quantity"))
                free_q = to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
                exp_d = format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp"))
                
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
                
                taxable = to_float(inv_item.get("taxable_amt") or inv_item.get("taxable_amount"))
                rate = to_float(inv_item.get("rate") or inv_item.get("price"))
                if taxable == 0.0 and rate > 0.0 and qty > 0:
                    taxable = rate * qty
                
                amt = to_float(inv_item.get("amount") or inv_item.get("net_amount"))
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
            
    extra_items = []
    for idx, inv_item in enumerate(inv_items):
        if idx not in matched_inv_indices:
            qty = to_float(inv_item.get("qty") or inv_item.get("quantity"))
            free_q = to_float(inv_item.get("free") or inv_item.get("sch") or inv_item.get("free_qty"))
            taxable = to_float(inv_item.get("taxable_amt") or inv_item.get("taxable_amount"))
            amt = to_float(inv_item.get("amount") or inv_item.get("net_amount"))
            extra_items.append({
                "product_name": inv_item.get("description") or inv_item.get("item_name") or "Unknown Product",
                "batch_no": inv_item.get("batch") or inv_item.get("batch_no") or "N/A",
                "quantity": int(qty) if qty.is_integer() else qty,
                "free_qty": int(free_q) if free_q.is_integer() else free_q,
                "expiry_date": format_expiry_iso(inv_item.get("expiry") or inv_item.get("exp")),
                "taxable_amount": round(taxable, 2),
                "net_amount": round(amt, 2)
            })
            
    # Evaluation
    total_ord_qty = sum(to_float(o.get("order_qty") or o.get("quantity")) for o in order_items)
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

    taxable_sum = sum(r["taxable_amount"] for r in reconciled_items)
    net_sum = sum(r["net_amount"] for r in reconciled_items)
    
    clean_meta = {
        "invoice_no": inv_meta.get("invoice_no") or inv_meta.get("invoice_number") or "N/A",
        "invoice_date": inv_meta.get("invoice_date") or inv_meta.get("date") or "N/A",
        "gstin": inv_meta.get("gstin") or inv_meta.get("extracted_gstin") or "N/A",
        "taxable_amount": round(to_float(inv_meta.get("taxable_amount") or taxable_sum), 2),
        "net_amount": round(to_float(inv_meta.get("net_amount") or inv_meta.get("total_amount") or net_sum), 2),
        "total_amount": round(to_float(inv_meta.get("total_amount") or inv_meta.get("net_amount") or net_sum), 2)
    }

    return {
        "success": True,
        "match_status": match_status,
        "summary": summary,
        "data": {
            "invoice_metadata": clean_meta,
            "reconciled_items": reconciled_items,
            "missing_items": missing_items,
            "extra_items": extra_items
        }
    }

order_example = {
  "order_code": "DO-2600016",
  "purchaser_name": "Demo Distributor",
  "purchaser_gstin": "32AAQCA0808D1Z5",
  "items": [
    {
      "product_id": 101,
      "order_item_id": 501,
      "product_name": "KNEE CAP",
      "side": None,
      "size": "M",
      "variant_label": "M",
      "order_qty": 20,
      "free_qty": 12,
      "unit": "Nos"
    },
    {
      "product_id": 102,
      "order_item_id": 502,
      "product_name": "KNEE CAP",
      "side": None,
      "size": "L",
      "variant_label": "L",
      "order_qty": 20,
      "free_qty": 0,
      "unit": "Nos"
    },
    {
      "product_id": 103,
      "order_item_id": 503,
      "product_name": "KNEE CAP",
      "side": None,
      "size": "XL",
      "variant_label": "XL",
      "order_qty": 20,
      "free_qty": 0,
      "unit": "Nos"
    }
  ]
}

invoice_example = {
    "invoice_metadata": {
        "invoice_no": "AV000170",
        "invoice_date": "2026-10-08",
        "gstin": "32AAQCA0808D1Z5",
        "taxable_amount": 8700.00,
        "net_amount": 9135.00,
        "total_amount": 9135.00
    },
    "line_items": [
      {"description": "AS-L01 KNEE CAP(PAIR) M", "batch": "ASKCM1225", "qty": 12, "free": 0, "rate": 145.00, "taxable_amt": 1740.00, "expiry": "12/30"},
      {"description": "AS-L01 KNEE CAP(PAIR) M", "batch": "KCM02261", "qty": 8, "free": 0, "rate": 145.00, "taxable_amt": 1160.00, "expiry": "02/31"},
      {"description": "AS-L01 KNEE CAP(PAIR) M", "batch": "ASKCM02261", "qty": 0, "free": 4, "rate": 145.00, "taxable_amt": 0.00, "expiry": "02/31"},
      {"description": "AS-L01 KNEE CAP PAIR L", "batch": "ASKCL06261", "qty": 20, "free": 4, "rate": 145.00, "taxable_amt": 2900.00, "expiry": "06/31"},
      {"description": "AS-L01 KNEE CAP(PAIR) XL", "batch": "ASKCXL06261", "qty": 20, "free": 4, "rate": 145.00, "taxable_amt": 2900.00, "expiry": "06/31"}
    ]
}

result = reconcile_invoice(order_example, invoice_example)
print(json.dumps(result, indent=2))
