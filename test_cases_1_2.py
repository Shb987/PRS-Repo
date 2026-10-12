import json
from test_reconcile_schema import reconcile_invoice

# Case 1: 5 Items Ordered + 1 Extra Item on Invoice
order_case1 = {
    "order_code": "DO-2600017",
    "items": [
        {"order_item_id": 1, "product_id": 1, "product_name": "Item A", "order_qty": 10},
        {"order_item_id": 2, "product_id": 2, "product_name": "Item B", "order_qty": 10},
        {"order_item_id": 3, "product_id": 3, "product_name": "Item C", "order_qty": 10},
        {"order_item_id": 4, "product_id": 4, "product_name": "Item D", "order_qty": 10},
        {"order_item_id": 5, "product_id": 5, "product_name": "Item E", "order_qty": 10}
    ]
}
invoice_case1 = {
    "invoice_metadata": {"invoice_no": "INV-001"},
    "line_items": [
        {"description": "Item A", "batch": "B1", "qty": 10, "expiry": "05/28"},
        {"description": "Item B", "batch": "B2", "qty": 10, "expiry": "05/28"},
        {"description": "Item C", "batch": "B3", "qty": 10, "expiry": "05/28"},
        {"description": "Item D", "batch": "B4", "qty": 10, "expiry": "05/28"},
        {"description": "Item E", "batch": "B5", "qty": 10, "expiry": "05/28"},
        {"description": "Extra Item F", "batch": "B6", "qty": 5, "expiry": "05/28"}
    ]
}

res1 = reconcile_invoice(order_case1, invoice_case1)
print("=== CASE 1 ===")
print("match_status:", res1["match_status"])
print("summary:", res1["summary"])
print("missing_items:", len(res1["data"]["missing_items"]))
print("extra_items:", len(res1["data"]["extra_items"]))

# Case 2: 6 Items Ordered, 5 Match, 6th Item Missing & Extra on Invoice
order_case2 = {
    "order_code": "DO-2600018",
    "items": [
        {"order_item_id": 1, "product_id": 1, "product_name": "Item A", "order_qty": 10},
        {"order_item_id": 2, "product_id": 2, "product_name": "Item B", "order_qty": 10},
        {"order_item_id": 3, "product_id": 3, "product_name": "Item C", "order_qty": 10},
        {"order_item_id": 4, "product_id": 4, "product_name": "Item D", "order_qty": 10},
        {"order_item_id": 5, "product_id": 5, "product_name": "Item E", "order_qty": 10},
        {"order_item_id": 6, "product_id": 6, "product_name": "Paracetamol 500mg", "order_qty": 10}
    ]
}
invoice_case2 = {
    "invoice_metadata": {"invoice_no": "INV-002"},
    "line_items": [
        {"description": "Item A", "batch": "B1", "qty": 10, "expiry": "05/28"},
        {"description": "Item B", "batch": "B2", "qty": 10, "expiry": "05/28"},
        {"description": "Item C", "batch": "B3", "qty": 10, "expiry": "05/28"},
        {"description": "Item D", "batch": "B4", "qty": 10, "expiry": "05/28"},
        {"description": "Item E", "batch": "B5", "qty": 10, "expiry": "05/28"},
        {"description": "Dolo 650mg", "batch": "B6", "qty": 10, "expiry": "05/28"}
    ]
}

res2 = reconcile_invoice(order_case2, invoice_case2)
print("\n=== CASE 2 ===")
print("match_status:", res2["match_status"])
print("summary:", res2["summary"])
print("missing_items:", len(res2["data"]["missing_items"]))
print("extra_items:", len(res2["data"]["extra_items"]))
