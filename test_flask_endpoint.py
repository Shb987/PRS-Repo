import json
import io
from app import app

client = app.test_client()

# Sample order payload from user prompt
order_payload = {
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
      "free_qty": 4,
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

# 1. Test POST /prs/retailer/validate-invoice with file and order_details
pdf_path = "C:/Users/User/Downloads/Invoice AU290.PDF"
with open(pdf_path, "rb") as f:
    pdf_bytes = f.read()

response = client.post(
    "/prs/retailer/validate-invoice",
    data={
        "order_details": json.dumps(order_payload),
        "file": (io.BytesIO(pdf_bytes), "Invoice_AU290.pdf")
    },
    content_type="multipart/form-data"
)

print(f"Status Code: {response.status_code}")
res_json = response.get_json()
print("Response JSON:")
print(json.dumps(res_json, indent=2))
