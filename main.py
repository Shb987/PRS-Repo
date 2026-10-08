from extractor import PDFExtractor
from parser import InvoiceParser
from invoice_validator import InvoiceValidator
import json
import sys
import os
import argparse
from tabulate import tabulate

# ==========================================
# CONFIGURATION SECTION
# ==========================================
DEFAULT_PDF_PATH = r"Invoice AU290.PDF" 
# ==========================================

def print_structured_output(data, mode="admin"):
    """
    Prints the extracted data in a professional, structured table format.
    """
    metadata = data.get("invoice_metadata", {})
    items = data.get("line_items", [])

    print("\n" + "+" + "-"*58 + "+")
    print(f"| {mode.upper()} INVOICE EXTRACTION RESULTS ".center(58, "-") + "|")
    print("+" + "-"*58 + "+")

    # 1. Print Metadata
    print("\n[+] INVOICE DETAILS")
    print("-" * 30)
    for key, value in metadata.items():
        label = key.replace("_", " ").title()
        print(f" {label:<18}: {value}")
    print("-" * 30)

    # 2. Print Line Items
    if items:
        print(f"\n[+] {mode.upper()} PRODUCT LIST")
        table_data = []
        headers = [k.replace("_", " ").title() for k in items[0].keys()]
        for item in items:
            table_data.append(list(item.values()))
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("\n[!] No line items found.")

    print("\n" + "="*60 + "\n")

def print_validation_output(validation_result):
    """
    Prints the AI verification results cleanly.
    """
    print("\n" + "="*60)
    print("🔍 INVOICE VERIFICATION & AUDIT RESULTS")
    print("="*60)
    print(f"Status       : {validation_result.get('status', '').upper()}")
    print(f"Match Status : {validation_result.get('match_status', '').upper()}")
    print(f"Message      : {validation_result.get('message', '')}")
    print("-" * 60)
    
    meta = validation_result.get('extracted_metadata', {})
    print("Extracted Metadata:")
    for k, v in meta.items():
        print(f"  - {k:<18}: {v}")
        
    v_details = validation_result.get('verification_details', {})
    print("\nVerification Details:")
    print(f"  - Matched Items    : {len(v_details.get('items_matched', []))}")
    print(f"  - Mismatched Items : {len(v_details.get('mismatched_items', []))}")
    print(f"  - Unit Mismatches  : {len(v_details.get('unit_mismatches', []))}")
    print(f"  - Free Item Errors : {len(v_details.get('free_item_errors', []))}")
    print(f"  - Variant Errors   : {len(v_details.get('variant_errors', []))}")
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Invoice Extraction & AI Verification CLI")
    parser.add_argument("pdf_path", nargs="?", default=DEFAULT_PDF_PATH, help="Path to the PDF file")
    parser.add_argument("--mode", choices=["admin", "retailer"], default="admin", help="Extraction mode (admin/retailer)")
    parser.add_argument("--order", help="Path to Order JSON file for verification")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    mode = args.mode
    order_file = args.order

    if not pdf_path or not os.path.exists(pdf_path):
        if not pdf_path:
            pdf_path = input("Please enter the path to the PDF file: ").strip()
        
        if not os.path.exists(pdf_path):
            print(f"Error: File '{pdf_path}' not found.")
            return

    try:
        extractor = PDFExtractor()
        
        # 1. Extract Raw Text & Tables
        extraction_result = extractor.extract_text(pdf_path)
        raw_text = extraction_result["raw_text"]
        source = extraction_result["extraction_source"]
        
        tables = extractor.extract_tables(pdf_path)
        
        print(f"[*] Extraction Source: {source} (Mode: {mode})")
        
        # 2. Parse Specific Contents
        extracted_data = InvoiceParser.parse(raw_text, tables, mode=mode)
        
        # 3. If Order file provided, validate
        if order_file and os.path.exists(order_file):
            with open(order_file, "r") as f:
                order_data = json.load(f)
            validator = InvoiceValidator()
            validation_result = validator.validate(
                invoice_text=raw_text,
                invoice_tables=tables,
                order_data=order_data,
                parsed_invoice=extracted_data
            )
            print_validation_output(validation_result)
            
            # Save validation json
            val_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_validation.json"
            with open(val_filename, "w") as f:
                json.dump(validation_result, f, indent=4)
            print(f"[OK] Validation result saved to: {val_filename}")
        else:
            # Output results structurally to terminal
            print_structured_output(extracted_data, mode=mode)
            
            # Save extraction to JSON file
            json_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{mode}.json"
            with open(json_filename, "w") as f:
                json.dump(extracted_data, f, indent=4)
            print(f"\n[OK] Results successfully saved to: {json_filename}")
        
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
