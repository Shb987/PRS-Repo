from extractor import PDFExtractor
from parser import InvoiceParser
import json
import sys
import os

# ==========================================
# CONFIGURATION SECTION
# ==========================================
# You can hardcode your PDF path here for quick testing
DEFAULT_PDF_PATH = r"Invoice AU290.PDF" 
# ==========================================

from tabulate import tabulate

import argparse

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
        # Get headers from the first item's keys (to handle different modes automatically)
        headers = [k.replace("_", " ").title() for k in items[0].keys()]
        for item in items:
            table_data.append(list(item.values()))
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    else:
        print("\n[!] No line items found.")

    print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Invoice Extraction CLI")
    parser.add_argument("pdf_path", nargs="?", default=DEFAULT_PDF_PATH, help="Path to the PDF file")
    parser.add_argument("--mode", choices=["admin", "retailer"], default="admin", help="Extraction mode (admin/retailer)")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    mode = args.mode

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
        
        # 3. Save only line items to JSON file
        json_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_{mode}.json"
        with open(json_filename, "w") as f:
            json.dump(extracted_data, f, indent=4)
        print(f"\n[OK] Results successfully saved to: {json_filename}")
        
        # 4. Output results structurally to terminal
        print_structured_output(extracted_data, mode=mode)
        
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()
