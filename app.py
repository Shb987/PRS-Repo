from flask import Flask, request, jsonify
from flask_cors import CORS
from extractor import PDFExtractor
from parser import InvoiceParser
from prescription_extractor import PrescriptionExtractor
from invoice_validator import InvoiceValidator
import os
import tempfile
import mimetypes
import json
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize Extractors & Validator
extractor = PDFExtractor()
prescription_extractor = PrescriptionExtractor()
invoice_validator = InvoiceValidator()

@app.route('/admin', methods=['POST'])
def extract_admin():
    return process_extraction(mode='admin')

@app.route('/retailer', methods=['POST'])
def extract_retailer():
    return process_extraction(mode='retailer')

@app.route('/prs/<order_type>/validate-invoice', methods=['GET', 'POST'], strict_slashes=False)
@app.route('/prs/<order_type>/validate_invoice', methods=['GET', 'POST'], strict_slashes=False)
@app.route('/prs/validate-invoice', methods=['GET', 'POST'], strict_slashes=False)
@app.route('/prs/validate_invoice', methods=['GET', 'POST'], strict_slashes=False)
@app.route('/validate', methods=['GET', 'POST'], strict_slashes=False)
@app.route('/validate-invoice', methods=['GET', 'POST'], strict_slashes=False)
@app.route('/extract-and-validate', methods=['GET', 'POST'], strict_slashes=False)
def validate_invoice(order_type='retailer', mode=None, **kwargs):
    """
    AI Invoice & Order Verification Endpoint.
    Validates uploaded invoice PDF or image against order contents from Laravel.
    Supports both /prs/validate-invoice and /prs/{order_type}/validate-invoice.
    Returns structured match/mismatch details according to the exact Laravel reconciliation schema.
    """
    if request.method == 'GET':
        return jsonify({
            "status": "ready",
            "service": "AI Invoice Validation Engine",
            "endpoints": [
                "/prs/validate-invoice",
                "/prs/<order_type>/validate-invoice"
            ],
            "method": "POST",
            "description": "Upload invoice file (PDF/image) and order_details JSON payload to validate."
        }), 200

    # Resolve order_type / mode from URL route parameters, query args, or form data
    resolved_order_type = (
        order_type or 
        mode or 
        request.args.get('order_type') or 
        request.args.get('mode') or 
        (request.get_json(silent=True) or {}).get('order_type') or 
        request.form.get('order_type') or 
        'retailer'
    )
    active_mode = resolved_order_type if resolved_order_type in ['admin', 'retailer'] else 'retailer'
    # 1. Parse order_data from request
    order_data = None
    
    if request.is_json:
        req_data = request.get_json() or {}
        order_data = req_data.get('order_details') or req_data.get('order_data') or req_data.get('order') or req_data
        
        # If invoice raw text or parsed data was sent directly in JSON
        invoice_text = req_data.get('invoice_text') or req_data.get('raw_text') or ""
        tables = req_data.get('tables') or []
        parsed_invoice = req_data.get('invoice_data') or req_data.get('parsed_invoice')
        
        if invoice_text or parsed_invoice:
            result = invoice_validator.validate(
                invoice_text=invoice_text,
                invoice_tables=tables,
                order_data=order_data,
                parsed_invoice=parsed_invoice
            )
            return jsonify(result)
            
    # Check form-data for order_details, order_data, or order (sent alongside uploaded PDF/Image)
    order_raw = request.form.get('order_details') or request.form.get('order_data') or request.form.get('order')
    if order_raw:
        try:
            order_data = json.loads(order_raw)
        except Exception:
            order_data = order_raw
            
    # Check for uploaded file (PDF or Image)
    file = request.files.get('file') or request.files.get('invoice') or request.files.get('document')
    if not file or file.filename == '':
        if not order_data:
            return jsonify({"error": "No file or order_details provided in the request"}), 400
        return jsonify({"error": "No file part in the request"}), 400

    filename_lower = file.filename.lower()
    allowed_exts = ('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.tiff', '.bmp')
    if not any(filename_lower.endswith(ext) for ext in allowed_exts):
        return jsonify({"error": f"Invalid file format. Allowed formats: {', '.join(allowed_exts)}"}), 400

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        # Extract text & tables based on file type
        if filename_lower.endswith('.pdf'):
            extraction_results = extractor.extract_text(temp_path)
            raw_text = extraction_results["raw_text"]
            tables = extractor.extract_tables(temp_path)
        else:
            # Image file: OCR extraction
            from PIL import Image
            import pytesseract
            img = Image.open(temp_path)
            raw_text = pytesseract.image_to_string(img)
            tables = []

        # Baseline parsing
        parsed_invoice = InvoiceParser.parse(raw_text, tables, mode=active_mode)

        # Validate against order data using AI Engine
        validation_result = invoice_validator.validate(
            invoice_text=raw_text,
            invoice_tables=tables,
            order_data=order_data or {},
            parsed_invoice=parsed_invoice
        )

        return jsonify(validation_result)

    except Exception as e:
        print(f"[ERROR] Invoice Validation Exception: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/extract-prescription', methods=['POST'])
def extract_prescription():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save to temp
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        # Check if it's an image or PDF
        mime_type, _ = mimetypes.guess_type(temp_path)
        
        if mime_type and (mime_type.startswith('image/') or mime_type == 'application/pdf'):
            image_path = temp_path
            
            if mime_type == 'application/pdf':
                # Convert PDF to images and use the first page
                from pdf2image import convert_from_path
                pages = convert_from_path(temp_path, dpi=200)
                if pages:
                    image_path = os.path.join(temp_dir, f"temp_{os.path.basename(temp_path)}.jpg")
                    pages[0].save(image_path, "JPEG")
                else:
                    return jsonify({"error": "Could not convert PDF to image"}), 400
            
            data = prescription_extractor.extract(image_path)
            
            # Clean up temp image if created
            if image_path != temp_path and os.path.exists(image_path):
                os.remove(image_path)
                
            return jsonify(data)
        else:
            return jsonify({"error": f"Unsupported file type: {mime_type}. Please upload an image or PDF."}), 400

    except Exception as e:
        print(f"[ERROR] Prescription Extraction Exception: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_extraction(mode):
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.pdf'):
        # Save file to a temporary location
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)

        try:
            # 1. Extract Text & Tables
            extraction_results = extractor.extract_text(temp_path)
            raw_text = extraction_results["raw_text"]
            source = extraction_results["extraction_source"]
            
            print(f"\n[DEBUG] {mode.upper()} Extraction Source: {source}")
            
            tables = extractor.extract_tables(temp_path)
            
            # 2. Parse Specific Contents
            extracted_data = InvoiceParser.parse(raw_text, tables, mode=mode)
            
            # If order_data or order_details is passed, include validation in the output
            order_raw = request.form.get('order_details') or request.form.get('order_data') or request.form.get('order')
            if order_raw:
                try:
                    order_data = json.loads(order_raw)
                except Exception:
                    order_data = order_raw

                validation = invoice_validator.validate(
                    invoice_text=raw_text,
                    invoice_tables=tables,
                    order_data=order_data,
                    parsed_invoice=extracted_data
                )
                extracted_data["validation"] = validation
                extracted_data["match_status"] = validation.get("match_status", "matched")
                extracted_data["match_percentage"] = validation.get("match_percentage", 0)
                extracted_data["summary"] = validation.get("summary", "")
                extracted_data["missing_items"] = validation.get("missing_items", [])
                extracted_data["extra_items"] = validation.get("extra_items", [])
                extracted_data["reconciled_items"] = validation.get("reconciled_items", [])
                if "data" in validation:
                    extracted_data["data"] = validation["data"]
            else:
                extracted_data["match_status"] = "unverified"
                extracted_data["match_percentage"] = 0

            # 3. Return full data (metadata + line items)
            return jsonify(extracted_data)

        except Exception as e:
            print(f"[ERROR] API Exception ({mode}): {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    else:
        return jsonify({"error": "Only PDF files are allowed"}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Invoice Extraction & AI Verification API"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)