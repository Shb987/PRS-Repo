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

@app.route('/validate', methods=['POST'])
@app.route('/validate-invoice', methods=['POST'])
@app.route('/extract-and-validate', methods=['POST'])
def validate_invoice():
    """
    AI Invoice & Order Verification Endpoint.
    Validates uploaded invoice PDF against order contents from Laravel.
    Returns structured match/mismatch details according to Scenarios 1, 2, and 3.
    """
    # 1. Parse order_data from request
    order_data = None
    
    if request.is_json:
        req_data = request.get_json() or {}
        order_data = req_data.get('order_data') or req_data.get('order') or req_data.get('order_details') or req_data
        
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
            
    # Check form-data for order_data (e.g. sent alongside uploaded PDF)
    if 'order_data' in request.form:
        try:
            order_data = json.loads(request.form['order_data'])
        except Exception:
            order_data = request.form['order_data']
    elif 'order' in request.form:
        try:
            order_data = json.loads(request.form['order'])
        except Exception:
            order_data = request.form['order']
            
    # Check for uploaded PDF file
    if 'file' not in request.files:
        if not order_data:
            return jsonify({"error": "No file or order_data provided in the request"}), 400
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, file.filename)
    file.save(temp_path)

    try:
        # Extract text & tables from PDF
        extraction_results = extractor.extract_text(temp_path)
        raw_text = extraction_results["raw_text"]
        tables = extractor.extract_tables(temp_path)

        # Baseline parsing
        parsed_invoice = InvoiceParser.parse(raw_text, tables, mode='retailer')

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
                pages = convert_from_path(temp_path, dpi=300)
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
            
            # If order_data is passed, include validation in the output
            order_data = request.form.get('order_data') or request.form.get('order')
            if order_data:
                validation = invoice_validator.validate(
                    invoice_text=raw_text,
                    invoice_tables=tables,
                    order_data=order_data,
                    parsed_invoice=extracted_data
                )
                extracted_data["validation"] = validation

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