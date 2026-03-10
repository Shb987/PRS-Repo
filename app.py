from flask import Flask, request, jsonify
from flask_cors import CORS
from extractor import PDFExtractor
from parser import InvoiceParser
from prescription_extractor import PrescriptionExtractor
import os
import tempfile
import mimetypes
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize Extractors
extractor = PDFExtractor()
prescription_extractor = PrescriptionExtractor()

@app.route('/admin', methods=['POST'])
def extract_admin():
    return process_extraction(mode='admin')

@app.route('/retailer', methods=['POST'])
def extract_retailer():
    return process_extraction(mode='retailer')

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
    return jsonify({"status": "healthy", "service": "Invoice Extraction API"})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)