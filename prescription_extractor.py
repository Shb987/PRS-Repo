import os
import base64
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class PrescriptionExtractor:
    """
    Extracts medicine details from medical prescription images using OpenAI's Vision API.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            # Note: In a production environment, you should handle this more gracefully
            # or ensure the environment variable is set before initialization.
            pass
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract(self, image_path: str) -> dict:
        """
        Processes a prescription image and returns structured JSON data.
        """
        if not self.client:
            return {"error": "OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."}

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        base64_image = self._encode_image(image_path)

        prompt = """
        You are a medical assistant specializing in reading doctor's prescriptions.
        Your task is to extract the details of the prescribed medicines from the provided image.
        
        Extract the following fields for each medicine:
        - name: The name of the medicine.
        - dosage: The dosage (e.g., 500mg, 5ml).
        - frequency: How often to take it (e.g., twice a day, 1-0-1).
        - duration: How long to take it (e.g., 5 days, 1 month).
        - notes: Any additional instructions (e.g., after food, at bedtime).
        
        Return the result as a JSON object with a key 'medicines' containing a list of objects.
        If a field is not found, use 'N/A'.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=1000,
            )

            result = json.loads(response.choices[0].message.content)
            return result

        except Exception as e:
            print(f"[ERROR] OpenAI Extraction failed: {e}")
            return {"error": f"OpenAI logic failed: {str(e)}"}

if __name__ == "__main__":
    # Example usage for testing
    pass
