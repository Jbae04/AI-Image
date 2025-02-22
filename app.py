from flask import Flask, render_template, request
import os
import base64
import logging
from PIL import Image
import io
from dotenv import load_dotenv
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

app = Flask(__name__)

def process_image(image_path, query):
    try:
        with open(image_path, "rb") as image_file:
            image_content = image_file.read()
            encoded_image = base64.b64encode(image_content).decode("utf-8")

        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {"error": f"Invalid image format: {str(e)}"}
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        def make_api_request(model):
            response = requests.post(
                GROQ_API_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000,
                },
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout=30
            )
            return response

        llama_11b_response = make_api_request("llama-3.2-11b-vision-preview")
        llama_90b_response = make_api_request("llama-3.2-90b-vision-preview")

        responses = {
            "llama-11b": "",
            "llama-90b": ""
        }

        for model, response in [("llama-11b", llama_11b_response), ("llama-90b", llama_90b_response)]:
            if response.status_code == 200:
                result = response.json()
                responses[model] = result["choices"][0]["message"]["content"]
                logger.info(f"Processed response from {model} API: {responses[model]}")
            else:
                responses[model] = f"Error: {response.status_code} - {response.text}"
                logger.error(f"Failed to process response from {model} API: {response.status_code} - {response.text}")

        return responses

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html', response_11b=None, response_90b=None)

@app.route('/process_image', methods=['POST'])
def process_uploaded_image():
    image = request.files['image']
    query = request.form['query']
    image_path = os.path.join("uploads", image.filename)
    image.save(image_path)

    result = process_image(image_path, query)

    return render_template('index.html', response_11b=result["llama-11b"], response_90b=result["llama-90b"])

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
