import base64
import requests
import io
from PIL import Image
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

def process_image(image_path, query):
    try:
        with open(image_path, "rb") as image_file:
            image_content =image_file.read()
            encoded_image = base64.b64encode(image_content).decode("utf-8")
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            return {f"error": f"Invalid image format: {str(e)}"}
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
            response =requests.post(
                GROQ_API_URL,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000,
                },
                headers = {
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                timeout = 30
            )
            return response

        llama_llb_response = make_api_request("llama-3.2-11b-vision-preview")
        llama_90b_response = make_api_request("llama-3.2-90b-vision-preview")

        reponses = {}

        for model, response in [("llama-11b", llama_llb_response), ("llama-90b", llama_90b_response)]:
            if response.status_code == 200:
                result = response.json()
                answer = result["choices"][0]["message"]["content"]
                logger.info(f"Processed response from {model} API: {answer}")
                reponses[model] = answer
            else:
                logger.error(f"Failed to process response from {model} API: {response.status_code} - {response.text}")
                reponses[model] = f"An error occurred while processing the image: {response.status_code} - {response.text}"
        return reponses

            
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return {f"error": f"An Unexpected error occured! : {str(e)}"}
    
    

if __name__ == "__main__":
    image_path = "images/test1.png"
    query = "What are the encorders in this image?"
    result = process_image(image_path, query)
    print(result)





