from PIL import Image, ImageDraw, ImageFont, ImageOps
import requests
import io
import os
from dotenv import load_dotenv
import streamlit as st
load_dotenv()

#API_KEY = os.getenv('HUGGINGFACE_API_KEY')
# Function to generate an image from text using the FLUX model
def generate_image_from_text(prompt):
    API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
    #API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.content

    try:
        image_bytes = query({"inputs": prompt})
        image = Image.open(io.BytesIO(image_bytes))
        return image
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

# Function to generate text using the Meta LLaMA 3.1 model via API
def generate_text_from_prompt(prompt, context):
    API_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3.1-8B-Instruct"
    #API_KEY = os.getenv('HUGGINGFACE_API_KEY')
    headers = {"Authorization": "Bearer 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"}

    try:
        combined_prompt = f"{prompt} {context}" if context else prompt
        response = requests.post(API_URL, headers=headers, json={"inputs": combined_prompt})
        response.raise_for_status()

        output = response.json()
        if isinstance(output, list) and len(output) > 0 and "generated_text" in output[0]:
            return output[0]["generated_text"].split("\n\n")[1]
            # return output[0]["generated_text"]
        else:
            raise ValueError(f"Unexpected response format: {output}")

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while generating the text: {e}")
        return None

# Function to add a watermark text on the generated image
def add_watermark(image: Image, watermark_text: str, font_size: int = 100) -> Image:
    draw = ImageDraw.Draw(image)

    try:
        # Adjust the path to the font file as needed
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Use textbbox method to get the text dimensions
    text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    width, height = image.size
    position = (width - text_width - 10, height - text_height - 10)
    
    draw.text(position, watermark_text, font=font, fill=(255, 255, 255, 128))
    return image
