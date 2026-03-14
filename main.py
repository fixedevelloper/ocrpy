from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
import numpy as np
import cv2
import io
import json
import os
from dotenv import load_dotenv
import uvicorn

# charger .env
load_dotenv()

app = FastAPI()

# variables environnement
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# configuration Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(GEMINI_MODEL)


def preprocess_image(image: Image.Image):
    """Améliore l'image pour OCR"""
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return Image.fromarray(thresh)


def analyze_with_gemini(image):
    """Analyse le document avec Gemini"""

    prompt = """
    Analyse ce document et retourne uniquement du JSON valide.

    Format :

    {
      "type_document": null,
      "nom": null,
      "date": null,
      "montant": null,
      "reference": null,
      "description": null
    }

    Si une information n'existe pas retourne null.
    """

    response = model.generate_content([prompt, image])
    text = response.text

    try:
        return json.loads(text)
    except:
        return {"raw_response": text}


@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):

    content = await file.read()

    images = []

    # PDF
    if file.filename.lower().endswith(".pdf"):

        pdf_images = convert_from_bytes(content)

        for img in pdf_images:
            images.append(preprocess_image(img))

    else:

        image = Image.open(io.BytesIO(content))
        images.append(preprocess_image(image))

    results = []

    for img in images:
        results.append(analyze_with_gemini(img))

    return {
        "success": True,
        "documents": results
    }
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port)