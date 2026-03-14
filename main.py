from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
import google.generativeai as genai
import numpy as np
import cv2
import io
import json

app = FastAPI()

# configuration Gemini
genai.configure(api_key="YOUR_GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")


def preprocess_image(image: Image.Image):
    """Améliore l'image pour OCR"""
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # améliorer contraste
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
    """Analyse le document avec Gemini et retourne JSON"""
    prompt = """
    Analyse ce document et retourne uniquement du JSON valide.

    Exemple de format :

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

    # gérer PDF
    if file.filename.lower().endswith(".pdf"):
        pdf_images = convert_from_bytes(content)
        for img in pdf_images:
            images.append(preprocess_image(img))
    else:
        image = Image.open(io.BytesIO(content))
        images.append(preprocess_image(image))

    results = [analyze_with_gemini(img) for img in images]

    return {"success": True, "documents": results}