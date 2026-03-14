from fastapi import FastAPI, UploadFile, File
from pdf2image import convert_from_bytes
from PIL import Image
from google import genai
import numpy as np
import cv2
import io
import json
import os
from dotenv import load_dotenv
import uvicorn
import logging
import pytesseract
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)
# charger .env
load_dotenv()

app = FastAPI()

# variables environnement
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# nouveau client Gemini
client = genai.Client(api_key=GEMINI_API_KEY)



def extract_text_from_image(image: Image.Image) -> str:
    """Extrait le texte brut depuis l'image"""
    return pytesseract.image_to_string(image, lang="fra")  # "fra" pour français
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


def analyze_text_with_gemini(text: str):
    """Analyse le texte extrait avec Gemini pour retourner JSON"""
    prompt = f"""
    Voici le texte d'un document : 
    {text}

    Analyse et retourne uniquement un JSON valide avec ce format :

    {{
      "type_document": null,
      "nom": null,
      "date": null,
      "montant": null,
      "reference": null,
      "description": null
    }}

    Si une information n'existe pas, retourne null.
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[prompt]
    )

    try:
        return json.loads(response.text)
    except:
        return {"raw_response": response.text}

@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        logger.debug(f"Received file: {file.filename}, size: {len(content)} bytes")

        images = []

        # Convertir PDF en images
        if file.filename.lower().endswith(".pdf"):
            pdf_images = convert_from_bytes(content)
            for img in pdf_images:
                images.append(preprocess_image(img))
        else:
            image = Image.open(io.BytesIO(content))
            images.append(preprocess_image(image))

        results = []
        for img in images:
            text = extract_text_from_image(img)       # 1️⃣ Extraire texte
            json_result = analyze_text_with_gemini(text)  # 2️⃣ Envoyer texte à Gemini
            results.append(json_result)

        logger.debug(f"Analysis result: {results}")
        return {"success": True, "documents": results}

    except Exception as e:
        logger.exception("Error processing document")
        return {"success": False, "error": str(e)}
    
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run(
        "main:app",           # Nom du module et de l'app
        host="0.0.0.0",
        port=port,
        reload=True,          # 🔄 active le rechargement automatique (dev)
        log_level="debug"     # 🐛 logs détaillés pour debugger
    )