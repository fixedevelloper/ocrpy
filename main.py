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

    # convertir image en bytes
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            prompt,
            {
                "mime_type": "image/png",
                "data": img_bytes.getvalue()
            }
        ]
    )

    text = response.text

    try:
        return json.loads(text)
    except:
        return {"raw_response": text}


@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...)):
    try:
        content = await file.read()
        logger.debug(f"Received file: {file.filename}, size: {len(content)} bytes")

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

        logger.debug(f"Analysis result: {results}")

        return {"success": True, "documents": results}

    except Exception as e:
        logger.exception("Error processing document")  # log complet du traceback
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