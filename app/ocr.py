# ocr.py
import cv2
import numpy as np
import easyocr

# Charger le lecteur EasyOCR une seule fois (anglais + arabe)
reader = easyocr.Reader(['en', 'ar'], gpu=False)

def preprocess_for_ocr(image):
    """
    Basic preprocessing for OCR: grayscale + slight denoise.
    EasyOCR gère déjà bien le bruit donc on évite les binarizations agressives.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray

def extract_text(image):
    """
    Extract text using EasyOCR.
    Returns:
    - text: str
    - confidence: float (0–100)
    """

    processed = preprocess_for_ocr(image)

    results = reader.readtext(processed)

    if len(results) == 0:
        return "", 0.0

    # Récupérer texte et confiances
    texts = []
    confidences = []

    for (bbox, txt, conf) in results:
        if txt.strip() != "":
            texts.append(txt)
            confidences.append(conf * 100)   # EasyOCR donne une conf entre 0 et 1

    full_text = " ".join(texts)
    avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0

    return full_text, avg_conf
