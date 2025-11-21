# pipeline.py

import cv2

from ocr import extract_text
from signature import check_signature_presence
from fusion import fuse_results

# Modules optionnels (si tu les ajoutes plus tard)
try:
    from photo import detect_photo
except:
    def detect_photo(image):
        return False

try:
    from checkbox import detect_checkboxes
except:
    def detect_checkboxes(image):
        return []
        

def run_full_pipeline(image_path):
    """
    Run all processing steps:
    - Load image
    - OCR extraction
    - Signature detection
    - Photo detection
    - Checkbox detection
    - Fusion of results
    """

    # 1. Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Impossible de lire l’image : {image_path}")

    # 2. OCR extraction
    ocr_text, ocr_conf = extract_text(image)

    # 3. Signature detection
    signature_present, signature_score = check_signature_presence(image)

    # 4. Photo detection (fallback = False)
    photo_found = detect_photo(image)

    # 5. Checkbox detection (fallback = [])
    checkboxes = detect_checkboxes(image)

    # 6. Fusion finale
    result = fuse_results(
        ocr_text=ocr_text,
        ocr_conf=ocr_conf,
        signature_present=signature_present,
        signature_score=signature_score,
        photo_found=photo_found,
        checkboxes=checkboxes
    )

    return result


if __name__ == "__main__":
    test_image = "test.jpg"  # à remplacer
    res = run_full_pipeline(test_image)
    print(res)
