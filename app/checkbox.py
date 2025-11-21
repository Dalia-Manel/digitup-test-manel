# checkbox.py
import cv2

def detect_checkboxes(image):
    """
    Detects squares (potential checkboxes).
    Returns list of bounding boxes and whether they appear checked.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if 20 < w < 80 and 20 < h < 80:  # checkbox size range
            roi = thresh[y:y+h, x:x+w]
            filled = (roi > 0).sum() / (roi.size)

            checked = filled > 0.25  # threshold for "checked"

            boxes.append({
                "box": (int(x), int(y), int(w), int(h)),
                "checked": bool(checked),
                "fill_ratio": float(filled)
            })

    return boxes
