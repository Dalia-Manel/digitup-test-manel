# photo.py
import cv2

def detect_photo(image):
    """
    Detect face using Haarcascade (fast + simple).
    Returns:
    - face_found: bool
    - (x, y, w, h)
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    if len(faces) == 0:
        return False, None

    return True, faces[0]  # first face
