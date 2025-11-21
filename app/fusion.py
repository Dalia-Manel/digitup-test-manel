# fusion.py
def fuse_results(
    ocr_text="",
    ocr_conf=0.0,
    signature_present=False,
    signature_score=0.0,
    photo_found=False,
    checkboxes=None
):
    """
    Combine OCR, signature, photo, and checkbox results into a structured dictionary.
    """

    # Sécurise checkboxes
    if checkboxes is None:
        checkboxes = []

    score_components = []

    # Normalisation confiance OCR (si ocr_conf est sur 100)
    score_components.append(ocr_conf / 100 if ocr_conf is not None else 0)

    # Score signature
    score_components.append(signature_score if signature_score is not None else 0)

    # Photo détectée
    score_components.append(1.0 if photo_found else 0.0)

    # Moyenne du ratio des cases cochées
    if len(checkboxes) > 0:
        avg_check = sum(b.get("fill_ratio", 0) for b in checkboxes) / len(checkboxes)
        score_components.append(avg_check)
    else:
        score_components.append(0.0)

    # Calcul score global sécurisé
    global_score = sum(score_components) / len(score_components) if score_components else 0

    return {
        "text": ocr_text,
        "ocr_confidence": ocr_conf,
        "signature_present": signature_present,
        "signature_score": signature_score,
        "photo_found": photo_found,
        "checkboxes": checkboxes,
        "global_score": float(global_score)
    }
