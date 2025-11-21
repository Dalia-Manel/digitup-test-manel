import streamlit as st
import json
from PIL import Image, ImageDraw
import numpy as np
import io
import cv2

# Imports des modules externes (à implémenter séparément)
try:
    from ocr import extract_text
    from signature import detect_signature_zone, check_signature_presence
    from face_detector import detect_photo
    from checkbox import detect_checkboxes
    from fusion import fuse_results
    from pdf2image import convert_from_bytes
except ImportError as e:
    st.error(f"⚠️ Erreur d'import : {e}")
    st.info("Assurez-vous d'avoir installé les dépendances : pdf2image, pillow, opencv-python, streamlit")


# Configuration de la page
st.set_page_config(
    page_title="Analyseur de Documents Administratifs",
    page_icon="📄",
    layout="wide"
)


def convert_pdf_to_image(pdf_bytes):
    """
    Convertit la première page d'un PDF en image PIL
    
    Args:
        pdf_bytes: Contenu du PDF en bytes
    
    Returns:
        PIL.Image: Première page du PDF convertie en image
    """
    try:
        images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1)
        return images[0] if images else None
    except Exception as e:
        st.error(f"Erreur lors de la conversion PDF : {e}")
        return None


def load_image(uploaded_file):
    """
    Charge un fichier uploadé et le convertit en image PIL
    
    Args:
        uploaded_file: Fichier uploadé via Streamlit
    
    Returns:
        PIL.Image: Image chargée
    """
    try:
        file_bytes = uploaded_file.read()
        
        # Si c'est un PDF, convertir en image
        if uploaded_file.type == "application/pdf":
            st.info("📄 Conversion de la première page du PDF en cours...")
            return convert_pdf_to_image(file_bytes)
        else:
            # Sinon, charger directement l'image
            return Image.open(io.BytesIO(file_bytes))
    except Exception as e:
        st.error(f"Erreur lors du chargement de l'image : {e}")
        return None


def annotate_image(image, signature_zones, photo_zone, checkboxes):
    """
    Annote l'image avec des rectangles pour les zones détectées
    
    Args:
        image: Image PIL à annoter
        signature_zones: Liste de zones de signature [(x, y, w, h), ...]
        photo_zone: Zone de photo (x, y, w, h) ou None
        checkboxes: Liste de cases cochées [(x, y, w, h, checked), ...]
    
    Returns:
        PIL.Image: Image annotée
    """
    # Créer une copie de l'image
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    
    # Annoter les signatures en rouge
    if signature_zones:
        for zone in signature_zones:
            if zone and len(zone) >= 4:
                x, y, w, h = zone[:4]
                draw.rectangle([x, y, x+w, y+h], outline="red", width=3)
                draw.text((x, y-20), "Signature", fill="red")
    
    # Annoter la photo en bleu
    if photo_zone:
        x, y, w, h = photo_zone[:4]
        draw.rectangle([x, y, x+w, y+h], outline="blue", width=3)
        draw.text((x, y-20), "Photo", fill="blue")
    
    # Annoter les cases cochées en vert (cochées) ou orange (non cochées)
    if checkboxes:
        for i, checkbox in enumerate(checkboxes):
            if len(checkbox) >= 5:
                x, y, w, h, checked = checkbox[:5]
                color = "green" if checked else "orange"
                draw.rectangle([x, y, x+w, y+h], outline=color, width=2)
                label = f"Case {i+1}: {'✓' if checked else '☐'}"
                draw.text((x, y-20), label, fill=color)
    
    return annotated


def analyze_document(image):
    """
    Analyse le document en appelant tous les modules de détection
    
    Args:
        image: Image PIL du document
    
    Returns:
        dict: Dictionnaire contenant tous les résultats d'analyse
    """
    results = {
        "text": None,
        "signature_zones": [],
        "signature_present": False,
        "photo_detected": False,
        "photo_zone": None,
        "checkboxes": [],
        "global_score": None,
        "anomalies": [],
        "errors": []
    }
    
    # Convertir l'image PIL en format compatible (numpy array)
    img_array = np.array(image)
    
    try:
        # 1. Extraction du texte OCR
        st.info("🔍 Extraction du texte...")
        results["text"] = extract_text(img_array)
    except Exception as e:
        results["errors"].append(f"Erreur OCR : {str(e)}")
    
    try:
        # 2. Détection des zones de signature
        st.info("✍️ Détection des signatures...")
        results["signature_zones"] = detect_signature_zone(img_array)
        results["signature_present"] = check_signature_presence(img_array, results["signature_zones"])
    except Exception as e:
        results["errors"].append(f"Erreur détection signature : {str(e)}")
    
    try:
        # 3. Détection de la photo d'identité
        st.info("📸 Détection de la photo d'identité...")
        photo_result = detect_photo(img_array)
        if isinstance(photo_result, dict):
            results["photo_detected"] = photo_result.get("detected", False)
            results["photo_zone"] = photo_result.get("zone", None)
        else:
            results["photo_detected"] = photo_result
    except Exception as e:
        results["errors"].append(f"Erreur détection photo : {str(e)}")
    
    try:
        # 4. Détection des cases cochées
        st.info("☑️ Détection des cases cochées...")
        results["checkboxes"] = detect_checkboxes(img_array)
    except Exception as e:
        results["errors"].append(f"Erreur détection cases : {str(e)}")
    
    try:
        # 5. Fusion des résultats et calcul du score global
        st.info("⚙️ Fusion des résultats...")
        fusion_result = fuse_results(results)
        if isinstance(fusion_result, dict):
            results["global_score"] = fusion_result.get("score", None)
            results["anomalies"] = fusion_result.get("anomalies", [])
        else:
            results["global_score"] = fusion_result
    except Exception as e:
        results["errors"].append(f"Erreur fusion : {str(e)}")
    
    return results


def display_results(results, annotated_image):
    """
    Affiche les résultats de l'analyse dans l'interface
    
    Args:
        results: Dictionnaire des résultats
        annotated_image: Image annotée avec les détections
    """
    st.header("📊 Résultats de l'analyse")
    
    # Afficher les erreurs s'il y en a
    if results["errors"]:
        st.error("⚠️ Erreurs rencontrées :")
        for error in results["errors"]:
            st.write(f"- {error}")
    
    # Créer des colonnes pour l'affichage
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Image annotée")
        st.image(annotated_image, use_container_width=True)
    
    with col2:
        # Score global
        st.subheader("🎯 Score global")
        if results["global_score"] is not None:
            score_value = results["global_score"]
            if isinstance(score_value, (int, float)):
                st.metric("Score de validité", f"{score_value:.1f}%")
                # Barre de progression
                st.progress(score_value / 100)
            else:
                st.write(score_value)
        else:
            st.warning("Score non disponible")
        
        # Anomalies
        if results["anomalies"]:
            st.subheader("⚠️ Anomalies détectées")
            for anomaly in results["anomalies"]:
                st.warning(f"• {anomaly}")
    
    # Résultats détaillés
    st.subheader("📋 Résultats détaillés")
    
    tabs = st.tabs(["📝 Texte OCR", "✍️ Signatures", "📸 Photo", "☑️ Cases cochées"])
    
    with tabs[0]:
        st.write("**Texte extrait du document :**")
        if results["text"]:
            st.text_area("Contenu", results["text"], height=200)
        else:
            st.info("Aucun texte extrait")
    
    with tabs[1]:
        st.write("**Détection de signatures :**")
        st.write(f"✓ Signature présente : **{'Oui' if results['signature_present'] else 'Non'}**")
        if results["signature_zones"]:
            st.write(f"Nombre de zones détectées : {len(results['signature_zones'])}")
            for i, zone in enumerate(results["signature_zones"]):
                with st.expander(f"Zone {i+1}"):
                    st.json(zone)
        else:
            st.info("Aucune zone de signature détectée")
    
    with tabs[2]:
        st.write("**Détection de photo d'identité :**")
        st.write(f"✓ Photo détectée : **{'Oui' if results['photo_detected'] else 'Non'}**")
        if results["photo_zone"]:
            st.json({"zone": results["photo_zone"]})
    
    with tabs[3]:
        st.write("**Cases cochées détectées :**")
        if results["checkboxes"]:
            st.write(f"Nombre de cases trouvées : {len(results['checkboxes'])}")
            checked_count = sum(1 for cb in results["checkboxes"] if len(cb) >= 5 and cb[4])
            st.write(f"Cases cochées : {checked_count}/{len(results['checkboxes'])}")
            
            for i, checkbox in enumerate(results["checkboxes"]):
                with st.expander(f"Case {i+1}"):
                    st.json(checkbox)
        else:
            st.info("Aucune case cochée détectée")
    
    # JSON structuré
    with st.expander("🔍 Voir le JSON complet"):
        # Préparer le JSON en retirant les erreurs pour l'affichage
        display_json = {k: v for k, v in results.items() if k != "errors"}
        st.json(display_json)


def generate_report(results, filename):
    """
    Génère un rapport textuel de l'analyse
    
    Args:
        results: Dictionnaire des résultats
        filename: Nom du fichier analysé
    
    Returns:
        str: Rapport formaté
    """
    report = f"""
# 📄 RAPPORT D'ANALYSE DE DOCUMENT
## Document analysé : {filename}
---

### ✅ VALIDATIONS

• **Signature** : {'✓ Présente' if results['signature_present'] else '✗ Absente'}
• **Photo d'identité** : {'✓ Détectée' if results['photo_detected'] else '✗ Non détectée'}
• **Cases cochées** : {len(results['checkboxes'])} case(s) détectée(s)

---

### 🎯 SCORES

• **Score global** : {results['global_score'] if results['global_score'] is not None else 'Non disponible'}

---

### ⚠️ ANOMALIES
"""
    
    if results["anomalies"]:
        for anomaly in results["anomalies"]:
            report += f"\n• {anomaly}"
    else:
        report += "\nAucune anomalie détectée ✓"
    
    report += "\n\n---\n\n### 📊 STATISTIQUES\n"
    report += f"\n• Zones de signature détectées : {len(results['signature_zones'])}"
    report += f"\n• Longueur du texte extrait : {len(results['text']) if results['text'] else 0} caractères"
    
    if results["checkboxes"]:
        checked = sum(1 for cb in results["checkboxes"] if len(cb) >= 5 and cb[4])
        report += f"\n• Cases cochées : {checked}/{len(results['checkboxes'])}"
    
    return report


def main():
    """
    Fonction principale de l'application Streamlit
    """
    # En-tête
    st.title(" Analyseur Intelligent de Documents Administratifs")
    st.markdown("---")
    
    # Barre latérale d'aide
    with st.sidebar:
        st.header("ℹ️ Guide d'utilisation")
        st.markdown("""
        **1. Uploadez votre document**
        - Formats acceptés : PDF, JPG, PNG
        - Le PDF sera automatiquement converti
        
        **2. Analyse automatique**
        - Extraction du texte (OCR)
        - Détection des signatures
        - Détection de la photo d'identité
        - Détection des cases cochées
        
        **3. Résultats**
        - Visualisation annotée
        - Score de validité
        - Rapport détaillé
        """)
        
        st.markdown("---")
        st.info("💡 **Astuce** : Les zones détectées sont annotées en couleur sur l'image")
    
    # Upload de fichier
    st.subheader("📤 Étape 1 : Charger un document")
    uploaded_file = st.file_uploader(
        "Choisissez un fichier (PDF, JPG, PNG)",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Formats acceptés : PDF (première page), JPEG, PNG"
    )
    
    if uploaded_file is not None:
        # Afficher les informations du fichier
        st.success(f"✓ Fichier chargé : **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        
        # Charger et afficher l'image
        with st.spinner("Chargement de l'image..."):
            image = load_image(uploaded_file)
        
        if image is None:
            st.error("Impossible de charger l'image. Vérifiez le format du fichier.")
            return
        
        st.subheader("📷 Document chargé")
        st.image(image, caption="Image originale", use_container_width=True)
        
        # Bouton d'analyse
        st.markdown("---")
        st.subheader("🔬 Étape 2 : Analyser le document")
        
        if st.button(" Lancer l'analyse", type="primary", use_container_width=True):
            with st.spinner("Analyse en cours... Cela peut prendre quelques secondes."):
                # Analyser le document
                results = analyze_document(image)
                
                # Annoter l'image
                annotated_image = annotate_image(
                    image,
                    results["signature_zones"],
                    results["photo_zone"],
                    results["checkboxes"]
                )
                
                # Sauvegarder les résultats dans la session
                st.session_state.results = results
                st.session_state.annotated_image = annotated_image
                st.session_state.filename = uploaded_file.name
            
            st.success("✅ Analyse terminée !")
        
        # Afficher les résultats si disponibles
        if "results" in st.session_state:
            st.markdown("---")
            display_results(st.session_state.results, st.session_state.annotated_image)
            
            # Bouton de génération de rapport
            st.markdown("---")
            st.subheader("📝 Étape 3 : Générer le rapport")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                if st.button("📄 Générer le rapport", use_container_width=True):
                    report = generate_report(st.session_state.results, st.session_state.filename)
                    st.session_state.report = report
            
            # Afficher le rapport s'il a été généré
            if "report" in st.session_state:
                st.markdown("---")
                st.subheader("📋 Rapport d'analyse")
                st.markdown(st.session_state.report)
                
                # Bouton de téléchargement
                st.download_button(
                    label="💾 Télécharger le rapport",
                    data=st.session_state.report,
                    file_name=f"rapport_{st.session_state.filename}.txt",
                    mime="text/plain"
                )
    
    else:
        # Message d'accueil
        st.info("👆 Commencez par uploader un document administratif pour l'analyser")
        
        # Exemple d'utilisation
        with st.expander("💡 Exemple de cas d'usage"):
            st.markdown("""
            **Documents compatibles :**
            - Formulaires administratifs
            - Contrats signés
            - Dossiers d'inscription
            - Documents d'identité
            - Questionnaires
            
            **Détections automatiques :**
            - ✍️ Signatures manuscrites
            - 📸 Photos d'identité
            - ☑️ Cases cochées/non cochées
            - 📝 Texte intégral (OCR)
            """)


if __name__ == "__main__":
    main()