# Digitup Technical Test – Intelligent Document Analysis  
**Candidate : Dalia Manel Akkouchi**

Ce dépôt contient une application d’analyse intelligente de documents administratifs, intégrant OCR (ar/en), analyse de signature, détection de photo d'identité, cases cochées, fusion multimodale et interface de démonstration.

---

## Fonctionnalités principales

### 1.  **OCR Multimodal (EasyOCR)**
- Reconnaissance du texte imprimé et manuscrit.
- Support **arabe + français**.
- Prétraitement automatique (binarisation, netteté, correction lumière).
- Retour du texte + score de confiance moyen.

### 2. **Analyse de signature**
- Détection de la zone de signature (heuristique : bas du document).
- Détection d’encre via seuil adaptatif.
- Score d’encre permettant un examen (anti-fraude basique).

### 3. **Détection de photo d'identité**
- Détection d’une zone probable contenant un visage.
- Basé sur Haar Cascades ou heuristique selon la version.
- Retour : `photo_found = True / False`.

### 4. **Reconnaissance de cases cochées**
- Détection simple de cases carrées.
- Calcul du `fill_ratio` (proportion d’encre).
- Retour des cases avec : position + taux de remplissage.

### 5. **Fusion multimodale**
Un module combine tous les résultats :
- texte OCR + confiance  
- présence signature + score  
- photo d'identité  
- cases cochées  
- score global (pondéré)

### 6. **Interface de démonstration**
Interface Streamlit :
- Upload PDF / JPG / PNG
- Visualisation annotée du document
- Rapport de fiabilité

---

## 📁 Structure du projet

```
digitup-test-manel/
│
├── app/
│ ├── app.py # Interface principale
│ ├── ocr.py # OCR EasyOCR
│ ├── signature.py # Analyse signature
│ ├── face_detector.py # Détection photo identité
│ ├── checkbox.py # Détection cases cochées
│ └── fusion.py # Fusion résultats multimodaux
│
├── src/
│  └── pipeline.py # Pipeline global
│
├── notebooks/
│  └── digitup-experiments-ipynb # Notebook du projet
│
├── requirements.txt
└── README.md 
```

2) Installer les dépendances
 ```
pip install -r requirements.txt
```
(Optionnel mais conseillé)

Installer torch compatible GPU :

pip install torch

Exécution de l'application
```
streamlit run app/app.py
```

## Architecture technique

Modulaire : chaque composant peut être remplacé ou amélioré indépendamment.

Pipeline clair dans pipeline.py.

Extensions faciles :

remplacer EasyOCR par un modèle Transformer

ajouter un module de détection de falsification

connecter une base de données

## Pistes d’amélioration (listes pour l’examinateur)

Fine-tuning EasyOCR pour documents administratifs algériens

Post-traitement linguistique avec un modèle LM arabophone

Détection de falsification basée sur texture / SIAMESE

Amélioration de la localisation des signatures par segmentation U-Net

Génération automatique de datasets synthétiques pour validation
