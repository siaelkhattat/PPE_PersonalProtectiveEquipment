"""Démo Streamlit : détection de casque et conformité EPI (inférence CPU).

Lancer depuis la racine du projet :
    streamlit run src/app.py
"""
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

from compliance import analyse, draw

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"

st.set_page_config(page_title="Détection EPI", page_icon="⛑️", layout="wide")


@st.cache_resource
def load_model(path: str):
    model = YOLO(path)
    # échauffement pour ne pas fausser la mesure du temps
    model.predict(np.zeros((416, 416, 3), dtype=np.uint8), imgsz=416, device="cpu", verbose=False)
    return model


# ---------- Barre latérale ----------
st.sidebar.title("⚙️ Paramètres")
conf = st.sidebar.slider("Seuil de confiance", 0.05, 0.95, 0.25, 0.05,
                         help="Plus il est bas, plus le modèle détecte (au risque de fausses alertes).")
imgsz = st.sidebar.select_slider("Résolution d'inférence", options=[320, 416, 512, 640], value=416)
source = st.sidebar.radio("Source de l'image", ["Importer une image", "Webcam (photo)"])
st.sidebar.markdown("---")
st.sidebar.caption("Modèle : YOLOv8n fine-tuné\n\nClasses : helmet, head, person\n\nInférence : CPU")

# ---------- En-tête ----------
st.title("⛑️ Détection d'équipements de protection (EPI)")
st.write("Chargez une photo de chantier : le système détecte les **casques** (vert) "
         "et les **têtes sans casque** (rouge).")

if not MODEL_PATH.exists():
    st.error(f"Modèle introuvable : {MODEL_PATH}. Placez `best.pt` dans le dossier `models/`.")
    st.stop()

model = load_model(str(MODEL_PATH))

# ---------- Image d'entrée ----------
if source == "Importer une image":
    file = st.file_uploader("Image (jpg, png)", type=["jpg", "jpeg", "png", "webp"])
else:
    file = st.camera_input("Prenez une photo")

if file is None:
    st.info("👆 Ajoutez une image pour lancer la détection.")
    st.stop()

pil = Image.open(file).convert("RGB")
bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# ---------- Inférence ----------
t0 = time.perf_counter()
result = model.predict(bgr, imgsz=imgsz, conf=conf, device="cpu", verbose=False)[0]
ms = (time.perf_counter() - t0) * 1000

dets = analyse(result)
n_ok = sum(d[5] for d in dets)
n_bad = len(dets) - n_ok
annotated = draw(bgr.copy(), dets)
annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ---------- Statut global ----------
if not dets:
    st.warning("Aucune tête ni casque détecté. Essayez de baisser le seuil de confiance.")
elif n_bad:
    st.error(f"🚨 ALERTE : {n_bad} personne(s) sans casque détectée(s)")
else:
    st.success("✅ Tous les travailleurs détectés portent un casque")

# ---------- Métriques ----------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Casques (conformes)", n_ok)
c2.metric("Sans casque", n_bad)
c3.metric("Temps d'inférence", f"{ms:.0f} ms")
c4.metric("Vitesse", f"{1000 / ms:.1f} FPS")

# ---------- Images ----------
left, right = st.columns(2)
left.subheader("Image originale")
left.image(pil, use_container_width=True)
right.subheader("Résultat")
right.image(annotated, use_container_width=True)

# ---------- Détail ----------
with st.expander("Détail des détections"):
    if dets:
        rows = [{"Classe": d[4],
                 "Statut": "Conforme" if d[5] else "Non conforme",
                 "Confiance": round(d[6], 2)} for d in dets]
        st.dataframe(rows, use_container_width=True)
    else:
        st.write("Aucune détection.")

# ---------- Téléchargement ----------
ok, buf = cv2.imencode(".jpg", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
if ok:
    st.download_button("⬇️ Télécharger le résultat", buf.tobytes(),
                       file_name="resultat_epi.jpg", mime="image/jpeg")
