# === PROGRAMME DE FORMATION YOLOv8 LOCAL (Spyder) ===
# Projet PIC - Année 2024-2025

import os
import zipfile
from ultralytics import YOLO
from IPython.display import Image

# === Chemins de base ===
HOME = "C:/Users/anleb/Desktop/Detection_PIC_2024-2025"
DATASET_ZIP = os.path.join(HOME, "roboflow.zip")
DATASET_DIR = os.path.join(HOME, "dataset")
DATA_YAML = os.path.join(HOME, "data", "data.yaml")
MODEL_PATH = os.path.join(HOME, "./weights/yolov8s.pt")

# === Extraction du dataset si non fait ===
if os.path.exists(DATASET_ZIP) and not os.path.exists(DATASET_DIR):
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)
    print("Dataset extrait avec succès.")
else:
    print("Dataset déjà extrait ou introuvable.")

# === Suppression des anciens résultats ===
results_path = os.path.join(HOME, "runs", "detect", "train")
if os.path.exists(results_path):
    import shutil
    shutil.rmtree(results_path)
    print("Ancien dossier de résultats supprimé.")

# === Entraînement du modèle ===
model = YOLO(MODEL_PATH)
model.train(
    data=DATA_YAML,
    epochs=1,
    imgsz=800,
    conf=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    mixup=0.6,
    plots=True
)

# === Affichage des résultats ===
Image(filename=os.path.join(HOME, "runs/detect/train/confusion_matrix_normalized.png"), width=800)
Image(filename=os.path.join(HOME, "runs/detect/train/PR_curve.png"), width=800)
Image(filename=os.path.join(HOME, "runs/detect/train/results.png"), width=800)
