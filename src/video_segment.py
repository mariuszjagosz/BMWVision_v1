import cv2
import torch
from pathlib import Path
from src.segment_utils import Segmenter
import numpy as np
import os

# ==========================
# Konfiguracja
# ==========================
VIDEO_PATH = Path("data/nagrania/przejazd1/example.mp4")
WEIGHTS_PATH = Path("models/twinlite/pretrained/large.pth")
OUTPUT_PATH = Path("outputs/output_video_example.mp4")
IMG_SIZE = 640

# ==========================
# Przygotowanie output folderu
# ==========================
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ==========================
# Inicjalizacja segmentera
# ==========================
segmenter = Segmenter(
    weight_path=WEIGHTS_PATH,
    config_name="large",
    img_size=IMG_SIZE
)

# ==========================
# Przetwarzanie wideo
# ==========================
cap = cv2.VideoCapture(str(VIDEO_PATH))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_video = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, fps, (width, height))

frame_idx = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    try:
        da_mask, ll_mask = segmenter.predict(frame)

        # Bardzo jasny zielony z dużą saturacją i mniejszą przezroczystością
        green_overlay = np.zeros_like(frame, dtype=np.uint8)
        green_overlay[da_mask == 1] = (0, 255, 0)  # nasycony jasnozielony

        blended = cv2.addWeighted(green_overlay, 0.45, frame, 0.55, 0)

        # Czerwona przezroczysta maska - lane lines
        red_overlay = np.zeros_like(frame)
        red_overlay[ll_mask == 1] = (0, 0, 255)
        blended = cv2.addWeighted(red_overlay, 0.3, blended, 0.7, 0)

        out_video.write(blended)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Przetworzono {frame_idx} klatek")

    except Exception as e:
        print(f"Błąd w klatce {frame_idx}: {e}")
        out_video.write(frame)

cap.release()
out_video.release()
print(f"Zapisano wynik do {OUTPUT_PATH}")
