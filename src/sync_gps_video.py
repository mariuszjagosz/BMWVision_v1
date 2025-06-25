import pandas as pd
import cv2
import numpy as np
import yaml
from pathlib import Path

# Oblicz ścieżkę do katalogu głównego projektu (jeden poziom wyżej)
project_root = Path(__file__).resolve().parent.parent
config_path = project_root / "config.yaml"

# Wczytaj konfigurację
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

# Ścieżki z konfiguracji
video_path = project_root / config["video_path"]
gps_path = project_root / config["gps_path"]
output_path = project_root / "output" / "interpolated_gps.csv"
offset = config.get("video_start_offset", 0.0)


# Wczytaj dane GPS
gps_df = pd.read_csv(gps_path)

# Upewnij się, że dane GPS mają odpowiednie kolumny
if "seconds_elapsed" not in gps_df.columns:
    raise ValueError("Plik GPS musi zawierać kolumnę 'seconds_elapsed'")
gps_df = gps_df[["seconds_elapsed", "latitude", "longitude", "bearing"]].copy()
gps_df.rename(columns={"seconds_elapsed": "time"}, inplace=True)
gps_df.sort_values("time", inplace=True)

# Wczytaj metadane z wideo
video = cv2.VideoCapture(str(video_path))
fps = video.get(cv2.CAP_PROP_FPS)
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
video.release()

if fps == 0:
    raise ValueError("Nie udało się odczytać FPS z pliku wideo. Sprawdź ścieżkę lub format pliku.")

# Oblicz czasy klatek wideo z offsetem
frame_times = np.arange(0, frame_count) / fps + offset

# Interpolacja pozycji GPS dla każdej klatki
interpolated_data = {
    "frame_id": [],
    "timestamp": [],
    "latitude": [],
    "longitude": [],
    "bearing": []
}

for i, t in enumerate(frame_times):
    interpolated_data["frame_id"].append(i)
    interpolated_data["timestamp"].append(t)
    interpolated_data["latitude"].append(np.interp(t, gps_df["time"], gps_df["latitude"]))
    interpolated_data["longitude"].append(np.interp(t, gps_df["time"], gps_df["longitude"]))
    interpolated_data["bearing"].append(np.interp(t, gps_df["time"], gps_df["bearing"]))

# Zapisz wynik jako CSV
interpolated_df = pd.DataFrame(interpolated_data)
output_path = project_root / "output" / "interpolated_gps.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
interpolated_df.to_csv(output_path, index=False)

print(f"[✓] Zapisano interpolowane dane GPS do: {output_path}")
