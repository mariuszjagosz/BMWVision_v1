# BMWVISION_V1

System wizualnej nawigacji oparty na analizie segmentacji obrazu i (opcjonalnie) danych GPS, służący do detekcji drogi, wizualizacji trajektorii oraz kierunku ruchu pojazdu.

## Opis projektu

Celem projektu jest stworzenie systemu, który:
- Wykrywa drogę na nagraniu z kamery samochodowej przy użyciu segmentacji obrazu (model **TwinLiteNetPlus**),
- Wyznacza trajektorię jazdy na podstawie wykrytej maski drogi (aktualnie bez GPS),
- Nakłada na każdą klatkę wideo strzałkę reprezentującą kierunek ruchu ,
- Generuje końcowy film `.mp4` z naniesioną wizualizacją,
- Docelowo: integruje również dane GPS i heading do rysowania rzeczywistej trajektorii.

##  Struktura katalogów

```
BMWVISION_V1/
│
├── data/                     # dane wejściowe – wideo i CSV
├── models/
│   └── twinlite/             # Model TwinLiteNetPlus i jego konfiguracja
│
├── output/                   # Katalog na wynikowe ramki lub dane pośrednie
├── outputs/                  # Wygenerowane filmy z wizualizacją
│   ├── output_video.mp4
│   └── output_video_example.mp4
│
├── src/
│   ├── segment_utils.py      # Segmentacja drogi – obsługa modelu TwinLiteNetPlus
│   ├── video_segment.py      # Główne przetwarzanie wideo i rysowanie trajektorii
│   ├── sync_gps_video.py     # Wstępna synchronizacja GPS z wideo (przygotowane)
│   ├── frame.jpg             # Przykładowa ramka (debug - testowanie segment_utils)
│
├── README.md
└── .gitignore
```

## Wymagania

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- Pandas

Instalacja zależności:

```bash
pip install -r requirements.txt
```

### Użycie pretrenowanych modeli
Pretrenowany model można pobrać z linku: [Modele](https://drive.google.com/drive/folders/1EqBzUw0b17aEumZmWYrGZmbx_XJqU-vz) i umieścić w katalogu models/twinlite/model/pretrained
##  Uruchomienie

1. Umieść wideo do analizy w katalogu `data/`.
2. Upewnij się, że `config.yaml` (jeśli używany) wskazuje na poprawne ścieżki.
3. Uruchom główny skrypt:

```bash
python src/video_segment.py
```

Wynikowy film zostanie zapisany w katalogu `outputs/`.

##  Planowane funkcjonalności (TODO)

- Synchronizacja i interpolacja danych GPS względem klatek wideo
- Rysowanie rzeczywistej trajektorii jazdy z użyciem współrzędnych GPS
- Mapowanie danych geograficznych na układ lokalny XY
-  Automatyczne porównanie wykrytej trajektorii z GPS (walidacja)

## 👤 Autorzy

Mariusz Jagosz, Bartosz Bieniek, Dawid Klimek

Segmentacja została oparta na modelu TwinLiteNetPlus autorstwa Quang-Huy Che et al., udostępnionym na licencji MIT.
Repozytorium: https://github.com/chequanghuy/TwinLiteNetPlus
Zobacz publikację: https://arxiv.org/abs/2403.16958
