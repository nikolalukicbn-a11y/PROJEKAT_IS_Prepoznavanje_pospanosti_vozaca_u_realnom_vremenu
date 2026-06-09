# PROJEKAT IS — Detekcija pospanosti (YOLOv8)

## Podaci

Kompletan dataset i modeli:  
https://drive.google.com/drive/folders/17jcK3s7X24LD4_K3c_GMdj2CXIIzzHDL?usp=sharing

## Potrebne biblioteke

```bash
pip install ultralytics pyyaml torch pandas matplotlib pillow tqdm
```

## Pokretanje

1. Skini `trening+val.tar` sa Google Drive linka iznad
2. Stavi ga u root folder projekta (pored `inteligentni_finale.py`)
3. Pokreni:

```bash
python inteligentni_finale.py
```

## Struktura projekta

```
├── inteligentni_finale.py      # Glavni skript (YOLO trening + evaluacija)
├── data.yaml                   # YOLO konfiguracija (klase: pospan, budan)
├── trening+val.tar             # Dataset (400 MB — na Drive-u)
└── test_finale.zip             # Test skup
```

## Klase

| ID  | Naziv   |
|-----|---------|
| 0   | Pospan  |
| 1   | Budan   |
