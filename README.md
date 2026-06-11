# PROJEKAT IS — Detekcija pospanosti (YOLOv8s)

Prepoznavanje pospanosti vozača u realnom vremenu pomoću YOLOv8s modela.

## Podaci

Kompletan dataset i modeli:  
https://drive.google.com/drive/folders/17jcK3s7X24LD4_K3c_GMdj2CXIIzzHDL?usp=sharing

## Instalacija (uv)

```bash
git clone https://github.com/nikolalukicbn-a11y/PROJEKAT_IS_Prepoznavanje_pospanosti_vozaca_u_realnom_vremenu.git
cd PROJEKAT_IS_Prepoznavanje_pospanosti_vozaca_u_realnom_vremenu
uv sync
```

## Pokretanje

Skripte se pokreću redom:

| # | Skripta | Opis |
|---|---------|------|
| 1 | `01_podaci.py` | Raspakivanje tara, podela 80/20, padding 640x640 |
| 2 | `02_trening.py` | YOLOv8s trening (150 epoha, mixup, copy-paste) |
| 3 | `03_testiranje.py` | Brzina inference-a, grafici metrika, test sa agnostic NMS |
| 4 | `04_epohe.py` | Ground Truth vs predikcija po epohama |
| 5 | `05_cuvanje_modela.py` | Export najboljeg modela (ONNX, TorchScript) |

```bash
python 01_podaci.py
python 02_trening.py
python 03_testiranje.py
python 04_epohe.py
python 05_cuvanje_modela.py
```

## Klase

| ID | Naziv |
|----|-------|
| 0 | Pospan |
| 1 | Budan |

## Parametri testiranja

```python
model = YOLO('runs/detect/drowsiness_v8s_improved/weights/best.pt')
results = model('slika.jpg', conf=0.55, agnostic_nms=True)
```

## Autor

Nikola Lukić, Marija Blagojević, Daria Obradović — Jun 2026
