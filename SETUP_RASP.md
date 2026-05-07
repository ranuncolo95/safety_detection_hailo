# Setup deployment Raspberry Pi 5 + Hailo-8L

Pipeline doppia: **YOLOv8m-seg** (segmentazione persone) + **SCDepthV3** (depth estimation calibrata in metri) con classificazione DANGER/WARNING/SAFE basata sulla depth ai piedi.

## Struttura file richiesta

```
progetto_finale_rasp/
├── inferenza_hailo_dual.py        # ← script principale (esecuzione)
├── depth_calibrato.py              # modulo DepthCalibrator (importato dallo script)
├── calib_params.json               # parametri calibrazione (dal PC host)
│
├── models/
│   ├── yolov8m_seg.hef             # da Hailo Model Zoo (Hailo-8L)
│   └── scdepthv3.hef               # da Hailo Model Zoo (Hailo-8L)
│
├── video_inferenza/
│   └── video_1.mp4                 # video di test
│
└── output/                         # video processati (creata automaticamente)
    └── output_hailo_dual.mp4
```

## File da copiare dal PC host

Dalla cartella di sviluppo del PC con GPU sposta sul Pi:

| File | Sorgente | Destinazione Pi |
|---|---|---|
| `calib_params.json` | `calibration/calib_params.json` | `./calib_params.json` |
| `depth_calibrato.py` | (creato durante calibrazione) | `./depth_calibrato.py` |
| `inferenza_hailo_dual.py` | (questo nuovo file) | `./inferenza_hailo_dual.py` |

## File da scaricare dal Hailo Model Zoo

Dopo registrazione su [hailo.ai/developer-zone](https://hailo.ai/developer-zone/), scarica per **Hailo-8L** (13 TOPS):

- `yolov8m_seg.hef`  → in `./models/`
- `scdepthv3.hef`    → in `./models/`

## Setup ambiente sul Pi

```bash
# 1. Hailo runtime (se non già fatto)
sudo apt install hailo-all
hailortcli fw-control identify   # verifica device

# 2. Python deps
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy
# hailo_platform è già installato nel sistema con hailo-all

# 3. Test del modulo calibrazione
python3 depth_calibrato.py calib_params.json
# Deve stampare la formula e un test su input simulato
```

## Esecuzione

### Run base con video

```bash
python3 inferenza_hailo_dual.py \
    --hef-yolo  ./models/yolov8m_seg.hef \
    --hef-depth ./models/scdepthv3.hef \
    --calib     ./calib_params.json \
    --video     ./video_inferenza/video_1.mp4 \
    --output    ./output/output_hailo_dual.mp4 \
    --display
```

### Soglie distanza personalizzate

```bash
# DANGER se persona < 2m, WARNING se < 4m
python3 inferenza_hailo_dual.py \
    --danger-dist 2.0 \
    --warning-dist 4.0 \
    --video ./video_inferenza/video_1.mp4
```

### Debug primo run (verifica shape output)

```bash
python3 inferenza_hailo_dual.py --debug-outputs --video ./video_inferenza/video_1.mp4
```

Stampa al primo frame le dimensioni di tutti i tensori output di YOLO e SCDepthV3. Importante per verificare che il decoder li interpreti correttamente.

## Logica classificazione (cosa cambia rispetto al vecchio script)

**Prima** (zone fisse pixel):
- Rettangolo `(x1, y1, x2, y2)` calcolato da `DANGER_W_FRAC` / `DANGER_H_FRAC`
- Persona "in pericolo" se la sua bbox interseca il rettangolo

**Adesso** (depth metrica):
- Per ogni persona viene presa la mask del 20% inferiore (i piedi)
- Si calcola la **depth mediana ai piedi** dalla calibrazione SCDepthV3
- Tre livelli:
  - `DANGER`  se mediana ≤ `--danger-dist`  (default 3 m) → rosso pieno + alert
  - `WARNING` se mediana ≤ `--warning-dist` (default 5 m) → giallo
  - `SAFE`    altrimenti → verde

Questo elimina il problema delle zone fisse che non si adattano alla scena.

## Note prestazionali

Su Hailo-8L, l'attivazione di un network group costa ~10-20 ms. Lo script attiva YOLO e SCDepth alternativamente per ogni frame, quindi c'è overhead. Atteso:

| Componente | Tempo per frame |
|---|---|
| Activation YOLO + inferenza | ~30-40 ms |
| Activation SCDepth + inferenza | ~15-25 ms |
| Decode YOLO (CPU) | ~20-40 ms |
| Postprocess depth + calibrazione (CPU) | ~5-10 ms |
| **Totale stimato** | **~70-115 ms** (≈9-14 fps) |

Per ottimizzazioni successive: usare `InferModel` API (HailoRT 4.17+) o lo scheduler Hailo per ridurre overhead di switch tra modelli.

## Troubleshooting

**`RuntimeError: Output YOLO incompleti per scala N`**  
Il HEF YOLO ha output diversi da quelli attesi. Lancia con `--debug-outputs` e verifica le shape. Lo script si aspetta:
- 3 detection heads: 80×80×64 (box), 80×80×80 (cls), 80×80×32 (mask coeff) — e analoghi a 40×40 e 20×20
- 1 proto: 160×160×32

**`Output SCDepthV3 non riconosciuto`**  
Controlla con `--debug-outputs` la shape dell'output. Se è insolita (es. canali multipli), il `postprocess_scdepth` va adattato al formato esatto.

**Tutte le persone vengono classificate `safe` anche se vicine**  
- Verifica con un print che `depth_m` abbia valori sensati (controlla `min/med/max` nel banner di info)
- Se i valori sono fuori scala, ricalibra: il fit attuale ha MAE 1.2m sulla zona 0.3-10m
- Aumenta `--danger-dist` per essere più conservativi (più probabili false positive ma meno false safe)
