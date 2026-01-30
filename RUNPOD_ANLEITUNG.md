# 🚀 Runpod Training Anleitung - RTX 5090

## Voraussetzungen
- Runpod Account mit Guthaben
- Dein Projekt als ZIP oder auf GitHub

---

## Schritt 1: Pod erstellen

1. Gehe zu [runpod.io](https://runpod.io) → **Pods** → **+ Deploy**
2. Wähle **RTX 5090** ($0.76/hr spot oder $0.89/hr on-demand)
3. Template: **RunPod Pytorch 2.8** auswählen
4. Disk Size: **50 GB** (für Dataset + Model)
5. Klicke **Deploy**

---

## Schritt 2: Verbinden

1. Warte bis Pod "Running" zeigt
2. Klicke auf **Connect** → **Start Web Terminal** oder **SSH**

---

## Schritt 3: Projekt hochladen

### Option A: GitHub (empfohlen)
```bash
cd /workspace
git clone https://github.com/DEIN_USERNAME/Inpainting.git
cd Inpainting
```

### Option B: ZIP Upload
```bash
cd /workspace
# Upload ZIP über Runpod File Browser oder:
# wget https://deine-url.com/inpainting.zip
unzip inpainting.zip
cd Inpainting
```

### Option C: SCP von lokalem PC
```bash
# Auf deinem Windows PC (PowerShell):
scp -P PORT -r "C:\Niklas\Schule\5.Klasse\DSAI\Inpainting" root@IP:/workspace/
```
(PORT und IP findest du unter "Connect" → "SSH over exposed TCP")

---

## Schritt 4: Dataset hochladen

Dein `data` Ordner muss diese Struktur haben:
```
/workspace/Inpainting/
├── data/
│   ├── dataset/           # Training Bilder
│   │   └── *.jpg
│   └── challenge_testset.npz
├── src/
│   ├── main.py
│   ├── train.py
│   ├── architecture.py
│   ├── test_setup.py      # NEU: Test-Script
│   └── ...
└── results/
```

Falls Dataset separat:
```bash
cd /workspace/Inpainting
mkdir -p data/dataset
# Upload deine Bilder in data/dataset/
```

---

## Schritt 5: Dependencies installieren

```bash
cd /workspace/Inpainting
pip install pillow matplotlib tqdm
```

(PyTorch, torchvision sind im Template bereits installiert)

---

## Schritt 6: Setup testen (WICHTIG!)

```bash
cd /workspace/Inpainting/src
python test_setup.py
```

Dies prüft:
- ✅ Alle Imports funktionieren
- ✅ CUDA verfügbar
- ✅ Model kann erstellt werden
- ✅ Forward pass funktioniert
- ✅ Loss function funktioniert
- ✅ Dataset gefunden

**Nur wenn alle Tests bestanden: Weiter zu Schritt 7!**

---

## Schritt 7: Training starten

```bash
cd /workspace/Inpainting/src
python main.py
```

### Mit tmux (empfohlen - läuft weiter wenn du disconnectest):
```bash
tmux new -s training
cd /workspace/Inpainting/src
python main.py

# Zum Detachen: Ctrl+B, dann D
# Zum Reconnecten: tmux attach -t training
```

### Mit nohup (Alternative):
```bash
cd /workspace/Inpainting/src
nohup python main.py > training.log 2>&1 &
tail -f training.log  # Log anschauen
```

---

## Schritt 8: Fortschritt beobachten

Training Output zeigt:
```
Step  1000/100000 | Loss: 0.025432 | LR: 1.50e-03 | 15.2 it/s | ETA: 108min
```

- **Loss**: Sollte sinken (gut wenn < 0.01)
- **it/s**: Iterations pro Sekunde (~15-25 auf RTX 5090)
- **ETA**: Geschätzte verbleibende Zeit

---

## Schritt 9: Ergebnisse herunterladen

Nach Training findest du:
- `results/best_model.pt` - Trainiertes Modell
- `results/testset/my_submission.npz` - Predictions
- `results/plots/` - Visualisierungen

### Download via SCP:
```bash
# Auf deinem Windows PC (PowerShell):
scp -P PORT root@IP:/workspace/Inpainting/results/best_model.pt .
scp -P PORT root@IP:/workspace/Inpainting/results/testset/my_submission.npz .
```

### Oder über Runpod File Browser

---

## 📊 Aktuelle Konfiguration

| Parameter | Wert |
|-----------|------|
| `base_channels` | 64 |
| `batchsize` | 64 |
| `n_updates` | 100.000 |
| `learningrate` | 1e-3 (max: 1e-2) |
| `num_workers` | 12 |
| `use_perceptual_loss` | True |
| Model Parameter | ~77M |

---

## 💡 Tipps

### GPU Auslastung checken:
```bash
watch -n 1 nvidia-smi
```
Sollte ~90-100% GPU-Util zeigen.

### Falls Out of Memory:
In `main.py` reduzieren:
- `batchsize`: 64 → 48 oder 32
- `base_channels`: 64 → 48

### Kosten sparen:
- Spot Instance nutzen ($0.76 statt $0.89)
- Pod stoppen wenn nicht trainiert wird
- Training mit tmux, dann Pod in "Stop" setzen (Daten bleiben)

---

## ⏱️ Erwartete Trainingszeit

| Updates | RTX 5090 | Kosten (Spot) |
|---------|----------|---------------|
| 50.000  | ~55 min  | ~$0.70 |
| 100.000 | ~110 min | ~$1.40 |

---

## 🎯 Erwartete Ergebnisse

Mit der aktuellen Konfiguration:
- **Validation RMSE**: 12-16 (sehr gut)
- **Test RMSE**: 13-17

Je niedriger der RMSE, desto besser!

---

## 🔧 Troubleshooting

### "CUDA out of memory"
→ Batch size reduzieren: `batchsize = 48` oder `32`

### "No module named 'datasets'"
→ Du bist nicht im src/ Ordner: `cd /workspace/Inpainting/src`

### Training sehr langsam (<5 it/s)
→ Check mit `nvidia-smi` ob GPU genutzt wird
→ Dataset auf langsamer Disk? Auf /workspace kopieren

### Connection lost während Training
→ Mit tmux reconnecten: `tmux attach -t training`

### Test-Script zeigt Fehler
→ Fehler genau lesen und beheben bevor Training gestartet wird!
