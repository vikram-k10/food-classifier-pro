# Indian Food Classifier Pro 🍛

An end-to-end deep learning project: **transfer learning, rigorous evaluation, calibrated confidence,
a "not sure" reject option, Grad-CAM explainability, ONNX export, and a web app + REST API.**

It upgrades the original single-notebook CNN classifier
([AdwitiyaKhare/food-classifier](https://github.com/AdwitiyaKhare/food-classifier)) — same task and dataset
(20 Indian/street-food classes from the
[Kaggle Indian Food Classification](https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification)
dataset) — into a reproducible experiment pipeline. The code in this repository is a rewrite; the original
project is credited as the starting point.

> **Status of the numbers:** this repo ships *code*, not trained weights. Accuracy figures appear only after you
> run the training commands below on the dataset (the results table at the bottom is filled by `foodai.compare`).

---

## What makes it more than "a classifier"

| Area | What is implemented | Where |
|---|---|---|
| **Transfer learning** | ResNet18/50, EfficientNet-B0, MobileNetV3, ConvNeXt-T; two-phase fine-tuning (frozen backbone → full network), discriminative learning rates | `foodai/models.py`, `train.py` |
| **Baseline** | From-scratch VGG-style CNN with BatchNorm — the reference point for the comparison | `models.py` |
| **Regularisation** | TrivialAugment, RandomResizedCrop, RandomErasing, **MixUp + CutMix**, label smoothing, dropout, weight decay (not on norm/bias), gradient clipping | `data.py`, `train.py` |
| **Optimisation** | AdamW, linear warm-up + cosine decay, mixed precision (CUDA), early stopping, best-checkpoint selection | `train.py` |
| **Data hygiene** | Auto-detects the dataset folder, drops corrupt images and **exact duplicates** (prevents train/test leakage), stratified 70/15/15 split saved to `splits.json`, imbalance report, optional class weights | `data.py` |
| **Evaluation** | Accuracy, top-3, macro/weighted P-R-F1, per-class table, normalised confusion matrix, most-confused pairs | `evaluate.py` |
| **Calibration** | Expected Calibration Error + **temperature scaling** fitted on validation only, reliability diagram | `evaluate.py` |
| **Reject option** | Confidence threshold chosen on validation to hit ~90% accuracy on answered images; accuracy-vs-coverage curve; synthetic-input sanity check | `evaluate.py`, `predict.py` |
| **Explainability** | **Grad-CAM** heat-maps in the CLI, the evaluation report and the web UI | `gradcam.py` |
| **Experiment tracking** | One command runs all models on *identical splits* and writes a comparison table + chart; optional ablations (no pretraining, no augmentation) | `compare.py` |
| **Deployment** | Flask UI + JSON API (top-3, Grad-CAM, reject flag, kcal estimate), ONNX export verified against PyTorch, Dockerfile, GitHub Actions CI | `webapp/`, `export.py`, `Dockerfile` |
| **Quality** | pytest suite that trains a tiny model on synthetic images and exercises every module | `tests/` |

---

## Quick start

### 1. Install
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q                                                 # a few minutes on CPU: trains a tiny model on synthetic images
```

### 2. Get the data
```bash
python scripts/download_dataset.py          # needs a free Kaggle account; prints the dataset path
# or download the archive manually and unzip it anywhere — the loader finds the class folders by itself

# The Kaggle archive is ~1.6 GB of large photos. Shrink it ONCE so training isn't limited by JPEG decoding
# (much faster epochs, a few hundred MB, no accuracy loss at 224 px):
python scripts/resize_dataset.py --src <dataset-path> --out data/indian-food-256 --size 256
```
Use `--data-dir data/indian-food-256 --num-workers 4` (or more cores if you have them) in the training commands below.

### 3. Train
```bash
# the single best model (pretrained EfficientNet-B0)
python -m foodai.train --data-dir <dataset-path> --arch efficientnet_b0 --epochs 25

# the full study: baseline vs 3 pretrained models + ablations, identical splits, copies the winner to models/best.pt
python -m foodai.compare --data-dir <dataset-path> --epochs 20 --ablation --export-best
```
Every run writes to `runs/<name>/`: `best.pt`, `metrics.json`, `history.png`, `confusion_matrix.png`,
`per_class_f1.png`, `reliability_diagram.png`, `selective_prediction.png`, `gradcam_examples.png`,
`per_class_metrics.csv`, `data_report.json`, `splits.json`, `config.json`.

**No GPU?** Use Google Colab (Runtime → GPU):
```python
!git clone <your-repo-url> && cd food-classifier-pro && pip install -q -r requirements.txt kagglehub
import kagglehub; path = kagglehub.dataset_download("l33tc0d3r/indian-food-classification")
!cd food-classifier-pro && python scripts/resize_dataset.py --src "{path}" --out /content/food256
!cd food-classifier-pro && python -m foodai.compare --data-dir /content/food256 --num-workers 4 --epochs 20 --ablation --export-best
```
Tip: on CPU only, add `--img-size 160 --epochs 10` and compare just `--archs baseline_cnn mobilenet_v3_large`.

### 4. Use the model
```bash
python -m foodai.predict --checkpoint models/best.pt --image samosa.jpg --save-gradcam cam.png
python -m webapp.app                                   # http://127.0.0.1:5000
python -m foodai.export --checkpoint models/best.pt    # models/model.onnx (+ preprocessing sidecar JSON)
docker build -t food-classifier . && docker run -p 5000:5000 -v "$(pwd)/models:/app/models" food-classifier
```

### REST API
```bash
curl -F "file=@samosa.jpg" http://127.0.0.1:5000/api/predict
```
```json
{"label": "samosa", "display_label": "Samosa", "confidence": 0.93, "is_confident": true,
 "threshold": 0.62, "uncertainty": 0.12,
 "topk": [{"label": "samosa", "confidence": 0.93}, {"label": "pakode", "confidence": 0.04}, "..."],
 "gradcam_png_b64": "...", "nutrition": {"serving": "1 piece", "kcal_min": 130, "kcal_max": 260}}
```
Other endpoints: `GET /api/health`, `GET /api/model-info`, `GET /api/classes`.

---

## How it works

```
Kaggle images ─► audit (corrupt / duplicates) ─► stratified split (saved) ─► augment + MixUp/CutMix
      ─► [frozen backbone → fine-tune] AdamW + cosine ─► best checkpoint (val accuracy)
      ─► test-set evaluation ─► temperature scaling + reject threshold (fit on validation only)
      ─► Grad-CAM gallery ─► checkpoint now carries calibration + metrics ─► web app / API / ONNX
```

### Design decisions worth explaining in a viva
* **Why transfer learning?** ~4k images is far too few to learn good features from scratch; ImageNet features
  transfer well to food. The baseline-vs-pretrained gap in the table quantifies this.
* **Why freeze first?** A randomly initialised head produces large gradients that would wreck pretrained
  features. Training the head alone for 2 epochs, then unfreezing with a 10× lower backbone LR, is the standard fix.
* **Why MixUp/CutMix + label smoothing?** They stop the network becoming over-confident on a small dataset and
  improve calibration, not just accuracy.
* **Why no leakage?** Duplicates are removed *before* splitting; the split is stratified and saved; temperature
  and threshold are fitted on validation and only *reported* on test.
* **Why calibrate?** Raw softmax is over-confident. After temperature scaling, "90 %" really means ~90 %,
  which is what makes a reject threshold meaningful.
* **What does Grad-CAM show?** `ReLU(Σ_k mean(∂score/∂A_k) · A_k)` over the last conv feature map — regions that
  increased the predicted class score. Use it to catch the model "cheating" (e.g. looking at the plate or background).
* **Why report macro-F1 and not only accuracy?** It weights every dish equally even if the class counts differ.

### Honest limitations
* 20 fixed dishes; photos of other food will be mapped to the nearest class unless confidence is below the threshold.
  The reject option reduces but does not eliminate this — it is a confidence heuristic, not a trained
  out-of-distribution detector. The "synthetic input" check only covers noise / flat colour / gradients.
* Visually similar dishes (e.g. chapati vs butter naan, pakode vs samosa) are the expected confusions.
* Calorie ranges in `data/nutrition.json` are rough typical-serving estimates, not nutritional data.
* Accuracy is a single seed on a single split; for a thesis-grade claim repeat the best model with a few different
  `--seed` values (give each a different `--name`) and report the spread.

---

## Project layout
```
foodai/            data.py  models.py  train.py  evaluate.py  gradcam.py  predict.py  export.py  compare.py  utils.py
webapp/            app.py  templates/index.html  static/{style.css,app.js}
tests/             pytest suite (synthetic data, trains a tiny model end to end)
scripts/           download_dataset.py  resize_dataset.py  make_synthetic_dataset.py
data/nutrition.json   rough kcal table      docs/REPORT_TEMPLATE.md   report skeleton
Dockerfile  requirements*.txt  .github/workflows/ci.yml
```

## Results

Fill this from `results/comparison.md` after running `foodai.compare` (it writes the markdown table for you):

| Experiment | Acc | Top-3 | Macro-F1 | ECE | Params (M) | CPU ms/img |
|---|---|---|---|---|---|---|
| *(run `python -m foodai.compare ...` and paste here)* | | | | | | |

## Credits
Original project and dataset choice: [AdwitiyaKhare/food-classifier](https://github.com/AdwitiyaKhare/food-classifier).
Dataset: [l33tc0d3r/indian-food-classification](https://www.kaggle.com/datasets/l33tc0d3r/indian-food-classification) (Kaggle).
Built with PyTorch, torchvision, scikit-learn, Flask.
