# Project Report — Indian Food Classification with Explainable, Calibrated Deep Learning

*(Fill the bracketed items with numbers from `runs/<name>/metrics.json` and `results/comparison.md`.)*

## 1. Abstract
We build an image classifier for 20 Indian/street-food categories. We compare a from-scratch CNN with
transfer-learned ResNet18, EfficientNet-B0 and MobileNetV3, study the effect of pretraining and augmentation,
calibrate model confidence, add a reject option for unfamiliar inputs, and explain predictions with Grad-CAM.
The best model, [arch], reaches [acc]% test accuracy ([f1] macro-F1).

## 2. Problem statement and motivation
Automatic dish recognition supports dietary logging, restaurant apps and accessibility tools. Challenges:
high intra-class variation, visually similar dishes, small dataset, uncontrolled photos.

## 3. Dataset
Kaggle *Indian Food Classification*: [N] images, 20 classes (see `runs/<name>/data_report.json`).
Pre-processing: removed [x] corrupt and [y] duplicate files; stratified 70/15/15 train/val/test split (seed 42).
Include `class distribution` from the data report.

## 4. Methodology
1. **Baseline CNN** – 5 conv blocks (conv-BN-ReLU ×2 + max-pool), global average pooling, dropout, linear head.
2. **Transfer learning** – ImageNet-pretrained backbone, new classifier head; phase 1 head-only, phase 2 full fine-tune
   (backbone LR = 0.1 × head LR).
3. **Training details** – AdamW, warm-up + cosine schedule, label smoothing 0.1, MixUp(α=0.2)/CutMix(α=1.0) on 50 % of batches,
   TrivialAugment, RandomResizedCrop, RandomErasing, early stopping (patience 7), image size 224.
4. **Calibration** – temperature scaling fitted on the validation split; ECE with 15 bins.
5. **Reject option** – lowest confidence threshold that gives ≥ 90 % accuracy on answered validation images.
6. **Explainability** – Grad-CAM on the last convolutional stage.

## 5. Experiments and results
Paste `results/comparison.md`, then discuss:
* Pretrained vs scratch: [Δ accuracy].
* Ablations: no-pretraining [acc], no-augmentation [acc].
* Accuracy vs size vs latency trade-off (params, CPU ms/img).
* Figures: `history.png`, `confusion_matrix.png`, `per_class_f1.png`, `reliability_diagram.png`,
  `selective_prediction.png`, `gradcam_examples.png`.

## 6. Error analysis
Most confused pairs (from `metrics.json → top_confusions`): [pair 1], [pair 2], [pair 3].
Discuss why (colour, shape, deep-fried look-alikes) and what Grad-CAM shows for wrong predictions.

## 7. Deployment
Flask web app + REST API (top-3, Grad-CAM, reject flag), ONNX export, Docker image, CI with automated tests.

## 8. Limitations and future work
Closed 20-class set; reject option is a confidence heuristic; single split/seed; ideas: more classes,
trained OOD detector, multi-seed confidence intervals, knowledge distillation to a smaller model, mobile deployment.

## 9. References
* He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016.
* Tan & Le, *EfficientNet*, ICML 2019.
* Howard et al., *Searching for MobileNetV3*, ICCV 2019.
* Zhang et al., *mixup*, ICLR 2018; Yun et al., *CutMix*, ICCV 2019.
* Guo et al., *On Calibration of Modern Neural Networks*, ICML 2017.
* Selvaraju et al., *Grad-CAM*, ICCV 2017.
