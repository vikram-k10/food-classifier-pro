"""Inference wrapper: calibrated probabilities, TTA, reject option and Grad-CAM in one call.

CLI::

    python -m foodai.predict --checkpoint models/best.pt --image samosa.jpg
"""
from __future__ import annotations

import argparse
import base64
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import build_transforms, denormalize, load_rgb
from .gradcam import GradCAM, overlay_cam
from .models import gradcam_layer, load_checkpoint
from .utils import get_device, pretty_name


class Predictor:
    def __init__(self, checkpoint: str, device: str = "auto", tta: bool = True, gate=None):
        self.gate = gate  # optional FoodGate: rejects non-food images before the dish classifier answers
        self.device = get_device(device)
        self.model, self.meta = load_checkpoint(checkpoint, self.device)
        self.arch: str = self.meta["arch"]
        self.classes: list[str] = self.meta["classes"]
        self.temperature: float = float(self.meta.get("temperature", 1.0))
        self.threshold: float = float(self.meta.get("threshold", 0.5))
        self.tta = tta
        self.transform = build_transforms(self.meta["img_size"], train=False)
        self._cam = GradCAM(self.model, gradcam_layer(self.model, self.arch))

    # ------------------------------------------------------------------ probabilities
    @torch.no_grad()
    def _probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        probs = F.softmax(logits / self.temperature, dim=1)
        if self.tta:  # test-time augmentation: average with the horizontally flipped view
            flipped = self.model(torch.flip(x, dims=[3]))
            probs = (probs + F.softmax(flipped / self.temperature, dim=1)) / 2.0
        return probs[0].cpu()

    def predict(self, image: Image.Image, topk: int = 3, explain: bool = True) -> dict:
        image = image.convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)
        probs = self._probabilities(x)
        k = min(topk, len(self.classes))
        conf, idx = probs.topk(k)
        entropy = float(-(probs * torch.log(probs.clamp_min(1e-12))).sum() / np.log(len(self.classes)))
        result = {
            "label": self.classes[int(idx[0])],
            "display_label": pretty_name(self.classes[int(idx[0])]),
            "confidence": float(conf[0]),
            "is_confident": bool(float(conf[0]) >= self.threshold),
            "threshold": self.threshold,
            "uncertainty": entropy,  # 0 = certain, 1 = uniform guess
            "topk": [
                {"label": self.classes[int(i)], "display_label": pretty_name(self.classes[int(i)]),
                 "confidence": float(c)}
                for c, i in zip(conf, idx)
            ],
        }
        result["is_food"], result["food_score"] = True, None
        if self.gate is not None:
            result["is_food"], result["food_score"] = self.gate.is_food(image)
            if not result["is_food"]:
                result["is_confident"] = False  # never name a dish for something that isn't food
        if explain:
            cam, _ = self._cam(x, class_idx=int(idx[0]))
            base = denormalize(x[0].cpu()).permute(1, 2, 0).numpy()
            result["model_view_png_b64"] = _to_b64(Image.fromarray((base * 255).astype(np.uint8)))
            result["gradcam_png_b64"] = _to_b64(Image.fromarray(overlay_cam(base, cam)))
        return result


def _to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main(argv=None):
    p = argparse.ArgumentParser(description="Classify one food image.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--no-gate", action="store_true", help="skip the 'is this food?' CLIP check")
    p.add_argument("--save-gradcam", default=None, help="write the Grad-CAM overlay to this PNG path")
    args = p.parse_args(argv)
    gate = None
    if not args.no_gate:
        from .foodgate import FoodGate
        gate = FoodGate(device="cpu")
    predictor = Predictor(args.checkpoint, args.device, tta=not args.no_tta, gate=gate)
    res = predictor.predict(load_rgb(args.image), explain=args.save_gradcam is not None)
    if not res["is_food"]:
        verdict = f"  (NOT FOOD: food score {res['food_score']:.2f})"
    else:
        verdict = "" if res["is_confident"] else f"  (below reject threshold {res['threshold']:.2f} -> 'not sure')"
    print(f"Prediction: {res['display_label']}  ({res['confidence'] * 100:.1f}%){verdict}")
    for t in res["topk"]:
        print(f"  {t['display_label']:<20s} {t['confidence'] * 100:5.1f}%")
    if args.save_gradcam:
        with open(args.save_gradcam, "wb") as f:
            f.write(base64.b64decode(res["gradcam_png_b64"]))
        print(f"Grad-CAM saved to {args.save_gradcam}")
    return res


if __name__ == "__main__":
    main()
