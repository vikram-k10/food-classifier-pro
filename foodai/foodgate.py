"""'Is this food at all?' gate (open-set rejection) using zero-shot CLIP.

Why this exists
---------------
The dish classifier only knows ~20 classes. Softmax must spread 100% of its probability over
those classes, so a screenshot, a person or a car still gets *some* dish name - sometimes with
high confidence. A confidence threshold cannot fix that reliably.

This gate asks a different question first: "is this a photo of food?". It compares the image
against text prompts for food and for common non-food things with CLIP, and the dish classifier
only answers when the image looks like food.

    gate = FoodGate()              # downloads openai/clip-vit-base-patch32 (~600 MB) on first use
    gate.food_score(pil_image)     # 0..1, probability the image is food
"""
from __future__ import annotations

import torch
from PIL import Image

MODEL_NAME = "openai/clip-vit-base-patch32"

FOOD_PROMPTS = [
    "a photo of food",
    "a photo of a plate of Indian food",
    "a close-up photo of a cooked dish",
    "a photo of a meal on a table",
    "a photo of a snack or street food",
    "a photo of a dessert or sweet",
]

NON_FOOD_PROMPTS = [
    "a screenshot of a computer screen",
    "a screenshot of a phone app",
    "a screenshot of source code",
    "a photo of a document or text",
    "a photo of a person",
    "a photo of a face",
    "a photo of an animal",
    "a photo of a building",
    "a photo of a car or vehicle",
    "a photo of a landscape or nature",
    "a photo of furniture or a room",
    "a photo of an electronic device",
    "a logo, drawing or cartoon",
    "a photo of clothing",
    "a blank or random image",
]


class FoodGate:
    def __init__(self, threshold: float = 0.5, device: str = "cpu", model_name: str = MODEL_NAME):
        from transformers import CLIPModel, CLIPProcessor  # imported lazily: optional dependency

        self.threshold = float(threshold)
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self._prompts = FOOD_PROMPTS + NON_FOOD_PROMPTS
        self._n_food = len(FOOD_PROMPTS)
        # Text tokens are fixed, so tokenise once. We use the plain forward pass (not get_*_features),
        # because its return type changed between transformers versions.
        self._text_inputs = self.processor(text=self._prompts, return_tensors="pt", padding=True).to(device)

    @torch.no_grad()
    def food_score(self, image: Image.Image) -> float:
        px = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"].to(self.device)
        out = self.model(pixel_values=px, **self._text_inputs)
        probs = out.logits_per_image.softmax(dim=-1)[0]  # already scaled by CLIP's learned temperature
        return float(probs[: self._n_food].sum())

    def is_food(self, image: Image.Image) -> tuple[bool, float]:
        score = self.food_score(image)
        return score >= self.threshold, score
