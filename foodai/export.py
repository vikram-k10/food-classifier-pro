"""Export a trained checkpoint to ONNX (portable, fast CPU inference) and verify it.

    python -m foodai.export --checkpoint models/best.pt --output models/model.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .models import load_checkpoint
from .utils import save_json


def export_onnx(checkpoint, output, opset=17, verify=True):
    model, meta = load_checkpoint(checkpoint, "cpu")
    size = meta["img_size"]
    dummy = torch.randn(1, 3, size, size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset,
    )
    try:  # newer torch defaults to the dynamo exporter; the classic exporter is the most portable
        torch.onnx.export(model, dummy, str(output), dynamo=False, **kwargs)
    except TypeError:  # older torch has no `dynamo` argument
        torch.onnx.export(model, dummy, str(output), **kwargs)

    # sidecar with everything a non-Python consumer needs to run the model correctly
    save_json(
        {"classes": meta["classes"], "img_size": size, "temperature": meta.get("temperature", 1.0),
         "reject_threshold": meta.get("threshold", 0.5),
         "preprocess": {"resize": int(round(size * 1.14)), "center_crop": size,
                        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}},
        output.with_suffix(".json"),
    )

    max_diff = None
    if verify:
        try:
            import onnxruntime as ort
        except ImportError:
            print("onnxruntime not installed - skipping numerical verification")
        else:
            sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            x = torch.randn(2, 3, size, size)
            with torch.no_grad():
                ref = model(x).numpy()
            got = sess.run(None, {"image": x.numpy()})[0]
            max_diff = float(np.abs(ref - got).max())
            print(f"ONNX vs PyTorch max |logit difference| = {max_diff:.2e}")
            assert max_diff < 1e-3, "ONNX output does not match PyTorch"
    print(f"Exported -> {output}")
    return max_diff


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="models/model.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--no-verify", action="store_true")
    a = p.parse_args(argv)
    export_onnx(a.checkpoint, a.output, a.opset, verify=not a.no_verify)


if __name__ == "__main__":
    main()
