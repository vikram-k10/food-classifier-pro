import base64
import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from foodai.data import find_dataset_root, make_splits, mix_batch, scan_images, audit_samples
from foodai.evaluate import expected_calibration_error, fit_temperature, select_threshold
from foodai.gradcam import GradCAM
from foodai.models import ARCHS, build_model, gradcam_layer, load_checkpoint


# ------------------------------------------------------------------ data
def test_root_autodetect_and_scan(synthetic_root):
    root = find_dataset_root(synthetic_root)
    assert root.name == "Images"
    classes, samples = scan_images(root)
    assert classes == ["class_0", "class_1", "class_2", "class_3"]
    assert len(samples) == 120


def test_audit_removes_corrupt_and_duplicates(tmp_path):
    for c in ("a", "b"):
        (tmp_path / c).mkdir()
        for i in range(3):
            Image.fromarray(np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)).save(tmp_path / c / f"{i}.png")
    (tmp_path / "a" / "bad.jpg").write_bytes(b"not an image")
    (tmp_path / "b" / "dup.png").write_bytes((tmp_path / "b" / "0.png").read_bytes())
    classes, samples = scan_images(tmp_path)
    clean, corrupt, dups = audit_samples(tmp_path, samples)
    assert corrupt == ["a/bad.jpg"]
    assert dups == ["b/dup.png"] or dups == ["b/0.png"]
    assert len(clean) == 6


def test_splits_are_disjoint_and_stratified(synthetic_root):
    _, samples = scan_images(find_dataset_root(synthetic_root))
    s = make_splits(samples, 0.15, 0.15, seed=1)
    paths = {k: {p for p, _ in v} for k, v in s.items()}
    assert not (paths["train"] & paths["val"]) and not (paths["train"] & paths["test"]) and not (paths["val"] & paths["test"])
    assert sum(len(v) for v in s.values()) == len(samples)
    for split in s.values():
        counts = np.bincount([y for _, y in split], minlength=4)
        assert counts.min() > 0 and counts.max() - counts.min() <= 2


def test_mix_batch_shapes_and_lambda():
    x = torch.randn(8, 3, 32, 32)
    y = torch.arange(8)
    for _ in range(20):
        xm, ya, yb, lam = mix_batch(x, y, 0.2, 1.0, prob=1.0)
        assert xm.shape == x.shape and 0.0 <= lam <= 1.0
    xm, ya, yb, lam = mix_batch(x, y, 0.2, 1.0, prob=0.0)
    assert lam == 1.0 and torch.equal(xm, x)


# ------------------------------------------------------------------ models + gradcam
@pytest.mark.parametrize("arch", ARCHS)
def test_every_architecture_forward_and_gradcam(arch):
    model = build_model(arch, num_classes=5, pretrained=False).eval()
    x = torch.randn(2, 3, 64, 64)
    assert model(x).shape == (2, 5)
    cam, cls = GradCAM(model, gradcam_layer(model, arch))(x[:1])
    assert cam.shape == (64, 64) and 0.0 <= cam.min() and cam.max() <= 1.0 + 1e-6 and 0 <= cls < 5


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_model("vgg_nonsense", 3)


def test_backbone_freezing_only_leaves_head_trainable():
    from foodai.models import get_head, set_backbone_trainable

    for arch in ["resnet18", "efficientnet_b0", "baseline_cnn"]:
        model = build_model(arch, 5, pretrained=False)
        head_ids = {id(p) for p in get_head(model).parameters()}
        set_backbone_trainable(model, False)
        assert all(p.requires_grad == (id(p) in head_ids) for p in model.parameters())
        set_backbone_trainable(model, True)
        assert all(p.requires_grad for p in model.parameters())


def test_transforms_output_shapes():
    from foodai.data import build_transforms

    img = Image.fromarray(np.random.randint(0, 255, (90, 120, 3), dtype=np.uint8))
    for train, augment in [(True, True), (True, False), (False, True)]:
        out = build_transforms(64, train=train, augment=augment)(img)
        assert out.shape == (3, 64, 64)


# ------------------------------------------------------------------ calibration helpers
def test_temperature_scaling_improves_overconfident_model():
    torch.manual_seed(0)
    n, k = 600, 5
    labels = torch.randint(0, k, (n,))
    logits = torch.randn(n, k)
    logits[torch.arange(n), labels] += 1.5  # ~weakly informative
    logits = logits * 6.0  # make it badly over-confident
    t = fit_temperature(logits, labels)
    assert t > 1.5
    nll_before = torch.nn.functional.cross_entropy(logits, labels)
    nll_after = torch.nn.functional.cross_entropy(logits / t, labels)
    assert nll_after < nll_before
    ece_before = expected_calibration_error(torch.softmax(logits, 1), labels)
    ece_after = expected_calibration_error(torch.softmax(logits / t, 1), labels)
    assert ece_after < ece_before


def test_select_threshold_reaches_target():
    rng = np.random.RandomState(0)
    conf = rng.uniform(0.2, 1.0, 2000)
    correct = rng.uniform(size=2000) < conf  # perfectly calibrated
    t = select_threshold(conf, correct, target_acc=0.9, min_coverage=0.1)
    assert 0.5 < t < 0.99
    assert correct[conf >= t].mean() >= 0.9


def test_select_threshold_falls_back_when_target_unreachable():
    conf = np.full(100, 0.3)
    correct = np.zeros(100, dtype=bool)
    assert select_threshold(conf, correct, target_acc=0.9) == 0.5


# ------------------------------------------------------------------ end to end
def test_training_run_artifacts_and_learning(trained_run):
    run_dir, summary = trained_run
    for name in ["best.pt", "metrics.json", "history.json", "history.png", "confusion_matrix.png",
                 "per_class_f1.png", "reliability_diagram.png", "selective_prediction.png",
                 "gradcam_examples.png", "per_class_metrics.csv", "splits.json", "data_report.json", "config.json"]:
        assert (run_dir / name).exists(), name
    m = json.loads((run_dir / "metrics.json").read_text())
    assert m["accuracy"] > 0.6, f"model failed to learn the easy synthetic task: {m['accuracy']}"
    assert 0.0 <= m["macro_f1"] <= 1.0 and m["temperature"] > 0 and 0 < m["reject_threshold"] < 1
    assert set(m["synthetic_input_rejection_rate"]) == {"uniform_noise", "solid_color", "gradient"}
    hist = json.loads((run_dir / "history.json").read_text())
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]


def test_augmentation_mixup_path_runs(augmented_run):
    run_dir, summary = augmented_run
    assert (run_dir / "best.pt").exists() and (run_dir / "metrics.json").exists()
    assert 0.0 <= summary["metrics"]["accuracy"] <= 1.0
    assert json.loads((run_dir / "config.json").read_text())["pretrained"] is False


def test_checkpoint_roundtrip_has_calibration(trained_run):
    run_dir, _ = trained_run
    model, meta = load_checkpoint(run_dir / "best.pt")
    assert meta["classes"] == ["class_0", "class_1", "class_2", "class_3"]
    assert meta["temperature"] > 0 and "accuracy" in meta["test_metrics"]


def test_predictor_returns_valid_distribution(trained_run, synthetic_root):
    from foodai.predict import Predictor

    run_dir, _ = trained_run
    pred = Predictor(str(run_dir / "best.pt"), device="cpu")
    img_path = next((synthetic_root / "archive" / "Images" / "class_2").glob("*.jpg"))
    res = pred.predict(Image.open(img_path).convert("RGB"))
    assert len(res["topk"]) == 3
    confs = [t["confidence"] for t in res["topk"]]
    assert confs == sorted(confs, reverse=True) and 0 < confs[0] <= 1
    assert 0.0 <= res["uncertainty"] <= 1.0
    Image.open(io.BytesIO(base64.b64decode(res["gradcam_png_b64"]))).verify()


def test_web_app_api(trained_run, synthetic_root):
    from webapp.app import create_app

    run_dir, _ = trained_run
    client = create_app(str(run_dir / "best.pt")).test_client()
    assert client.get("/api/health").get_json()["model_loaded"] is True
    assert client.get("/").status_code == 200
    assert len(client.get("/api/classes").get_json()["classes"]) == 4
    assert client.get("/api/model-info").get_json()["arch"] == "baseline_cnn"

    img_path = next((synthetic_root / "archive" / "Images" / "class_1").glob("*.jpg"))
    with open(img_path, "rb") as f:
        r = client.post("/api/predict", data={"file": (f, "x.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["topk"]

    assert client.post("/api/predict", data={}, content_type="multipart/form-data").status_code == 400
    bad = client.post("/api/predict", data={"file": (io.BytesIO(b"hello"), "x.jpg")}, content_type="multipart/form-data")
    assert bad.status_code == 400
    wrong_ext = client.post("/api/predict", data={"file": (io.BytesIO(b"x"), "x.txt")}, content_type="multipart/form-data")
    assert wrong_ext.status_code == 400


def test_web_app_without_model_degrades_gracefully(tmp_path):
    from webapp.app import create_app

    client = create_app(str(tmp_path / "missing.pt")).test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/health").get_json()["model_loaded"] is False
    assert client.post("/api/predict", data={}, content_type="multipart/form-data").status_code == 503


def test_onnx_export_matches_pytorch(trained_run, tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from foodai.export import export_onnx

    run_dir, _ = trained_run
    diff = export_onnx(run_dir / "best.pt", tmp_path / "m.onnx")
    assert diff is not None and diff < 1e-3
