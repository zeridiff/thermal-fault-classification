r"""
predict.py
----------
Run the trained model (from train_cnn.py) on one image or a folder of
images and print the predicted fault class + confidence.

Usage:
    python predict.py path\to\image.jpg
    python predict.py path\to\folder_of_images
"""

import sys
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train_cnn import build_model, IMAGE_EXTENSIONS, CLAHETransform

MODEL_DIR = Path("outputs")
MODEL_PATH = MODEL_DIR / "motor_fault_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"


def load_model():
    with open(CLASS_NAMES_PATH) as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(meta["model_name"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    ops = []
    if meta.get("enable_clahe"):
        ops.append(CLAHETransform())
    ops += [
        transforms.Resize((meta["img_size"], meta["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    transform = transforms.Compose(ops)

    return model, transform, meta["classes"], device


def predict_image(model, transform, classes, device, image_path):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    order = probs.argsort()[::-1]
    top3 = [(classes[i], float(probs[i])) for i in order[:3]]
    return top3


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}. Run train_cnn.py first.")
        sys.exit(1)

    model, transform, classes, device = load_model()

    if target.is_dir():
        paths = [p for p in sorted(target.rglob("*")) if p.suffix.lower() in IMAGE_EXTENSIONS]
    else:
        paths = [target]

    for p in paths:
        top3 = predict_image(model, transform, classes, device, p)
        pred_class, pred_conf = top3[0]
        others = ", ".join(f"{c}: {conf:.1%}" for c, conf in top3[1:])
        print(f"{p.name:30s} -> {pred_class:15s} ({pred_conf:.1%})   [next: {others}]")


if __name__ == "__main__":
    main()
