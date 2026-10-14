"""
camera_inference.py
--------------------
Runs the trained model on a live camera feed and overlays the predicted
fault class on the video window. Press 'q' to quit.

*** READ THIS BEFORE YOU RUN IT ***
------------------------------------
The model was trained entirely on THERMAL images from a Dali-tech IRT
camera. A normal laptop/webcam captures ordinary visible-light RGB video --
a completely different physical signal (reflected light, not radiated
heat). Pointing your laptop webcam at a motor will NOT give you real fault
predictions, no matter how the code is written, because the model has
never seen what a healthy vs. faulty motor looks like in visible light --
it only knows the colour patterns that correspond to temperature in a
thermal image. Running this script against a plain webcam is only useful
as a demo of the plumbing (camera -> preprocessing -> model -> overlay),
not as an actual fault detector.

This script becomes genuinely useful if:
  1. You have an actual thermal/IR camera (e.g. a Dali-tech, FLIR, Seek
     Thermal, or similar unit) that can output a standard USB video stream.
     Many of these register as a normal webcam device, in which case
     CAMERA_INDEX below just needs to point at that device and everything
     works as intended.
  2. Or you export frames from the thermal camera's own software/app and
     feed those files through predict.py instead of this live script.

If you only have a regular webcam, treat this script as a placeholder for
when you get thermal hardware, not as a working fault detector today.
"""

import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from train_cnn import build_model, CLAHETransform

MODEL_DIR = Path("outputs")
MODEL_PATH = MODEL_DIR / "motor_fault_model.pth"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"

CAMERA_INDEX = 0        # change if you have multiple cameras attached
CONFIDENCE_THRESHOLD = 0.5


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


def main():
    if not MODEL_PATH.exists():
        print(f"No trained model found at {MODEL_PATH}. Run train_cnn.py first.")
        return

    model, transform, classes, device = load_model()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Could not open camera index {CAMERA_INDEX}.")
        return

    print("Press 'q' to quit.")
    print("Reminder: this only produces meaningful predictions on a thermal camera feed (see file docstring).")

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        x = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        top_idx = probs.argmax()
        label, conf = classes[top_idx], float(probs[top_idx])

        color = (0, 200, 0) if conf >= CONFIDENCE_THRESHOLD else (0, 165, 255)
        text = f"{label}: {conf:.1%}"
        cv2.rectangle(frame_bgr, (0, 0), (frame_bgr.shape[1], 40), (30, 30, 30), -1)
        cv2.putText(frame_bgr, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Motor fault detection (thermal input expected)", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
