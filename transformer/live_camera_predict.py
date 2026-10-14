"""
live_camera_predict_corrected.py
--------------------------------
Raspberry Pi Camera Module 2 -> ONNX transformer fault classifier.

IMPORTANT COLOUR FIX
--------------------
Picamera2/libcamera pixel-format names are counter-intuitive:

- BGR888 returned by capture_array() is laid out [R, G, B]
- RGB888 returned by capture_array() is laid out [B, G, R]

This script requests BGR888 so the NumPy array is true RGB for the CNN/PIL.
For OpenCV preview, RGB is converted to BGR only at display time.

Examples
--------
Live classifier + preview:
    python live_camera_predict_corrected.py --display --min-confidence 0.60

Terminal-only classifier:
    python live_camera_predict_corrected.py --min-confidence 0.60

Controls in preview window:
    q = quit
    s = save current camera frame as camera_live_capture.jpg
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
from PIL import Image


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def preprocess_rgb(frame_rgb, img_size, mean, std):
    """Match the CNN evaluation pipeline: RGB -> 224 -> [0,1] -> normalize."""
    image = Image.fromarray(frame_rgb, mode="RGB")
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)

    x = np.asarray(image, dtype=np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW

    mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    x = (x - mean) / std

    return np.expand_dims(x, axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(
        description="Correct-colour live Camera Module 2 ONNX classifier"
    )
    ap.add_argument("--model", default="transformer_fault_classifier.onnx")
    ap.add_argument("--meta", default="transformer_fault_classifier.json")
    ap.add_argument("--display", action="store_true")
    ap.add_argument("--camera-width", type=int, default=640)
    ap.add_argument("--camera-height", type=int, default=480)
    ap.add_argument("--interval", type=float, default=0.25,
                    help="Minimum seconds between CNN inferences")
    ap.add_argument("--min-confidence", type=float, default=0.60)
    ap.add_argument("--save-name", default="camera_live_capture.jpg")
    args = ap.parse_args()

    model_path = Path(args.model)
    meta_path = Path(args.meta)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta["class_names"]
    img_size = int(meta["img_size"])
    mean = meta["mean"]
    std = meta["std"]

    # ONNX Runtime
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    # Camera
    picam2 = Picamera2()

    # Counter-intuitive Picamera2 naming:
    # BGR888 capture_array() gives [R,G,B], which is what PIL/CNN needs.
    camera_config = picam2.create_preview_configuration(
        main={
            "format": "BGR888",
            "size": (args.camera_width, args.camera_height),
        },
        controls={"FrameRate": 15},
        buffer_count=4,
    )

    picam2.configure(camera_config)
    picam2.start()

    # Allow exposure and white-balance to settle before classification.
    time.sleep(2.0)

    cv2 = None
    if args.display:
        try:
            import cv2 as _cv2
            cv2 = _cv2
        except ImportError as exc:
            picam2.stop()
            raise RuntimeError(
                "OpenCV is not available. Install it with:\n"
                "sudo apt install -y python3-opencv"
            ) from exc

    print("=" * 72)
    print("LIVE TRANSFORMER FAULT CLASSIFIER - CORRECTED RGB")
    print("=" * 72)
    print("Camera       : Raspberry Pi Camera Module 2")
    print(f"Camera frame : {args.camera_width}x{args.camera_height}")
    print("Picamera fmt : BGR888 -> NumPy [R,G,B]")
    print(f"CNN input    : {img_size}x{img_size} RGB")
    print(f"Threshold    : {args.min_confidence*100:.0f}%")
    if args.display:
        print("Controls     : q = quit, s = save current frame")
    else:
        print("Stop         : Ctrl+C")
    print("=" * 72)

    last_infer_time = 0.0
    last_label = "Starting"
    last_conf = 0.0
    last_ms = 0.0
    last_top3 = []

    try:
        while True:
            # With BGR888, this NumPy array is [R,G,B] according to Picamera2 docs.
            frame_rgb = picam2.capture_array("main")

            now = time.perf_counter()

            if now - last_infer_time >= args.interval:
                x = preprocess_rgb(
                    frame_rgb,
                    img_size,
                    mean,
                    std,
                )

                t0 = time.perf_counter()
                logits = session.run(None, {input_name: x})[0]
                last_ms = (time.perf_counter() - t0) * 1000.0

                probs = softmax(logits)[0]
                order = np.argsort(probs)[::-1]
                pred = int(order[0])

                last_label = classes[pred]
                last_conf = float(probs[pred])
                last_top3 = [
                    (classes[int(i)], float(probs[int(i)]))
                    for i in order[:3]
                ]
                last_infer_time = now

                shown = (
                    last_label
                    if last_conf >= args.min_confidence
                    else "UNCERTAIN"
                )

                top3_text = " | ".join(
                    f"{name} {prob*100:.1f}%"
                    for name, prob in last_top3
                )

                print(
                    f"\rPrediction: {shown:<10} "
                    f"{last_conf*100:6.2f}% | "
                    f"{last_ms:6.1f} ms | "
                    f"{top3_text:<55}",
                    end="",
                    flush=True,
                )

            if cv2 is not None:
                # frame_rgb is true RGB; OpenCV imshow expects BGR.
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                shown = (
                    last_label
                    if last_conf >= args.min_confidence
                    else "UNCERTAIN"
                )

                # Prediction panel
                cv2.rectangle(frame_bgr, (10, 10), (470, 105), (0, 0, 0), -1)
                cv2.putText(
                    frame_bgr,
                    f"{shown}  {last_conf*100:.1f}%",
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"Inference: {last_ms:.0f} ms",
                    (20, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    "q: quit   s: save frame",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                )

                cv2.imshow(
                    "Transformer Fault Classifier - Correct RGB",
                    frame_bgr,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

                if key == ord("s"):
                    # Save RGB correctly using PIL.
                    Image.fromarray(frame_rgb, mode="RGB").save(args.save_name)
                    print(f"\nSaved camera frame: {args.save_name}")

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        picam2.stop()

        if cv2 is not None:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
