"""Étape 3 : fine-tuning de YOLOv8n sur le dataset EPI + évaluation sur le split test."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "yolo" / "data.yaml"
IMG_SIZE = 416


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="yolov8n_full")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--freeze", type=int, default=0, help="nb de couches gelées (10 = backbone)")
    p.add_argument("--device", default="0", help="0 = GPU Colab, mps = Mac, cpu")
    args = p.parse_args()

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA),
        imgsz=IMG_SIZE,
        epochs=args.epochs,
        batch=args.batch,
        freeze=args.freeze or None,
        patience=15,
        device=args.device,
        project=str(ROOT / "runs"),
        name=args.name,
        seed=42,
        exist_ok=True,
    )

    best = YOLO(ROOT / "runs" / args.name / "weights" / "best.pt")
    m = best.val(data=str(DATA), split="test", imgsz=IMG_SIZE, device=args.device,
                 project=str(ROOT / "runs"), name=f"{args.name}_test", exist_ok=True)
    print(f"\n=== TEST ({args.name}) ===")
    print(f"mAP50    : {m.box.map50:.3f}")
    print(f"mAP50-95 : {m.box.map:.3f}")
    print(f"Précision: {m.box.mp:.3f} | Rappel: {m.box.mr:.3f}")
    for i, name in best.names.items():
        print(f"  {name:8s} AP50={m.box.ap50[i]:.3f}  AP50-95={m.box.ap[i]:.3f}")


if __name__ == "__main__":
    main()