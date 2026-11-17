"""Étape 2 : YOLOv8n pré-entraîné (COCO), baseline sans fine-tuning et cartes de caractéristiques."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
IMG_DIR = ROOT / "data" / "yolo" / "images" / "test"
IMG_SIZE = 416
# Couches du backbone YOLOv8n : strides 4, 8, 16, 32
LAYERS = {2: "C2f (stride 4)", 4: "C2f (stride 8)", 6: "C2f (stride 16)", 9: "SPPF (stride 32)"}


def pick_image():
    return sorted(IMG_DIR.glob("*.jpg"))[0]


def baseline_inference(model, img_path):
    """Le modèle COCO ne connaît ni casque ni tête nue : seule 'person' sera pertinente."""
    res = model.predict(str(img_path), imgsz=IMG_SIZE, device="cpu", conf=0.25, verbose=False)[0]
    cv2.imwrite(str(FIG / "baseline_coco.jpg"), res.plot())
    print("Classes détectées par le modèle COCO :", {model.names[int(c)] for c in res.boxes.cls})


def extract_feature_maps(model, img_path):
    feats = {}
    hooks = [
        model.model.model[i].register_forward_hook(
            lambda m, inp, out, i=i: feats.__setitem__(i, out.detach())
        )
        for i in LAYERS
    ]
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).unsqueeze(0)

    model.model.eval()
    with torch.no_grad():
        model.model(x)
    for h in hooks:
        h.remove()
    return img, feats


def plot_feature_maps(img, feats, n_channels=4):
    fig, axes = plt.subplots(len(LAYERS), n_channels + 2, figsize=(3 * (n_channels + 2), 3 * len(LAYERS)))
    for row, (idx, title) in enumerate(LAYERS.items()):
        f = feats[idx][0]  # (C, H, W)
        axes[row, 0].imshow(img)
        axes[row, 0].set_title("Image")
        axes[row, 1].imshow(f.mean(0).numpy(), cmap="viridis")
        axes[row, 1].set_title(f"{title}\nmoyenne ({f.shape[0]} canaux, {f.shape[1]}x{f.shape[2]})", fontsize=8)
        for c in range(n_channels):
            axes[row, c + 2].imshow(f[c].numpy(), cmap="viridis")
            axes[row, c + 2].set_title(f"canal {c}", fontsize=8)
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "feature_maps.png", dpi=150)
    plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")  # téléchargé automatiquement au premier appel
    model.info()
    img_path = pick_image()
    print("Image utilisée :", img_path.name)

    baseline_inference(model, img_path)
    img, feats = extract_feature_maps(model, img_path)
    for i, f in feats.items():
        print(f"Couche {i} ({LAYERS[i]}) : {tuple(f.shape)}")
    plot_feature_maps(img, feats)


if __name__ == "__main__":
    main()