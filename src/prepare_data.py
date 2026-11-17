"""Étape 1 : exploration, prétraitement (416x416) et conversion Pascal VOC -> YOLO."""
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "yolo"
FIG = ROOT / "reports" / "figures"

IMG_SIZE = 416
CLASSES = ["helmet", "head", "person"]          # à adapter après l'exploration
SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
SEED = 42
IMG_EXT = {".png", ".jpg", ".jpeg"}


def index_images():
    return {p.stem: p for p in RAW.rglob("*") if p.suffix.lower() in IMG_EXT}


def parse_xml(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bb = obj.find("bndbox")
        xmin, ymin, xmax, ymax = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((name, xmin, ymin, xmax, ymax))
    return w, h, boxes


def load_annotations():
    images = index_images()
    records = []
    for xml in sorted(RAW.rglob("*.xml")):
        if xml.stem not in images:
            continue
        w, h, boxes = parse_xml(xml)
        records.append({"stem": xml.stem, "img": images[xml.stem], "w": w, "h": h, "boxes": boxes})
    return records


def explore(records):
    FIG.mkdir(parents=True, exist_ok=True)
    rows = [(r["stem"], b[0]) for r in records for b in r["boxes"]]
    df = pd.DataFrame(rows, columns=["stem", "classe"])
    print(f"Images annotées : {len(records)}")
    print(f"Boîtes totales  : {len(df)}")
    print("\nRépartition des classes :")
    print(df["classe"].value_counts())
    print("\nBoîtes par image :")
    print(df.groupby("stem").size().describe())
    print("\nRésolutions les plus fréquentes :")
    print(pd.Series([(r["w"], r["h"]) for r in records]).value_counts().head(5))

    df["classe"].value_counts().plot(kind="bar", title="Répartition des classes")
    plt.ylabel("Nombre de boîtes")
    plt.tight_layout()
    plt.savefig(FIG / "class_distribution.png", dpi=150)
    plt.close()


def preview(records, n=6):
    random.seed(SEED)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, r in zip(axes.ravel(), random.sample(records, min(n, len(records)))):
        img = cv2.cvtColor(cv2.imread(str(r["img"])), cv2.COLOR_BGR2RGB)
        for name, x1, y1, x2, y2 in r["boxes"]:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(img, name, (int(x1), max(int(y1) - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        ax.imshow(img)
        ax.set_title(r["stem"])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIG / "samples_annotated.png", dpi=150)
    plt.close()


def to_yolo(w, h, x1, y1, x2, y2):
    x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
    y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
    return ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)


def convert(records):
    random.seed(SEED)
    random.shuffle(records)
    n = len(records)
    n_train, n_val = int(n * SPLITS["train"]), int(n * SPLITS["val"])
    parts = {
        "train": records[:n_train],
        "val": records[n_train:n_train + n_val],
        "test": records[n_train + n_val:],
    }
    cls_id = {c: i for i, c in enumerate(CLASSES)}

    for split, items in parts.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for r in tqdm(items, desc=f"{split} ({len(items)})"):
            img = cv2.imread(str(r["img"]))
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(OUT / "images" / split / f"{r['stem']}.jpg"), img)

            lines = []
            for name, x1, y1, x2, y2 in r["boxes"]:
                if name not in cls_id:
                    continue
                xc, yc, bw, bh = to_yolo(r["w"], r["h"], x1, y1, x2, y2)
                if bw > 0 and bh > 0:
                    lines.append(f"{cls_id[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (OUT / "labels" / split / f"{r['stem']}.txt").write_text("\n".join(lines))

    yaml = (
        f"path: {OUT.as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASSES)}\nnames: {CLASSES}\n"
    )
    (OUT / "data.yaml").write_text(yaml)
    print("data.yaml écrit dans", OUT / "data.yaml")


def main():
    records = load_annotations()
    if not records:
        raise SystemExit("Aucune annotation trouvée dans data/raw. Vérifie le téléchargement.")
    explore(records)
    preview(records)
    convert(records)


if __name__ == "__main__":
    main()