import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.segmentation.segmenter import Segmenter
from src.detection.detector import Detector
import os
import cv2
import numpy as np

MODEL = ...

segmenter = MODEL('path/to/model_weights')

UNLABELED_DIR = 'path/to/unlabeled_dir'
NEW_LABELS_DIR = 'path/to/labeled_dir'
OUTPUT_VIS_DIR = "path/to/visual_inspect"

for img_name in os.listdir(UNLABELED_DIR):
    if not img_name.lower().endswith((".jpg", ".png")):
        continue

    print("->", img_name)
    img_path = os.path.join(UNLABELED_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Unable to load {img_name}")
        continue

    img = cv2.resize(img, (segmenter.input_size, segmenter.input_size))
    h, w = img.shape[:2]

    results = segmenter.predict(img)

    yolo_seg_lines = []
    for i, mask in enumerate(results["masks"]): # main loop
        cls_id = results["classes"][i]
        box = results["boxes"][i]
        normalized_coords = [f"{x / w:.6f} {y / h:.6f}" for x, y in mask]
        line = f"{cls_id} " + " ".join(normalized_coords)
        yolo_seg_lines.append(line)

        # visualizations
        pts = np.array(mask).astype(int)
        cv2.polylines(img, [pts], isClosed=True, color=(255, 0, 0), thickness=2) # B G R

        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        class_name = segmenter.class_names[cls_id]
        cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if yolo_seg_lines:
        base_name = os.path.splitext(img_name)[0]
        os.makedirs(NEW_LABELS_DIR, exist_ok=True)
        os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

        with open(os.path.join(NEW_LABELS_DIR, base_name + ".txt"), "w") as f:
            f.write("\n".join(yolo_seg_lines))
        cv2.imwrite(os.path.join(OUTPUT_VIS_DIR, base_name + ".jpg"), img)
