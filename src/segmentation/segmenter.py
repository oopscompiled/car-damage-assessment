# src/segmentation/segmenter.py
import cv2
from ultralytics import YOLO

class Segmenter:
    def __init__(self, model_path, conf_thresh=0.5, input_size=1280):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.input_size = input_size
        self.class_names = self.model.names

    def predict(self, img):
        if img is None:
            raise ValueError("Image is None")

        results = self.model.predict(img, conf=self.conf_thresh, verbose=False)[0]

        masks, boxes, classes, confidences = [], [], [], []

        if results.masks is not None:
            for i, mask in enumerate(results.masks.xy):
                cls_id = int(results.boxes.cls[i].item())
                conf = float(results.boxes.conf[i].item())

                masks.append(mask)
                boxes.append(results.boxes.xyxy[i].cpu().numpy())
                classes.append(cls_id)
                confidences.append(conf)

        return {
            "masks": masks,
            "boxes": boxes,
            "classes": classes,
            "confidences": confidences
        }
