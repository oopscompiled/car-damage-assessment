# src/segmentation/segmenter.py
from ultralytics import YOLO

class Segmenter:
    def __init__(self, model_path,input_size=1280):
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.class_names = self.model.names

    def predict(self, img, conf=0.5):
        if img is None:
            raise ValueError("Image is None")

        results = self.model.predict(img, conf=conf, verbose=False)[0]

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
