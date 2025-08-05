# src/segmentation/segmenter.py
from ultralytics import YOLO
import numpy as np

# class Segmenter:
#     def __init__(self, model_path,input_size=1280):
#         self.model = YOLO(model_path)
#         self.input_size = input_size
#         self.class_names = self.model.names

#     def predict(self, img, conf=0.5):
#         if img is None:
#             raise ValueError("Image is None")

#         results = self.model.predict(img, conf=conf, verbose=False)[0]

#         masks, boxes, classes, confidences = [], [], [], []

#         if results.masks is not None:
#             for i, mask in enumerate(results.masks.xy):
#                 cls_id = int(results.boxes.cls[i].item())
#                 conf = float(results.boxes.conf[i].item())

#                 masks.append(mask)
#                 boxes.append(results.boxes.xyxy[i].cpu().numpy())
#                 classes.append(cls_id)
#                 confidences.append(conf)

#         return {
#             "masks": masks,
#             "boxes": boxes,
#             "classes": classes,
#             "confidences": confidences
#         }

# class Segmenter:
#     def __init__(self, model_path, input_size=1280):
#         self.model = YOLO(model_path)
#         self.input_size = input_size
#         self.class_names = self.model.names

#     def predict(self, image, iou=0.3, conf=0.5, filter_classes=True): # dummy argument for inference class
        
#         if image is None:
#             raise ValueError("Image is None")

#         results = self.model.predict(image, iou=iou, conf=conf, verbose=False)[0]

#         boxes, classes, confidences = [], [], []

#         if results.boxes is not None:
#             for i in range(len(results.boxes)):
#                 cls_id = int(results.boxes.cls[i].item())
#                 conf_score = float(results.boxes.conf[i].item())
#                 box = results.boxes.xyxy[i].cpu().numpy()

#                 boxes.append(box)
#                 classes.append(cls_id)
#                 confidences.append(conf_score)

#         return {
#             "boxes": boxes,
#             "classes": classes,
#             "confidences": confidences
#         }

# class Segmenter:
#     def __init__(self, model_path, input_size=1280):
#         self.model = YOLO(model_path)
#         self.input_size = input_size
#         self.class_names = self.model.names

#     def predict(self, image, iou=0.3, conf=0.5, filter_classes=True):
#         if image is None:
#             raise ValueError("Image is None")

#         results = self.model.predict(image, iou=iou, conf=conf, verbose=False)[0]

#         boxes, classes, confidences, masks = [], [], [], []

#         if results.boxes is not None:

#             for i in range(len(results.boxes)):

#                 cls_id = int(results.boxes.cls[i].item())
#                 conf_score = float(results.boxes.conf[i].item())
#                 box = results.boxes.xyxy[i].cpu().numpy()

#                 boxes.append(box)
#                 classes.append(cls_id)
#                 confidences.append(conf_score)

#                 if results.masks is not None:
#                     # results.masks.data — torch.Tensor [N, H, W]
#                     mask = results.masks.data[i].cpu().numpy().astype(np.uint8)
#                     masks.append(mask)
#                 else:
#                     masks.append(None)

#         return {
#             "boxes": boxes,
#             "classes": classes,
#             "confidences": confidences,
#             "masks": masks
#         }

class Segmenter:
    def __init__(self, model_path, input_size=1280):
        self.model = YOLO(model_path)
        self.input_size = input_size
        self.class_names = self.model.names

    def predict(self, image, iou=0.3, conf=0.5, filter_classes=True):
        if image is None:
            raise ValueError("Image is None")

        results = self.model.predict(image, iou=iou, conf=conf, verbose=False)[0]

        boxes, classes, confidences, masks = [], [], [], []

        if results.boxes is not None:
            for i in range(len(results.boxes)):
                cls_id = int(results.boxes.cls[i].item())
                conf_score = float(results.boxes.conf[i].item())
                box = results.boxes.xyxy[i].cpu().numpy()

                boxes.append(box)
                classes.append(cls_id)
                confidences.append(conf_score)

            if results.masks is not None:
                for m in results.masks.xy:
                    masks.append(np.array(m, dtype=np.int32))
            else:
                masks = [None] * len(boxes)

        return {
            "boxes": boxes,
            "classes": classes,
            "confidences": confidences,
            "masks": masks
        }