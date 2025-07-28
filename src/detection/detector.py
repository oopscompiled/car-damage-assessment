# https://docs.ultralytics.com/modes/predict/#__tabbed_1_2
# src/detection/detector.py

from ultralytics import YOLO

class Detector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.positive_classes =['normal_glass', 'normal_light', 'normal_tire']

    def predict(self, image, iou=0.3, conf=0.5, save_predict=False, filter_classes=True):
        results = self.model.predict(image, stream=True, conf=conf, iou=iou, verbose=False)

        all_results = []
        for i, result in enumerate(results):
            names = result.names
            boxes = result.boxes

            if filter_classes:

                class_ids = boxes.cls.cpu().numpy().astype(int)
                keep_indices = [
                    idx for idx, class_id in enumerate(class_ids)
                    if names[class_id] not in self.positive_classes
                ]
                boxes.data = boxes.data[keep_indices]
                result.boxes = boxes

            all_results.append({
                "result": result,
                "boxes": result.boxes,
                "probs": result.probs,
                "obb": result.obb,
                "masks": result.masks,
                "keypoints": result.keypoints,
                "orig_img": result.orig_img,
                "names": names,
                "plot_img": result.plot()
            })

            if save_predict:
                result.save(filename=f"detector_result_{i}.jpg")

        return all_results