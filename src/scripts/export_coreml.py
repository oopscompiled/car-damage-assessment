from ultralytics import YOLO

model = YOLO('path/to/yolo_model')

def export_model(model, format="coreml", imgsz=960, half=True, device="mps"):
    return model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        device=device
    )

export_model(model, format="coreml", imgsz=960, half=True, device="mps")