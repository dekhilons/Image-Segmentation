# Brain Tumor Detection using YOLO & SAM

This repository provides an end-to-end pipeline for **brain tumor detection** using **YOLO** for object detection and **SAM** for segmentation. The dataset is sourced from [Roboflow](https://universe.roboflow.com/brain-tumor-detection-wsera/tumor-detection-ko5jp/dataset/8).

## Installation
```bash
pip install --upgrade ultralytics roboflow
```

## Dataset Download
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
dataset = rf.workspace("brain-tumor-detection-wsera").project("tumor-detection-ko5jp").version(8).download("yolov11")
```

## Training YOLO Model
```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="Tumor-Detection-8/data.yaml", epochs=20, imgsz=640, device=0)
```

## Inference
```python
model = YOLO("runs/detect/train/weights/best.pt")
results = model("test.jpeg", save=True)
```

## Segmentation with SAM
```python
from ultralytics import SAM
sam_model = SAM("sam2_b.pt")
sam_results = sam_model(results[0].orig_img, bboxes=results[0].boxes.xyxy, save=True, device=0)
```

## Results
- **YOLO Detection:** `runs/detect/train/results.png`
- **SAM Segmentation:** `runs/segment/predict/image0.jpg`

## Acknowledgments
- **Roboflow** for dataset hosting  
- **Ultralytics** for YOLO  
- **Meta AI** for SAM  
