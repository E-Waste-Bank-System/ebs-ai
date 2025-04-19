# E-Waste Object Classification Model

This repository contains a YOLOv11-based object classification model for electronic waste (e-waste) items.

## Model Details

- **Framework**: YOLOv11
- **Classes**: 10 e-waste categories
- **Input**: RGB images
- **Output**: Bounding boxes and class predictions

### Classes
1. Battery
2. Keyboard
3. Microwave
4. Mobile
5. Mouse
6. PCB
7. Player
8. Printer
9. Television
10. Washing Machine

## Directory Structure

```
object-classification-model/
├── data/               # Dataset (not included in repo)
│   ├── images/        # Training and validation images
│   └── labels/        # Annotation files
├── train/             # Training configuration
│   └── e-waste.yaml   # Dataset configuration
└── venv/              # Python virtual environment (not included)
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
- Place training images in `data/images/train`
- Place validation images in `data/images/val`
- Place corresponding label files in `data/labels`

## Training

To train the model:
```bash
yolo detect train model=yolov11n.pt data=train/e-waste.yaml epochs=100 imgsz=640
```

## Inference

To run inference on new images:
```bash
yolo predict model=path/to/best.pt source=path/to/image.jpg
```

## Notes

- Download Dataset [here](https://www.kaggle.com/datasets/akshat103/e-waste-image-dataset)
- Model weights and checkpoints should be stored separately
- Use the provided YAML configuration for training
