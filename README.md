# Blood Cell Detector 🔬

AI-powered blood cell detection using YOLOv8, trained on the BCCD dataset.

## What it detects
- RBC — Red Blood Cells (Oxygen transport)
- WBC — White Blood Cells (Immune defense)  
- Platelets (Blood clotting)

## Model Performance
- mAP50: 91.75%
- mAP50-95: 69.54%

## How to run
```bash
pip install -r requirements.txt
python app.py
```

## Tech Stack
- YOLOv8 (Ultralytics)
- Gradio
- Trained on BCCD Dataset (Roboflow)
