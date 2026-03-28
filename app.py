
import gradio as gr
from ultralytics import YOLO
from PIL import Image
import numpy as np

model = YOLO('best.pt')  # make sure best.pt is in the same folder

CELL_INFO = {
    "RBC":      {"full": "Red Blood Cells",   "role": "Oxygen transport"},
    "WBC":      {"full": "White Blood Cells",  "role": "Immune defense"},
    "Platelets":{"full": "Platelets",          "role": "Blood clotting"},
}

def predict(img):
    if img is None:
        return None, "Please upload an image."
    image = Image.fromarray(img).convert("RGB")
    results = model(image, conf=0.25)
    annotated = results[0].plot()[:, :, ::-1]
    boxes = results[0].boxes
    names = results[0].names
    counts = {}
    for cls_id in boxes.cls.tolist():
        label = names[int(cls_id)]
        counts[label] = counts.get(label, 0) + 1
    if not counts:
        summary = "No cells detected."
    else:
        lines = ["**Detection Summary**\n"]
        for cell, count in counts.items():
            info = CELL_INFO.get(cell, {"full": cell, "role": ""})
            lines.append(f"**{info['full']} ({cell})** — {count} detected\n_{info['role']}_")
        lines.append(f"\n**Total: {sum(counts.values())} cells**")
        summary = "\n\n".join(lines)
    return annotated, summary

gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload Blood Smear"),
    outputs=[gr.Image(label="Result"), gr.Markdown(label="Summary")],
    title="Blood Cell Detector"
).launch()
