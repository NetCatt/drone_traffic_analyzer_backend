import glob
import os
import cv2
from ultralytics import YOLO
from collections import Counter

files = sorted(glob.glob("temp/*_input.mp4"), key=os.path.getmtime, reverse=True)
p = files[0] if files else None

cap = cv2.VideoCapture(p)
ok, frame = cap.read()
cap.release()

# Test both models
for model_name in ["yolov8n.pt", "yolov8m.pt"]:
    print(f"\n=== {model_name} ===")
    m = YOLO(model_name)
    r = m.predict(frame, conf=0.15, iou=0.4, imgsz=1408, verbose=False)[0]
    boxes = r.boxes

    if boxes and len(boxes) > 0:
        classes = boxes.cls.cpu().numpy().astype(int)
        cls_counts = Counter(classes)
        print(f"Total detections: {len(boxes)}")
        print("Class breakdown:")
        
        names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
            8: "boat",
            9: "traffic light",
            10: "fire hydrant",
        }
        
        for cls_id in sorted(cls_counts.keys()):
            name = names.get(cls_id, f"class_{cls_id}")
            print(f"  {cls_id}: {name} -> {cls_counts[cls_id]}")
        
        vehicle_classes = {2, 3, 5, 7}
        vehicle_count = sum(1 for c in classes if c in vehicle_classes)
        print(f"Vehicle classes {vehicle_classes}: {vehicle_count}")
    else:
        print("No detections")
