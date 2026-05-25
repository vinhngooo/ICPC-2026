
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model('test2.jpg')

for box in results[0].boxes:

    cls_id = int(box.cls[0])

    name = model.names[cls_id]

    print(name)