import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Mở webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Detect
    results = model(frame)

    # Lấy tên object
    for box in results[0].boxes:

        cls_id = int(box.cls[0])

        name = model.names[cls_id]

        print(name)

    # Vẽ bounding box
    annotated = results[0].plot()

    # Hiển thị
    cv2.imshow("YOLOv8", annotated)

    # Nhấn q để thoát
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
