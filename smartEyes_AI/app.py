import cv2
import numpy as np
import base64
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from ultralytics import YOLO

app = FastAPI()

# 1. LOAD MÔ HÌNH YOLOV8
model = YOLO('yolov8n.pt')

TARGET_CLASSES = {
    0: "người",
    1: "bàn học",
    2: "cửa ra vào",
    3: "bậc thang"
}

# 2. HỆ THỐNG COOLDOWN CẢNH BÁO
last_alert_time = {obj: 0 for obj in TARGET_CLASSES.values()}
ALERT_COOLDOWN = 5.0  # Chờ 5 giây trước khi cảnh báo lại cùng một vật thể


# 3. WEBSOCKET GIAO TIẾP THỜI GIAN THỰC
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Nhận frame ảnh từ Frontend
            data = await websocket.receive_text()
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # (Tùy chọn) Resize ảnh nhỏ lại trước khi predict để tăng FPS nếu vẫn chậm
            # frame = cv2.resize(frame, (640, 480))

            # Chạy nhận diện YOLOv8
            results = model.predict(frame, verbose=False)

            objects_to_alert = []
            current_time = time.time()

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])

                    if cls_id in TARGET_CLASSES:
                        obj_name = TARGET_CLASSES[cls_id]

                        # Vẽ bounding box lên ảnh
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, obj_name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Kiểm tra cooldown để đưa vào danh sách cảnh báo
                        if current_time - last_alert_time[obj_name] > ALERT_COOLDOWN:
                            objects_to_alert.append(obj_name)
                            last_alert_time[obj_name] = current_time

            # HIỂN THỊ CAMERA LÊN MÀN HÌNH LAPTOP
            cv2.imshow("Man hinh Nhan dien - Laptop", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Nhấn 'q' trên laptop để đóng cửa sổ nhận diện

            # Gửi TÍN HIỆU VĂN BẢN (JSON) về điện thoại thay vì gửi ảnh Base64
            if objects_to_alert:
                alert_data = json.dumps({"objects": objects_to_alert})
                await websocket.send_text(alert_data)

    except WebSocketDisconnect:
        print("Client đã ngắt kết nối")
    except Exception as e:
        print(f"Lỗi: {e}")
    finally:
        cv2.destroyAllWindows()


@app.get("/")
async def get():
    with open("new_index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())

# mở terminal : uvicorn main:app --host 0.0.0.0 --port 8000
# mở cmd chạy : ngrok http 8000

