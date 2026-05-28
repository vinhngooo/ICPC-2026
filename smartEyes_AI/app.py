import cv2
import time
import threading
import os
from flask import Flask, jsonify, send_file, render_template_string
from flask_cors import CORS
from ultralytics import YOLO
from gtts import gTTS
from flask import send_from_directory

app = Flask(__name__)
CORS(app)
model = YOLO("yolov8n.pt")

VIET_LABELS = {
    "person": "người",
    "car": "xe hơi",
    "motorcycle": "xe máy",
    "bicycle": "xe đạp",
    "bus": "xe buýt",
    "truck": "xe tải",
    "dog": "chó",
    "cat": "mèo",
}
DANGER = {"person", "car", "motorcycle", "bus", "truck"}

latest_alert = {"message": "", "timestamp": 0, "label": ""}
last_spoken = {}
COOLDOWN = 4

os.makedirs("audio", exist_ok=True)

def get_direction(box, frame_width):
    """Tính hướng dựa theo tâm bounding box"""
    x1, y1, x2, y2 = box.xyxy[0]
    center_x = (x1 + x2) / 2
    ratio = center_x / frame_width

    if ratio < 0.35:
        return "bên trái"
    elif ratio > 0.65:
        return "bên phải"
    else:
        return "phía trước"

def get_distance(box, frame_width, frame_height):
    """Ước tính khoảng cách dựa theo diện tích bounding box"""
    x1, y1, x2, y2 = box.xyxy[0]
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    ratio = box_area / frame_area

    if ratio > 0.25:
        return "rất gần"
    elif ratio > 0.08:
        return "gần"
    else:
        return ""  # Xa thì không cần nói khoảng cách

def build_message(label, direction, distance):
    """Ghép câu cảnh báo hoàn chỉnh"""
    name = VIET_LABELS.get(label, label)
    if distance:
        return f"Có {name} {distance} {direction}, chú ý!"
    else:
        return f"Có {name} {direction}"

def generate_audio(key, text):
    path = f"audio/{key}.mp3"
    if not os.path.exists(path):
        gTTS(text=text, lang="vi").save(path)
    return path

def run_yolo():
    cap = cv2.VideoCapture(1)  # index iVCam của bạn
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame, verbose=False)
        now = time.time()

        # Tìm vật thể nguy hiểm GẦN NHẤT để ưu tiên cảnh báo
        best_box = None
        best_label = None
        best_area = 0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label not in DANGER:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = box
                best_label = label

        if best_box is not None:
            key = f"{best_label}_{int(now // COOLDOWN)}"
            if now - last_spoken.get(best_label, 0) > COOLDOWN:
                direction = get_direction(best_box, frame_width)
                distance = get_distance(best_box, frame_width, frame_height)
                msg = build_message(best_label, direction, distance)

                # Generate audio nếu chưa có
                audio_key = f"{best_label}_{direction.replace(' ', '_')}_{distance.replace(' ', '_')}"
                threading.Thread(
                    target=generate_audio,
                    args=(audio_key, msg),
                    daemon=True
                ).start()

                latest_alert["message"] = msg
                latest_alert["timestamp"] = now
                latest_alert["label"] = audio_key
                last_spoken[best_label] = now
                print(f"🔊 {msg}")

        # Vẽ vùng chia 3 lên màn hình để debug
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
        cv2.line(frame, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
        cv2.putText(frame, "TRAI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, "GIUA", (w//2 - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, "PHAI", (2*w//3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        annotated = results[0].plot()
        # Vẽ đường chia lên frame annotated
        cv2.line(annotated, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
        cv2.line(annotated, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
        cv2.imshow("SmartEyes", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/alert")
def get_alert():
    return jsonify(latest_alert)

@app.route("/audio/<path:key>")
def get_audio(key):
    path = f"audio/{key}.mp3"
    # Nếu file chưa kịp generate thì chờ tối đa 2s
    for _ in range(20):
        if os.path.exists(path):
            return send_file(path, mimetype="audio/mpeg")
        time.sleep(0.1)
    return "", 404



@app.route("/phone")
def phone():
    return send_from_directory(".", "smarteyes_ui.html")

if __name__ == "__main__":
    threading.Thread(target=run_yolo, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)
