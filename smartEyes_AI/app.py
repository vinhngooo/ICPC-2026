import cv2
import time
import threading
import os
import base64
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from gtts import gTTS

app = Flask(__name__)
CORS(app)
model = YOLO("yolov8n.pt")

VIET_LABELS = {
    "person": "người", "car": "xe hơi", "motorcycle": "xe máy",
    "bicycle": "xe đạp", "bus": "xe buýt", "truck": "xe tải",
    "dog": "chó", "cat": "mèo"
}
DANGER = {"person", "car", "motorcycle", "bus", "truck"}

latest_alert = {"message": "", "timestamp": 0, "label": ""}
last_spoken = {}
COOLDOWN = 4

os.makedirs("audio", exist_ok=True)

def get_direction(box, frame_width):
    x1, y1, x2, y2 = box.xyxy[0]
    center_x = (x1 + x2) / 2
    ratio = center_x / frame_width
    if ratio < 0.35: return "bên trái"
    elif ratio > 0.65: return "bên phải"
    else: return "phía trước"

def get_distance(box, frame_width, frame_height):
    x1, y1, x2, y2 = box.xyxy[0]
    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    ratio = box_area / frame_area
    if ratio > 0.25: return "rất gần"
    elif ratio > 0.08: return "gần"
    else: return ""

def build_message(label, direction, distance):
    name = VIET_LABELS.get(label, label)
    if distance: return f"Có {name} {distance} {direction}, chú ý!"
    else: return f"Có {name} {direction}"

def generate_audio(key, text):
    path = f"audio/{key}.mp3"
    if not os.path.exists(path):
        gTTS(text=text, lang="vi").save(path)

@app.route("/process_frame", methods=["POST"])
def process_frame():
    global latest_alert
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400

    # Giải mã hình ảnh từ trình duyệt gửi lên
    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    frame_height, frame_width = frame.shape[:2]
    results = model(frame, verbose=False)
    now = time.time()

    best_box = None
    best_label = None
    best_area = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label not in DANGER: continue

        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2 - x1) * (y2 - y1)
        if area > best_area:
            best_area = area
            best_box = box
            best_label = label

    if best_box is not None:
        if now - last_spoken.get(best_label, 0) > COOLDOWN:
            direction = get_direction(best_box, frame_width)
            distance = get_distance(best_box, frame_width, frame_height)
            msg = build_message(best_label, direction, distance)

            audio_key = f"{best_label}_{direction.replace(' ', '_')}_{distance.replace(' ', '_')}"
            threading.Thread(target=generate_audio, args=(audio_key, msg), daemon=True).start()

            latest_alert["message"] = msg
            latest_alert["timestamp"] = now
            latest_alert["label"] = audio_key
            last_spoken[best_label] = now
            print(f"🔊 {msg}")

    return jsonify(latest_alert)

@app.route("/audio/<path:key>")
def get_audio(key):
    path = f"audio/{key}.mp3"
    for _ in range(20):
        if os.path.exists(path):
            return send_file(path, mimetype="audio/mpeg")
        time.sleep(0.1)
    return "", 404

@app.route("/phone")
def phone():
    return send_from_directory(".", "templates/smarteyes_ui.html")

if __name__ == "__main__":
    # Thêm ssl_context='adhoc' để bắt buộc chạy HTTPS
    app.run(host="0.0.0.0", port=5000, ssl_context='adhoc')



# run link on phone : https://192.168.52.104:5000/phone
