import cv2
import time
import threading
import os
import base64
import numpy as np
import queue
import torch
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from gtts import gTTS

app = Flask(__name__)
CORS(app)

# Chỉ detect đúng class nguy hiểm, bỏ qua 70+ class thừa (chỉ áp dụng cho yolov8n)
# door và stairs từ model phụ được merge sau nên không cần liệt kê ở đây
DANGER_CLASS_IDS = [0, 1, 2, 3, 5, 7, 13, 15, 16, 56, 60]
# person, bicycle, car, motorcycle, bus, truck, bench, cat, dog, chair, dining table

class CombinedModel:
    """Bọc nhiều model thành 1, gọi như YOLO bình thường."""
    def __init__(self):
        self._main = YOLO("yolov8n.pt")
        self.names = dict(self._main.names)

        # Thêm doors.pt nếu tồn tại
        self._door = None
        if os.path.exists("doors.pt"):
            self._door = YOLO("doors.pt")
            next_id = max(self.names.keys()) + 1
            self.names[next_id] = "door"
            self._door_id = next_id

        # Thêm stairs.pt nếu tồn tại
        self._stairs = None
        if os.path.exists("stairs.pt"):
            self._stairs = YOLO("stairs.pt")
            next_id = max(self.names.keys()) + 1
            self.names[next_id] = "stairs"
            self._stairs_id = next_id

    def __call__(self, frame, **kwargs):
        results = self._main(frame, **kwargs)

        if self._door is not None:
            door_results = self._door(frame, verbose=False)
            for box in door_results[0].boxes:
                box.cls[:] = self._door_id
                results[0].boxes = _merge_boxes(results[0].boxes, box)

        if self._stairs is not None:
            stairs_results = self._stairs(frame, verbose=False)
            for box in stairs_results[0].boxes:
                box.cls[:] = self._stairs_id
                results[0].boxes = _merge_boxes(results[0].boxes, box)

        return results

def _merge_boxes(base, extra):
    if base.data.shape[0] == 0:
        return extra
    base.data = torch.cat([base.data, extra.data], dim=0)
    return base

model = CombinedModel()

VIET_LABELS = {
    "person": "người", "car": "xe hơi", "motorcycle": "xe máy",
    "bicycle": "xe đạp", "bus": "xe buýt", "truck": "xe tải",
    "dog": "chó", "cat": "mèo", "bench": "ghế đá",
    "chair": "ghế", "dining table": "cái bàn", "door": "cửa",
    "stairs": "cầu thang",
}
DANGER = {"person", "car", "motorcycle", "bus", "truck", "bench",
          "bicycle", "cat", "dog", "chair", "dining table", "door", "stairs"}

MAX_DEVICES = 3
COOLDOWN = 4

os.makedirs("audio", exist_ok=True)

device_states = {}
device_lock = threading.Lock()
_next_id = 1

def get_device_state(device_id):
    with device_lock:
        if device_id not in device_states:
            state = {
                "infer_queue": queue.Queue(maxsize=1),
                "frame_queue": queue.Queue(maxsize=1),
                "latest_alert": {"message": "", "timestamp": 0, "label": ""},
                "last_spoken": {},
                "last_seen": {},
            }
            device_states[device_id] = state
            threading.Thread(target=inference_worker, args=(device_id,), daemon=True).start()
        return device_states[device_id]


def inference_worker(device_id):
    state = device_states[device_id]
    while True:
        frame = state["infer_queue"].get()

        frame_height, frame_width = frame.shape[:2]
        results = model(frame, classes=DANGER_CLASS_IDS, verbose=False)
        now = time.time()

        best_box = None
        best_label = None
        best_area = 0
        current_labels = set()

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            if label not in DANGER: continue
            current_labels.add(label)
            x1, y1, x2, y2 = box.xyxy[0]
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = box
                best_label = label

        last_seen = state["last_seen"]
        last_spoken = state["last_spoken"]
        latest_alert = state["latest_alert"]

        for lbl in current_labels:
            last_seen[lbl] = now
        for lbl in list(last_spoken.keys()):
            if now - last_seen.get(lbl, 0) > 2.0:
                del last_spoken[lbl]

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
                print(f"[Thiet bi {device_id}] {msg}")

        annotated = results[0].plot()
        h, w = annotated.shape[:2]
        cv2.line(annotated, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
        cv2.line(annotated, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
        cv2.putText(annotated, f"Thiet bi {device_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        fq = state["frame_queue"]
        if fq.full():
            try: fq.get_nowait()
            except queue.Empty: pass
        fq.put(annotated)

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
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400

    device_id = str(data.get("device_id", "1"))
    state = get_device_state(device_id)

    img_data = base64.b64decode(data['image'].split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    infer_q = state["infer_queue"]
    if infer_q.full():
        try: infer_q.get_nowait()
        except queue.Empty: pass
    infer_q.put(frame)

    return jsonify(state["latest_alert"])

@app.route("/audio/<path:key>")
def get_audio(key):
    path = f"audio/{key}.mp3"
    for _ in range(20):
        if os.path.exists(path):
            return send_file(path, mimetype="audio/mpeg")
        time.sleep(0.1)
    return "", 404

@app.route("/")
def index():
    from flask import redirect
    return redirect("/phone")

@app.route("/register", methods=["POST"])
def register():
    global _next_id
    with device_lock:
        assigned = str(_next_id)
        _next_id = (_next_id % MAX_DEVICES) + 1
    get_device_state(assigned)
    return jsonify({"device_id": assigned})

@app.route("/phone")
def phone():
    return send_from_directory("templates", "index.html")

@app.route("/monitor")
def monitor():
    return send_from_directory("templates", "monitor.html")

@app.route("/alert_state")
def alert_state():
    device_id = str(request.args.get("device_id", "1"))
    state = get_device_state(device_id)
    return jsonify(state["latest_alert"])

def start_flask():
    app.run(host="0.0.0.0", port=5000, ssl_context='adhoc', use_reloader=False)

# Ảnh placeholder màu tối khi chưa có thiết bị nào gửi frame
PLACEHOLDER_H, PLACEHOLDER_W = 240, 320

def make_placeholder(device_id):
    img = np.zeros((PLACEHOLDER_H, PLACEHOLDER_W, 3), dtype=np.uint8)
    cv2.putText(img, f"Thiet bi {device_id}", (20, PLACEHOLDER_H // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.putText(img, "Chua ket noi", (20, PLACEHOLDER_H // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    return img

if __name__ == "__main__":
    threading.Thread(target=start_flask, daemon=True).start()
    print("🚀 Server Flask đang chạy ngầm...")
    print("📱 Điện thoại 1: /phone?id=1")
    print("📱 Điện thoại 2: /phone?id=2")
    print("📱 Điện thoại 3: /phone?id=3")

    TARGET_H = 360

    while True:
        panels = []
        for did in ["1", "2", "3"]:
            state = get_device_state(did)
            try:
                frame = state["frame_queue"].get_nowait()
            except queue.Empty:
                frame = make_placeholder(did)

            # Resize giữ tỉ lệ chiều cao cố định
            h, w = frame.shape[:2]
            new_w = int(w * TARGET_H / h)
            frame = cv2.resize(frame, (new_w, TARGET_H))
            panels.append(frame)

        combined = np.hstack(panels)
        cv2.imshow("SmartEyes - 3 Thiet Bi", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

