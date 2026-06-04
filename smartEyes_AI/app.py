import cv2
import io
import time
import webbrowser
import threading
import os
import base64
import numpy as np
import queue
import asyncio
import torch
import edge_tts
from collections import deque
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

DANGER_CLASS_IDS = [0, 1, 2, 3, 5, 7, 13, 15, 16, 56, 60]

class CombinedModel:
    """Bọc nhiều model thành 1, gọi như YOLO bình thường."""
    def __init__(self):
        self._main = YOLO("yolov8s.pt")
        self.names = dict(self._main.names)

        self._door = None
        if os.path.exists("doors.pt"):
            self._door = YOLO("doors.pt")
            next_id = max(self.names.keys()) + 1
            self.names[next_id] = "door"
            self._door_id = next_id

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
                box.data = box.data.clone()
                box.data[:, 5] = self._door_id
                results[0].boxes = _merge_boxes(results[0].boxes, box)

        if self._stairs is not None:
            stairs_results = self._stairs(frame, verbose=False)
            for box in stairs_results[0].boxes:
                box.data = box.data.clone()
                box.data[:, 5] = self._stairs_id
                results[0].boxes = _merge_boxes(results[0].boxes, box)

        results[0].names = self.names
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

PRIORITY = {
    "car": 5, "motorcycle": 5, "bus": 5, "truck": 5,
    "bicycle": 4,
    "person": 3,
    "dog": 2, "cat": 2,
    "stairs": 1, "door": 1, "bench": 1, "chair": 1, "dining table": 1,
}

# Chiều cao thực tế (cm) dùng để ước lượng khoảng cách
KNOWN_HEIGHTS_CM = {
    "person": 170, "car": 150, "motorcycle": 110, "bus": 280, "truck": 250,
    "bicycle": 100, "dog": 50, "cat": 30, "bench": 90, "chair": 90,
    "dining table": 75, "door": 200, "stairs": 100,
}
# Tiêu cự ước tính cho camera điện thoại ở độ phân giải ~720p
FOCAL_LENGTH_PX = 600

DISTANCES = ["dưới 1 mét", "khoảng 1 mét", "khoảng 2 mét", ""]

MAX_DEVICES = 3
COOLDOWN = 6

APPROACH_RATIO    = 1.25
APPROACH_COOLDOWN = 3
TRACK_WINDOW      = 4

os.makedirs("audio", exist_ok=True)

# ---------------------------------------------------------------------------
# Simple IoU tracker (không cần thư viện ngoài)
# ---------------------------------------------------------------------------

def _compute_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

class SimpleTracker:
    """Gán track_id bền vững cho các detection bằng IoU matching."""
    def __init__(self, iou_threshold=0.3, max_lost=5):
        self._tracks  = {}   # track_id -> {"box": [x1,y1,x2,y2], "lost": int}
        self._next_id = 1
        self._iou_thr = iou_threshold
        self._max_lost = max_lost

    def update(self, detections):
        """detections: list of [x1,y1,x2,y2, label, area]
        Trả về list of [x1,y1,x2,y2, label, area, track_id]
        """
        if not detections:
            for tid in list(self._tracks):
                self._tracks[tid]["lost"] += 1
                if self._tracks[tid]["lost"] > self._max_lost:
                    del self._tracks[tid]
            return []

        track_ids  = list(self._tracks.keys())
        track_boxes = [self._tracks[tid]["box"] for tid in track_ids]

        matched_det  = {}   # det_idx  -> track_id
        matched_trk  = set()

        for i, det in enumerate(detections):
            best_iou, best_j = self._iou_thr, None
            for j, tbox in enumerate(track_boxes):
                if j in matched_trk:
                    continue
                iou = _compute_iou(det[:4], tbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None:
                matched_det[i] = track_ids[best_j]
                matched_trk.add(best_j)

        new_tracks = {}
        result = []
        for i, det in enumerate(detections):
            if i in matched_det:
                tid = matched_det[i]
            else:
                tid = self._next_id
                self._next_id += 1
            new_tracks[tid] = {"box": det[:4], "lost": 0}
            result.append(det + [tid])

        # Giữ track bị mất tạm thời
        for j, tid in enumerate(track_ids):
            if j not in matched_trk:
                trk = self._tracks[tid]
                trk["lost"] += 1
                if trk["lost"] <= self._max_lost:
                    new_tracks[tid] = trk

        self._tracks = new_tracks
        return result

# ---------------------------------------------------------------------------

device_states = {}
device_lock = threading.Lock()
_next_id = 1

def get_device_state(device_id):
    with device_lock:
        if device_id not in device_states:
            state = {
                "infer_queue":          queue.Queue(maxsize=1),
                "latest_alert":         {"message": "", "timestamp": 0, "label": ""},
                "last_spoken":          {},
                "last_seen":            {},
                "area_history":         {},
                "last_approach_spoken": {},
                "latest_frame":         None,
                "frame_lock":           threading.Lock(),
                "tracker":              SimpleTracker(),
            }
            device_states[device_id] = state
            threading.Thread(target=inference_worker, args=(device_id,), daemon=True).start()
        return device_states[device_id]


def inference_worker(device_id):
    state = device_states[device_id]
    while True:
        frame = state["infer_queue"].get()
        try:
            _do_inference(device_id, state, frame)
        except Exception as e:
            import traceback
            print(f"[ERROR device {device_id}] {e}")
            traceback.print_exc()

def _do_inference(device_id, state, frame):
    frame_width = frame.shape[1]
    results = model(frame, classes=DANGER_CLASS_IDS, verbose=False)
    now = time.time()

    # Thu thập tất cả detection nguy hiểm
    raw_dets = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        if label not in DANGER:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        area = (x2 - x1) * (y2 - y1)
        raw_dets.append([x1, y1, x2, y2, label, area])
        state["last_seen"][label] = now

    # Gán track ID bền vững
    tracked = state["tracker"].update(raw_dets)
    # tracked: [[x1,y1,x2,y2, label, area, track_id], ...]

    # Sắp xếp theo priority cao → thấp, rồi area lớn → nhỏ
    tracked.sort(key=lambda d: (PRIORITY.get(d[4], 0), d[5]), reverse=True)

    # Cập nhật area_history và dọn entry của track đã hết hạn
    area_history = state["area_history"]
    alive_tks = {f"t{tid}" for tid in state["tracker"]._tracks}
    for tk in list(area_history.keys()):
        if tk not in alive_tks:
            del area_history[tk]
    for det in tracked:
        tk = f"t{det[6]}"
        if tk not in area_history:
            area_history[tk] = deque(maxlen=TRACK_WINDOW)
        area_history[tk].append(det[5])

    last_spoken          = state["last_spoken"]
    last_approach_spoken = state["last_approach_spoken"]
    latest_alert         = state["latest_alert"]

    # Dọn cooldown hết hạn
    for lbl in list(last_spoken.keys()):
        if now - state["last_seen"].get(lbl, 0) > 2.0:
            del last_spoken[lbl]

    # --- Kiểm tra vật thể đang tiến lại (ưu tiên cao nhất, 1 cảnh báo) ---
    approach_fired = False
    for det in tracked[:3]:
        x1, y1, x2, y2, label, area, tid = det
        tk   = f"t{tid}"
        hist = area_history.get(tk, deque())
        is_approaching = (
            len(hist) == TRACK_WINDOW
            and hist[-1] > hist[0] * APPROACH_RATIO
            and hist[-1] > hist[-2]
        )
        if is_approaching and now - last_approach_spoken.get(label, 0) > APPROACH_COOLDOWN:
            direction = get_direction(x1, x2, frame_width)
            msg       = build_approach_message(label, direction)
            audio_key = f"approach_{label}_{direction.replace(' ', '_')}"
            threading.Thread(target=generate_audio, args=(audio_key, msg), daemon=True).start()
            latest_alert.update({"message": msg, "timestamp": now, "label": audio_key})
            last_approach_spoken[label] = now
            last_spoken[label]          = now
            print(f"[TIEP CAN device {device_id}] {msg}")
            approach_fired = True
            break

    # --- Cảnh báo đa vật thể (tối đa 2 object hết cooldown) ---
    if not approach_fired:
        due_objects = []
        for det in tracked:
            x1, y1, x2, y2, label, _, tid = det
            if now - last_spoken.get(label, 0) > COOLDOWN:
                direction = get_direction(x1, x2, frame_width)
                distance  = get_distance(y1, y2, label)
                due_objects.append((label, direction, distance))
                if len(due_objects) >= 2:
                    break

        if due_objects:
            msg = build_multi_message(due_objects)
            if len(due_objects) == 1:
                label, direction, distance = due_objects[0]
                audio_key = f"{label}_{direction.replace(' ', '_')}_{distance.replace(' ', '_')}"
            else:
                audio_key = "multi_" + "_".join(d[0] for d in due_objects)
            threading.Thread(target=generate_audio, args=(audio_key, msg), daemon=True).start()
            latest_alert.update({"message": msg, "timestamp": now, "label": audio_key})
            for label, _, _ in due_objects:
                last_spoken[label] = now
            print(f"[Thiet bi {device_id}] {msg}")

    annotated = results[0].plot()
    h, w = annotated.shape[:2]
    cv2.line(annotated, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
    cv2.line(annotated, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
    cv2.putText(annotated, f"Thiet bi {device_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    with state["frame_lock"]:
        state["latest_frame"] = buf.tobytes()

def get_direction(x1, x2, frame_width):
    ratio = ((x1 + x2) / 2) / frame_width
    if ratio < 0.35:   return "bên trái"
    elif ratio > 0.65: return "bên phải"
    else:              return "phía trước"

def get_distance(y1, y2, label):
    """Ước lượng khoảng cách thực tế dựa vào chiều cao bounding box."""
    box_h = float(y2 - y1)
    if box_h < 10:
        return ""
    real_h  = KNOWN_HEIGHTS_CM.get(label, 150)
    dist_cm = (real_h * FOCAL_LENGTH_PX) / box_h
    if dist_cm < 100:   return "dưới 1 mét"
    elif dist_cm < 200: return "khoảng 1 mét"
    elif dist_cm < 300: return "khoảng 2 mét"
    else:               return ""

def get_traffic_light_color(frame, box):
    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    red = cv2.countNonZero(cv2.bitwise_or(
        cv2.inRange(hsv, (0,   120, 120), (10,  255, 255)),
        cv2.inRange(hsv, (160, 120, 120), (180, 255, 255))
    ))
    green = cv2.countNonZero(cv2.inRange(hsv, (40, 80, 80), (90, 255, 255)))
    if red == 0 and green == 0:
        return None
    return "red" if red >= green else "green"

def build_message(label, direction, distance):
    name = VIET_LABELS.get(label, label)
    if distance: return f"Có {name} {distance} {direction}, chú ý!"
    else:        return f"Có {name} {direction}"

def build_approach_message(label, direction):
    name = VIET_LABELS.get(label, label)
    return f"Cẩn thận! {name} đang tiến lại {direction}!"

def build_multi_message(objects):
    """objects: list of (label, direction, distance)"""
    if len(objects) == 1:
        return build_message(*objects[0])
    parts = []
    for label, direction, distance in objects:
        name = VIET_LABELS.get(label, label)
        parts.append(f"{name} {distance} {direction}".strip() if distance
                     else f"{name} {direction}")
    return "Chú ý! " + " và ".join(parts) + "!"

VOICE = "vi-VN-HoaiMyNeural"
RATE  = "+15%"

# Event loop bền vững cho toàn bộ audio — tránh conflict asyncio/Flask trên Windows
_audio_loop = asyncio.new_event_loop()
threading.Thread(target=_audio_loop.run_forever, daemon=True, name="audio-loop").start()

async def _gen_audio_async(key: str, text: str, sem: asyncio.Semaphore = None):
    path = f"audio/{key}.mp3"
    if os.path.exists(path):
        return

    async def _save():
        for attempt in range(3):
            try:
                await edge_tts.Communicate(text, VOICE, rate=RATE).save(path)
                return
            except Exception:
                if attempt < 2:
                    await asyncio.sleep(1 + attempt)
        print(f"[Audio] Bỏ qua {key} (sẽ tạo lúc cần)")

    if sem:
        async with sem:
            await _save()
    else:
        await _save()

def generate_audio(key: str, text: str):
    asyncio.run_coroutine_threadsafe(_gen_audio_async(key, text), _audio_loop)

def pregenerate_audio():
    directions = ["bên trái", "bên phải", "phía trước"]

    async def _run():
        sem = asyncio.Semaphore(5)  # tối đa 5 request đồng thời đến TTS service
        tasks = [
            _gen_audio_async(
                f"{label}_{direction.replace(' ','_')}_{distance.replace(' ','_')}",
                build_message(label, direction, distance),
                sem
            )
            for label in VIET_LABELS
            for direction in directions
            for distance in DISTANCES
        ] + [
            _gen_audio_async(
                f"approach_{label}_{direction.replace(' ','_')}",
                build_approach_message(label, direction),
                sem
            )
            for label in VIET_LABELS
            for direction in directions
        ]
        await asyncio.gather(*tasks)
        print(f"[Audio] Pre-generated {len(tasks)} clips xong.")

    asyncio.run_coroutine_threadsafe(_run(), _audio_loop).result()

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

@app.route("/frame/<device_id>")
def get_frame(device_id):
    state = get_device_state(device_id)
    with state["frame_lock"]:
        data = state["latest_frame"]
    if data is None:
        placeholder = make_placeholder(device_id)
        _, buf = cv2.imencode('.jpg', placeholder)
        data = buf.tobytes()
    return send_file(io.BytesIO(data), mimetype='image/jpeg')

PLACEHOLDER_H, PLACEHOLDER_W = 240, 320

def make_placeholder(device_id):
    img = np.zeros((PLACEHOLDER_H, PLACEHOLDER_W, 3), dtype=np.uint8)
    cv2.putText(img, f"Thiet bi {device_id}", (20, PLACEHOLDER_H // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 80), 2)
    cv2.putText(img, "Chua ket noi", (20, PLACEHOLDER_H // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    return img

if __name__ == "__main__":
    import shutil
    shutil.rmtree("audio", ignore_errors=True)
    os.makedirs("audio", exist_ok=True)
    print("[Audio] Dang tao truoc tat ca audio clips (edge-tts)...")
    pregenerate_audio()
    print("Server chay tai: https://192.168.62.197:5000")
    print("Dien thoai 1: /phone?id=1")
    print("Dien thoai 2: /phone?id=2")
    print("Dien thoai 3: /phone?id=3")
    print("Monitor:      /monitor")
    threading.Timer(1.5, lambda: webbrowser.open("https://127.0.0.1:5000/monitor")).start()
    app.run(host="0.0.0.0", port=5000, ssl_context='adhoc', use_reloader=False)
