import cv2
import io
import math
import time
import webbrowser
import threading
import os
import base64
import hashlib
import numpy as np
import queue
import asyncio
import torch
import tempfile
import requests as http_requests
import edge_tts
from collections import deque
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from faster_whisper import WhisperModel

app = Flask(__name__)
CORS(app)

DANGER_CLASS_IDS = [0, 1, 2, 3, 5, 7, 13, 15, 16, 56, 60]

FRAME_SKIP   = 2  # Bỏ qua N-1 frame, chỉ inference frame thứ N
AUX_INTERVAL = 3  # Chạy door/stairs mỗi N lần inference chính, cache giữa các lần

class CombinedModel:
    def __init__(self):
        self._main = YOLO("yolo11n.pt")
        self.names = dict(self._main.names)
        self._aux_counter = 0
        self._aux_cache: list = []  # list of Tensor (n×6) — kết quả door/stairs được cache

        self._door_id   = None
        self._stairs_id = None

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

    def to_device(self, device):
        self._main.model.to(device)
        if self._door is not None:
            self._door.model.to(device)
        if self._stairs is not None:
            self._stairs.model.to(device)
        if device == "cpu":
            torch.cuda.empty_cache()
        print(f"[YOLO] Moved to {device}")

    def __call__(self, frame, **kwargs):
        results = self._main(frame, **kwargs)
        self._aux_counter += 1

        # Chạy door/stairs mỗi AUX_INTERVAL lần, cache tensor kết quả
        if self._aux_counter % AUX_INTERVAL == 0:
            self._aux_cache = []
            if self._door is not None:
                d = self._door(frame, verbose=False)[0].boxes.data.clone()
                if d.shape[0] > 0:
                    d[:, 5] = self._door_id
                    self._aux_cache.append(d)
            if self._stairs is not None:
                d = self._stairs(frame, verbose=False)[0].boxes.data.clone()
                if d.shape[0] > 0:
                    d[:, 5] = self._stairs_id
                    self._aux_cache.append(d)

        # Merge cached aux boxes vào kết quả chính
        for extra in self._aux_cache:
            results[0].boxes = _merge_boxes(results[0].boxes, extra)

        results[0].names = self.names
        return results

    def track(self, frame, **kwargs):
        """Gọi BoT-SORT built-in của Ultralytics, sau đó merge aux boxes."""
        results = self._main.track(frame, **kwargs)
        self._aux_counter += 1

        if self._aux_counter % AUX_INTERVAL == 0:
            self._aux_cache = []
            if self._door is not None:
                d = self._door(frame, verbose=False)[0].boxes.data.clone()
                if d.shape[0] > 0:
                    d[:, 5] = self._door_id
                    self._aux_cache.append(d)
            if self._stairs is not None:
                d = self._stairs(frame, verbose=False)[0].boxes.data.clone()
                if d.shape[0] > 0:
                    d[:, 5] = self._stairs_id
                    self._aux_cache.append(d)

        for extra in self._aux_cache:
            results[0].boxes = _merge_boxes(results[0].boxes, extra)

        results[0].names = self.names
        return results

def _merge_boxes(base_boxes, extra_data: torch.Tensor):
    """Thêm extra_data (n×6) vào base_boxes (có thể n×7 khi BoT-SORT đang track).
    Nếu base có 7 cột (x1,y1,x2,y2,track_id,conf,cls), chèn -1 làm track_id cho aux boxes.
    """
    if extra_data.shape[0] == 0:
        return base_boxes
    base_data = base_boxes.data
    if base_data.shape[0] == 0:
        base_boxes.data = extra_data
        return base_boxes
    if base_data.shape[1] == 7 and extra_data.shape[1] == 6:
        # Chèn cột track_id = -1 vào vị trí 4 cho aux boxes
        pad = torch.full((extra_data.shape[0], 1), -1.0,
                         device=extra_data.device, dtype=extra_data.dtype)
        extra_data = torch.cat([extra_data[:, :4], pad, extra_data[:, 4:]], dim=1)
    base_boxes.data = torch.cat([base_data, extra_data], dim=0)
    return base_boxes

model = CombinedModel()
INFER_DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[Model] Device: {'GPU (CUDA)' if INFER_DEVICE == 0 else 'CPU'}")

_depth_device = "cuda" if torch.cuda.is_available() else "cpu"
print("[Metric3Dv2] Đang tải model metric depth...")
_depth_model = torch.hub.load(
    r"C:\Users\Vinh\.cache\torch\hub\yvanyin_metric3d_main",
    "metric3d_vit_small", pretrain=True, source="local"
)
_depth_model = _depth_model.to(_depth_device).eval()
_depth_lock  = threading.Lock()
print(f"[Metric3Dv2] Sẵn sàng ({_depth_device.upper()})")

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

DISTANCES = ["rất gần", "khoảng 2 mét", "khoảng 3 mét", "khoảng 5 mét", ""]

# Navigation mode
NAV_COOLDOWN         = 2.5   # giây giữa 2 hướng dẫn liên tiếp
NAV_REPEAT_COOLDOWN  = 5.0   # giây trước khi lặp lại cùng một thông điệp
NAV_DANGER_THRESHOLD = 4.0   # tổng điểm — vùng coi là có vật cản
NAV_URGENT_THRESHOLD = 12.0  # tổng điểm — vật cản rất gần, cần dừng/cẩn thận ngay
DEPTH_INTERVAL       = 5     # Chạy DepthAnything mỗi N inference (GPU: 5 ổn)

NAV_MESSAGES = {
    "nav_straight":   "Đường thông, tiếp tục đi thẳng",
    "nav_caution":    "Có vật cản phía trước, đi chậm lại",
    "nav_turn_left":  "Rẽ trái",
    "nav_turn_right": "Rẽ phải",
    "nav_blocked":    "Dừng lại! Xung quanh có vật cản",
    "nav_start":      "Bật chế độ dẫn đường",
    "nav_stop_mode":  "Tắt chế độ dẫn đường",
}

MAX_DEVICES = 3
COOLDOWN = 6

APPROACH_RATIO    = 1.40   # cần tăng 40% area để tránh jitter YOLO gây false positive
APPROACH_COOLDOWN = 3
TRACK_WINDOW      = 5      # window rộng hơn → trend ổn định hơn

os.makedirs("audio", exist_ok=True)

# ---------------------------------------------------------------------------
# Whisper (nhận dạng giọng nói local)
# ---------------------------------------------------------------------------

print("[Whisper] Đang tải model small...")
_whisper_device       = "cuda" if torch.cuda.is_available() else "cpu"
_whisper_compute_type = "float16" if _whisper_device == "cuda" else "int8"
whisper_model = WhisperModel("small", device=_whisper_device, compute_type=_whisper_compute_type)
print(f"[Whisper] Sẵn sàng. Device: {_whisper_device.upper()} ({_whisper_compute_type})")

# ---------------------------------------------------------------------------
# OpenRouter Vision API
# ---------------------------------------------------------------------------

OPENROUTER_API_KEY = "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"
OPENROUTER_URL     = "https://openrouter.ai/api/v1/chat/completions"

# Thử tuần tự — dùng model đầu tiên, nếu hết quota (429) thì fallback; reset sau 24h
_FREE_VISION_MODELS = [
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "meta-llama/llama-4-scout:free",
]
_model_exhausted: dict = {}  # model -> timestamp khi bị 429

def _is_exhausted(model: str) -> bool:
    return (time.time() - _model_exhausted.get(model, 0)) < 86400
print(f"[OpenRouter] Race {len(_FREE_VISION_MODELS)} models: {_FREE_VISION_MODELS}")

def _resize_image_b64(image_b64: str, max_side: int = 512) -> str:
    raw = base64.b64decode(image_b64)
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


_SYSTEM_PROMPT = (
    "Bạn là mắt AI cho người khiếm thị. Trả lời bằng tiếng Việt, ngắn gọn, không lặp ý. "
    "Mô tả vật/người/chướng ngại vật, đọc chữ, hướng dẫn trái/phải/thẳng/dừng, cảnh báo nguy hiểm. "
    "Không mô tả màu sắc hay ngoại hình chi tiết — chỉ nêu thông tin cần thiết để di chuyển an toàn. "
    "Không từ chối, không hỏi lại, không giải thích — đi thẳng vào nội dung."
)

def _call_one_model(model: str, image_b64: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]},
        ],
        "max_tokens": 120,
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://smarteyes.local",
        "X-Title": "SmartEyes AI",
    }
    resp = http_requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=25)
    if not resp.ok:
        print(f"[AI] {model} HTTP {resp.status_code}: {resp.text[:300]}")
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def query_ai(image_b64: str, user_text: str) -> str:
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    image_b64 = _resize_image_b64(image_b64)

    errors = []
    for m in _FREE_VISION_MODELS:
        if _is_exhausted(m):
            print(f"[AI] {m} bỏ qua (hết quota 24h)")
            continue
        t0 = time.time()
        try:
            text = _call_one_model(m, image_b64, user_text)
            if text:
                print(f"[AI] {m} → {time.time()-t0:.1f}s ✓")
                return text
        except Exception as e:
            if "429" in str(e):
                _model_exhausted[m] = time.time()
                print(f"[AI] {m} hết quota → chuyển fallback")
            errors.append(f"{m}: {e}")
            print(f"[AI] {m} → {e}")

    raise Exception(f"Tất cả model đều lỗi: {errors}")

async def _tts_to_bytes(text: str, key: str) -> bytes:
    path = f"audio/{key}.mp3"
    if not os.path.exists(path):
        await edge_tts.Communicate(text, VOICE, rate=RATE).save(path)
    with open(path, "rb") as f:
        return f.read()

def tts_sync(text: str, key: str) -> str:
    """Generate TTS synchronously and return base64-encoded mp3."""
    future = asyncio.run_coroutine_threadsafe(_tts_to_bytes(text, key), _audio_loop)
    try:
        audio_bytes = future.result(timeout=20)
        return base64.b64encode(audio_bytes).decode()
    except Exception as e:
        print(f"[TTS Error] {e}")
        return ""

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
                "mode":                 "yolo",
                "latest_frame_b64":     None,
                "frame_count":          0,
                "depth_map":            None,
                "_depth_ctr":           0,
                "last_direction":       {},
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

MONITOR_TIMEOUT = 5  # giây — sau khoảng này coi monitor offline, bỏ qua plot

def _do_inference(device_id, state, frame):
    frame_height, frame_width = frame.shape[:2]
    # BoT-SORT built-in tracking (persist=True giữ trạng thái tracker giữa các frame)
    results = model.track(frame, classes=DANGER_CLASS_IDS, verbose=False,
                          device=INFER_DEVICE, persist=True, tracker="botsort.yaml")
    now = time.time()

    # Chạy depth estimation mỗi DEPTH_INTERVAL inference
    state["_depth_ctr"] += 1
    if state["_depth_ctr"] % DEPTH_INTERVAL == 0:
        try:
            state["depth_map"] = _estimate_depth(frame)
        except Exception as e:
            print(f"[Depth] Lỗi: {e}")

    # Thu thập tất cả detection nguy hiểm (kể cả aux boxes được merge)
    raw_dets = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        if label not in DANGER:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        area = (x2 - x1) * (y2 - y1)
        # box.id = None nếu tracker chưa assign; -1 nếu là aux box (door/stairs)
        tid = int(box.id[0]) if box.id is not None else -1
        raw_dets.append([x1, y1, x2, y2, label, area, tid])
        state["last_seen"][label] = now

    # Sắp xếp theo priority cao → thấp, rồi area lớn → nhỏ
    tracked = sorted(raw_dets, key=lambda d: (PRIORITY.get(d[4], 0), d[5]), reverse=True)

    # Cập nhật area_history chỉ cho box có track ID hợp lệ (>= 0)
    area_history = state["area_history"]
    alive_tks = {f"t{d[6]}" for d in tracked if d[6] >= 0}
    for tk in list(area_history.keys()):
        if tk not in alive_tks:
            del area_history[tk]
    for det in tracked:
        if det[6] < 0:
            continue  # bỏ qua aux boxes (door/stairs) — không có track ID ổn định
        tk = f"t{det[6]}"
        if tk not in area_history:
            area_history[tk] = deque(maxlen=TRACK_WINDOW)
        area_history[tk].append(det[5])

    last_spoken          = state["last_spoken"]
    last_approach_spoken = state["last_approach_spoken"]
    latest_alert         = state["latest_alert"]
    last_direction       = state["last_direction"]  # {str(tid): direction}

    # Xóa direction state cho track đã mất
    alive_tids = {str(d[6]) for d in tracked if d[6] >= 0}
    for k in list(last_direction.keys()):
        if k not in alive_tids:
            del last_direction[k]

    # Dọn cooldown hết hạn
    for lbl in list(last_spoken.keys()):
        if now - state["last_seen"].get(lbl, 0) > 2.0:
            del last_spoken[lbl]

    # --- Kiểm tra vật thể đang tiến lại (ưu tiên cao nhất, 1 cảnh báo) ---
    # Yêu cầu ít nhất 4/5 frame liên tiếp tăng trưởng >= 5% để tránh jitter YOLO
    approach_fired = False
    for det in tracked:
        x1, y1, x2, y2, label, area, tid = det
        tk   = f"t{tid}"
        hist = area_history.get(tk, deque())
        if len(hist) < TRACK_WINDOW:
            continue
        growth_steps = sum(1 for i in range(len(hist)-1) if hist[i+1] > hist[i] * 1.05)
        is_approaching = (
            growth_steps >= TRACK_WINDOW - 2          # ít nhất 3/4 hoặc 4/5 frame tăng
            and hist[-1] > hist[0] * APPROACH_RATIO   # tổng tăng >= 40%
        )
        if is_approaching and now - last_approach_spoken.get(label, 0) > APPROACH_COOLDOWN:
            tid_key   = str(tid)
            direction = get_direction(x1, x2, frame_width, last_direction.get(tid_key))
            last_direction[tid_key] = direction
            msg       = build_approach_message(label, direction)
            audio_key = f"approach_{label}_{direction.replace(' ', '_')}"
            threading.Thread(target=generate_audio, args=(audio_key, msg), daemon=True).start()
            latest_alert.update({"message": msg, "timestamp": now, "label": audio_key})
            last_approach_spoken[label] = now
            last_spoken[label]          = now
            print(f"[TIEP CAN device {device_id}] {msg}")
            approach_fired = True
            break

    # --- Dẫn đường liên tục (nav mode) hoặc cảnh báo đa vật thể ---
    if not approach_fired and state.get("nav_mode", False):
        _nav_guidance(device_id, state, tracked, frame_width, frame_height, now)

    # --- Cảnh báo vật thể (1 object, ưu tiên gần nhất) ---
    if not approach_fired and not state.get("nav_mode", False):
        depth_map = state.get("depth_map")

        for det in tracked:
            x1, y1, x2, y2, label, _, tid = det
            tid_key   = str(tid)
            direction = get_direction(x1, x2, frame_width, last_direction.get(tid_key))
            last_direction[tid_key] = direction

            if now - last_spoken.get(label, 0) > COOLDOWN:
                distance  = get_distance_with_depth(x1, y1, x2, y2, label, depth_map)
                msg       = build_message(label, direction, distance)
                audio_key = f"{label}_{direction.replace(' ', '_')}_{distance.replace(' ', '_')}"
                threading.Thread(target=generate_audio, args=(audio_key, msg), daemon=True).start()
                latest_alert.update({"message": msg, "timestamp": now, "label": audio_key})
                last_spoken[label] = now
                print(f"[Thiet bi {device_id}] {msg}")
                break

    # Luôn render annotated frame — phone và monitor đều có thể xem
    annotated = results[0].plot()
    h, w = annotated.shape[:2]
    # Vạch phân chia vùng trái/giữa/phải (khớp ngưỡng 33%/67%)
    cv2.line(annotated, (w // 3, 0), (w // 3, h), (0, 255, 0), 1)
    cv2.line(annotated, (2 * w // 3, 0), (2 * w // 3, h), (0, 255, 0), 1)
    for det in tracked:
        x1a, y1a, lbl = det[0], det[1], det[4]
        name = VIET_LABELS.get(lbl, lbl)
        cv2.putText(annotated, name, (int(x1a), max(16, int(y1a) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f"TB {device_id}", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
    with state["frame_lock"]:
        state["latest_frame"] = buf.tobytes()

def _estimate_depth(frame_bgr):
    """Trả về depth map float32 cùng kích thước frame. Giá trị = khoảng cách thực (mét)."""
    h_orig, w_orig = frame_bgr.shape[:2]
    rgb_origin = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Resize giữ tỉ lệ, fit vào 616×1064 (ViT input size)
    input_size = (616, 1064)
    scale = min(input_size[0] / h_orig, input_size[1] / w_orig)
    rgb = cv2.resize(rgb_origin, (int(w_orig * scale), int(h_orig * scale)),
                     interpolation=cv2.INTER_LINEAR)

    # Pad về đúng kích thước với giá trị mean
    h, w = rgb.shape[:2]
    pad_h, pad_w   = input_size[0] - h, input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                             pad_w_half, pad_w - pad_w_half,
                             cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    # Normalize và chuyển thành tensor
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std  = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb_t = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb_t = (rgb_t - mean) / std
    rgb_t = rgb_t[None].to(_depth_device)

    with _depth_lock:
        with torch.no_grad():
            pred_depth, _, _ = _depth_model.inference({"input": rgb_t})
        if _depth_device == "cuda":
            torch.cuda.empty_cache()

    # Xóa padding
    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                             pad_info[2]: pred_depth.shape[1] - pad_info[3]]

    # Upsample về kích thước gốc
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None], (h_orig, w_orig), mode="bilinear", align_corners=False
    ).squeeze()

    # Canonical camera space → metric (mét): nhân focal đã scale theo resize (theo example chính thức)
    pred_depth = pred_depth * (FOCAL_LENGTH_PX * scale / 1000.0)
    pred_depth = torch.clamp(pred_depth, 0, 50)

    return pred_depth.cpu().numpy().astype(np.float32)

def _vfh_nav(tracked, depth_map, frame_w, frame_h):
    """
    Vector Field Histogram+ (VFH+): xây polar histogram 180° phía trước,
    tìm valley (vùng trống) gần hướng thẳng nhất.
    Trả về (nav_key, urgency).
    """
    HIST_BINS  = 36          # 5° mỗi bin → tổng 180°
    SECTOR_DEG = 180.0 / HIST_BINS
    histogram  = np.zeros(HIST_BINS, dtype=np.float32)

    for det in tracked:
        x1, y1, x2, y2, label = det[0], det[1], det[2], det[3], det[4]
        cx = (x1 + x2) / 2

        if depth_map is not None:
            px     = min(frame_w - 1, max(0, int(cx)))
            py     = min(frame_h - 1, max(0, int((y1 + y2) / 2)))
            dist_m = max(0.1, float(depth_map[py, px]))   # mét thực
        else:
            box_h  = max(y2 - y1, 1)
            real_h = KNOWN_HEIGHTS_CM.get(label, 150)
            dist_m = max(0.1, (real_h * FOCAL_LENGTH_PX) / (box_h * 100.0))

        proximity = max(0.0, (5.0 - min(dist_m, 5.0)) / 5.0)  # 1=0m, 0=5m+
        prio      = PRIORITY.get(label, 1)
        magnitude = prio * (proximity ** 2) * 10.0              # phi tuyến

        # Góc nằm ngang của vật so với trung tâm frame (−90° trái, 0° thẳng, +90° phải)
        angle_deg  = math.degrees(math.atan2(cx - frame_w / 2, frame_h * 0.6))
        angle_deg  = max(-90.0, min(90.0, angle_deg))
        half_w_deg = math.degrees(math.atan2((x2 - x1) / 2, frame_h * 0.6))

        bin_ctr = int((angle_deg + 90.0) / SECTOR_DEG)
        spread  = max(1, int(half_w_deg / SECTOR_DEG) + 1)
        for b in range(max(0, bin_ctr - spread), min(HIST_BINS, bin_ctr + spread + 1)):
            histogram[b] += magnitude

    # Làm trơn histogram (3-bin moving average — bước chuẩn của VFH+)
    hist_s = np.convolve(histogram, np.ones(3) / 3, mode="same")

    urgency    = float(hist_s.max()) if hist_s.max() > 0 else 0.0
    CENTER_BIN = HIST_BINS // 2   # bin 18 = 0° = thẳng

    if urgency < 1.0:
        return "nav_straight", urgency

    # Tìm các valley (bin liên tiếp dưới ngưỡng)
    THRESHOLD   = 2.5
    valleys, in_valley, v_start = [], False, 0
    for i, h in enumerate(hist_s):
        if h < THRESHOLD and not in_valley:
            in_valley, v_start = True, i
        elif h >= THRESHOLD and in_valley:
            in_valley = False
            valleys.append((v_start, i - 1))
    if in_valley:
        valleys.append((v_start, HIST_BINS - 1))

    if not valleys:
        return "nav_blocked", urgency

    # Chọn valley gần hướng thẳng nhất
    best_v   = min(valleys, key=lambda v: abs((v[0] + v[1]) // 2 - CENTER_BIN))
    best_deg = (best_v[0] + best_v[1]) / 2 * SECTOR_DEG - 90.0

    ANGLE_TOL = 15.0   # ° — trong khoảng này vẫn coi là thẳng
    if best_deg < -ANGLE_TOL:
        return "nav_turn_left", urgency
    elif best_deg > ANGLE_TOL:
        return "nav_turn_right", urgency
    elif urgency > 6.0:
        return "nav_caution", urgency
    else:
        return "nav_straight", urgency


def _nav_guidance(device_id, state, tracked, frame_width, frame_height, now):
    """Dẫn đường bằng VFH+ (Vector Field Histogram+) + Metric3Dv2."""
    if not state.get("nav_mode", False):
        return
    if now - state.get("last_nav_spoken", 0) < NAV_COOLDOWN:
        return

    depth_map    = state.get("depth_map")
    key, urgency = _vfh_nav(tracked, depth_map, frame_width, frame_height)

    msg = NAV_MESSAGES[key]
    if msg == state.get("last_nav_message", "") and \
       now - state.get("last_nav_spoken", 0) < NAV_REPEAT_COOLDOWN:
        return

    generate_audio(key, msg)
    state["latest_alert"].update({"message": msg, "timestamp": now, "label": key})
    state["last_nav_spoken"]  = now
    state["last_nav_message"] = msg
    print(f"[NAV device {device_id}] {msg} (urgency={urgency:.1f})")

def get_direction(x1, x2, frame_width, prev=None):
    ratio = ((x1 + x2) / 2) / frame_width
    # Hysteresis: ổn định hướng, tránh dao động ở biên
    if prev == "bên trái"  and ratio < 0.42: return "bên trái"
    if prev == "bên phải" and ratio > 0.58: return "bên phải"
    if ratio < 0.33: return "bên trái"
    if ratio > 0.67: return "bên phải"
    return "phía trước"

def get_distance(y1, y2, label):
    """Ước lượng khoảng cách thực tế dựa vào chiều cao bounding box."""
    box_h = float(y2 - y1)
    if box_h < 10:
        return ""
    real_h  = KNOWN_HEIGHTS_CM.get(label, 150)
    dist_cm = (real_h * FOCAL_LENGTH_PX) / box_h
    if dist_cm < 150:   return "rất gần"
    elif dist_cm < 250: return "khoảng 2 mét"
    elif dist_cm < 400: return "khoảng 3 mét"
    elif dist_cm < 600: return "khoảng 5 mét"
    else:               return ""

def _sample_depth(depth_map, cx, cy, patch=20):
    """Lấy median vùng patch để tránh nhiễu pixel đơn."""
    h, w = depth_map.shape
    y1, y2 = max(0, cy - patch // 2), min(h, cy + patch // 2)
    x1, x2 = max(0, cx - patch // 2), min(w, cx + patch // 2)
    roi = depth_map[y1:y2, x1:x2].flatten()
    valid = roi[(roi > 0.05) & (roi < 40.0)]
    return float(np.median(valid)) if len(valid) >= 4 else float(depth_map[cy, cx])

def get_distance_with_depth(x1, y1, x2, y2, label, depth_map):
    """Ước lượng khoảng cách: dùng metric depth (mét) từ Metric3Dv2, fallback về focal length."""
    if depth_map is not None:
        cx = min(depth_map.shape[1]-1, max(0, int((x1+x2)/2)))
        cy = min(depth_map.shape[0]-1, max(0, int((y1+y2)/2)))
        d = _sample_depth(depth_map, cx, cy)
        if   d < 1.5: return "rất gần"
        elif d < 2.5: return "khoảng 2 mét"
        elif d < 4.0: return "khoảng 3 mét"
        elif d < 6.0: return "khoảng 5 mét"
        else:         return ""
    return get_distance(y1, y2, label)

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
        sem = asyncio.Semaphore(5)
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
        ] + [
            _gen_audio_async(key, text, sem)
            for key, text in NAV_MESSAGES.items()
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

    # Luôn lưu frame mới nhất (dùng cho AI query và monitor)
    state["latest_frame_b64"] = data['image']

    if state.get("mode") == "ai":
        return jsonify(state["latest_alert"])

    # Frame skip — chỉ inference mỗi FRAME_SKIP frame, trả kết quả cache cho frame bị bỏ
    state["frame_count"] += 1
    if state["frame_count"] % FRAME_SKIP != 0:
        return jsonify(state["latest_alert"])

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
    for _ in range(80):
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

@app.route("/nav_mode/<device_id>", methods=["GET"])
def get_nav_mode(device_id):
    state = get_device_state(device_id)
    return jsonify({"nav_mode": state.get("nav_mode", False)})

@app.route("/nav_mode/<device_id>", methods=["POST"])
def toggle_nav_mode(device_id):
    state = get_device_state(device_id)
    state["nav_mode"] = not state.get("nav_mode", False)
    is_on = state["nav_mode"]
    key = "nav_start" if is_on else "nav_stop_mode"
    msg = NAV_MESSAGES[key]
    generate_audio(key, msg)
    state["latest_alert"].update({"message": msg, "timestamp": time.time(), "label": key})
    return jsonify({"nav_mode": is_on})

@app.route("/alert_state")
def alert_state():
    device_id = str(request.args.get("device_id", "1"))
    state = get_device_state(device_id)
    return jsonify(state["latest_alert"])

@app.route("/mode/<device_id>", methods=["POST"])
def set_mode(device_id):
    state = get_device_state(device_id)
    new_mode = (request.json or {}).get("mode", "yolo")
    if new_mode not in ("yolo", "ai"):
        return jsonify({"error": "Invalid mode"}), 400
    state["mode"] = new_mode
    return jsonify({"mode": new_mode})

@app.route("/ai_query", methods=["POST"])
def ai_query():
    data = request.json or {}
    device_id = str(data.get("device_id", "1"))
    user_text  = data.get("query", "").strip()
    if not user_text:
        return jsonify({"error": "No query"}), 400

    state = get_device_state(device_id)
    image_b64 = data.get("image") or state.get("latest_frame_b64")
    if not image_b64:
        return jsonify({"error": "Chưa có hình ảnh từ camera"}), 400

    try:
        response_text = query_ai(image_b64, user_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    print(f"[AI] response_text={response_text!r}")
    if not response_text:
        return jsonify({"error": "AI trả về rỗng"}), 500

    tts_key   = "ai_" + hashlib.md5(response_text.encode()).hexdigest()[:8]
    audio_b64 = tts_sync(response_text, tts_key)
    print(f"[AI] tts done, audio_b64 len={len(audio_b64)}")

    now = time.time()
    state["latest_alert"].update({"message": response_text, "timestamp": now, "label": tts_key})
    return jsonify({"response": response_text, "audio_b64": audio_b64, "timestamp": now})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio"}), 400
    ext = os.path.splitext(audio_file.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        audio_file.save(f.name)
        tmp_path = f.name
    try:
        segments, _ = whisper_model.transcribe(
            tmp_path, language="vi",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
        )
        text = " ".join(s.text for s in segments).strip()
        return jsonify({"text": text})
    finally:
        os.unlink(tmp_path)

@app.route("/frame/<device_id>")
def get_frame(device_id):
    state = get_device_state(device_id)
    state["last_monitor_poll"] = time.time()
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

def _ensure_cert(cert_path="cert.pem", key_path="key.pem"):
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import datetime, ipaddress
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"SmartEyes")])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject).issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
            .add_extension(x509.SubjectAlternativeName([
                x509.DNSName(u"localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]), critical=False)
            .sign(key, hashes.SHA256())
        )
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption()))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        print(f"[SSL] Da tao cert moi: {cert_path}")
        return cert_path, key_path
    except ImportError:
        print("[SSL] Thieu cryptography, dung adhoc cert")
        return None, None

if __name__ == "__main__":
    import sys, shutil
    use_ssl = "--no-ssl" not in sys.argv
    shutil.rmtree("audio", ignore_errors=True)
    os.makedirs("audio", exist_ok=True)
    print("[Audio] Dang tao truoc tat ca audio clips (edge-tts)...")
    pregenerate_audio()
    scheme = "https" if use_ssl else "http"
    print(f"Server chay tai: {scheme}://192.168.62.197:5000")
    print("Dien thoai 1: /phone?id=1")
    print("Dien thoai 2: /phone?id=2")
    print("Dien thoai 3: /phone?id=3")
    print("Monitor:      /monitor")
    threading.Timer(1.5, lambda: webbrowser.open(f"{scheme}://127.0.0.1:5000/monitor")).start()
    if use_ssl:
        cert_f, key_f = _ensure_cert()
        ssl_ctx = (cert_f, key_f) if cert_f else 'adhoc'
    else:
        ssl_ctx = None
    app.run(host="0.0.0.0", port=5000, ssl_context=ssl_ctx, use_reloader=False)
