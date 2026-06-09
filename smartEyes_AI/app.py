# -*- coding: utf-8 -*-
"""
SmartEye Pro - Doi mat thu hai cho nguoi khiem thi (ban hop nhat, viet lai tu dau)

Kien truc:
  - Backend (Flask) la "bo nao": YOLO11n + model cua/cau thang + uoc luong do sau
    Metric3Dv2 + bam vet BoT-SORT + dan duong + hoi dap AI (Vision LLM) + OCR.
  - Backend chi tra ve VAN BAN (cau canh bao / cau tra loi). Trinh duyet lo toan bo
    am thanh (doc bang Web Speech, rung) + camera + SOS + phat hien te nga + thu giong noi.

Chay:
    python app.py            -> https://<IP-LAN>:5000  (mac dinh bat HTTPS de dien thoai dung camera/mic)
    python app.py --no-ssl   -> http://localhost:5000  (chi test tren laptop)

Cac thanh phan nang (do sau, OCR, AI) deu suy bien muot: thieu thi tu bo qua,
he thong van chay duoc bang phan con lai.
"""

import os
import io
import re
import time
import base64
import socket
import argparse
import threading
import queue
from collections import deque

import cv2
import numpy as np
import torch
import requests as http_requests
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO


# ===========================================================================
# Nap bien moi truong tu .env (khong ghi de bien da co)
# ===========================================================================
def _load_dotenv(path=".env"):
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

app = Flask(__name__)
CORS(app)

# ===========================================================================
# Cau hinh
# ===========================================================================
INFER_DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[Model] Thiet bi suy luan: {'GPU (CUDA)' if INFER_DEVICE == 0 else 'CPU'}")

CONF_THRESHOLD = 0.45            # nguong tin cay toi thieu cho YOLO chinh
FRAME_SKIP     = 2               # chi suy luan moi N frame
DEPTH_INTERVAL = 5               # chay uoc luong do sau moi N lan suy luan (che do thuong)
DEPTH_INTERVAL_NAV = 2           # khi dan duong/homing: cap nhat do sau day hon de phan ung nhanh
COOLDOWN       = 6               # giay giua 2 lan doc cung 1 nhan
APPROACH_COOLDOWN = 3            # giay giua 2 canh bao "dang tien lai"
TRACK_WINDOW   = 5               # so frame de xet xu huong lai gan

# COCO class ids YOLO chinh quan tam (nguoi, xe co, vat dung de va vap)
DANGER_CLASS_IDS = [0, 1, 2, 3, 5, 7, 13, 15, 16, 56, 60]

# --- Model phu door/stairs/fan ---
AUX_INTERVAL = 3                 # chay door/stairs moi N lan suy luan, cache giua cac lan
AUX_CONF     = 0.15              # nguong conf cho cua
STAIRS_CONF  = 0.12             # nguong conf RIENG cho cau thang (thuong conf thap)
AUX_VOTE_K   = 2                 # phai xuat hien >= K trong M lan aux gan nhat (chong nhap nhay)
AUX_VOTE_M   = 3
# Nhan tu model phu (bo qua nguong CONF_THRESHOLD cua model chinh)
AUX_LABELS   = {"door", "stairs", "fan"}

# --- Phat hien quat bang YOLO-World (open-vocabulary, COCO khong co lop "fan") ---
FAN_ENABLE      = True
FAN_WEIGHTS     = "yolov8s-world.pt"   # tu tai lan dau (can goi clip)
FAN_PROMPTS     = ["fan", "electric fan", "ceiling fan", "table fan"]
FAN_CONF        = 0.10                 # world model conf thuong thap
FAN_AUX_EVERY   = 2                    # chay moi N chu ky aux (nang tren CPU -> chay thua)
# Prior hinh hoc cho cau thang (luoi an toan long, loai box be/cao bat thuong)
STAIRS_MIN_AR        = 0.30
STAIRS_MIN_BOTTOM    = 0.15
STAIRS_MIN_AREA_FRAC = 0.005

# --- Phat hien tien lai bang do sau that ---
APPROACH_DEPTH_RATIO = 0.78      # khoang cach hien tai <= 78% luc dau window (lai gan >= 22%)
APPROACH_MIN_DROP_M  = 0.4       # va lai gan it nhat 0.4 m (loc nhieu)
APPROACH_MAX_DIST_M  = 6.0       # chi canh bao vat trong 6 m
APPROACH_RATIO       = 1.40      # fallback khong co do sau: can tang 40% dien tich

# --- Dan duong free-space (tu depth map, met) ---
NAV_COOLDOWN        = 2.0
NAV_REPEAT_COOLDOWN = 5.0
NAV_STOP_M  = 1.2                # vat gan hon -> dung lai
NAV_SLOW_M  = 2.0                # vat gan hon -> phai re/di cham (nhay hon truoc)
NAV_CLEAR_M = 2.8                # thoang hon -> di thang
# Phat hien vat can TUONG DOI (chong sai lech tuyet doi cua do sau khong hieu chuan):
# cot giua gan hon han hai ben va o trong tam voi nhau -> coi nhu co vat truoc mat.
NAV_REL_M     = 3.5              # chi xet tuong doi khi vat trong khoang nay
NAV_REL_RATIO = 0.65            # cot giua < 65% do thoang nho nhat cua hai ben

# --- Dan duong toi muc tieu cu the (homing) ---
GOAL_ARRIVE_M        = 1.0
GOAL_ARRIVE_BOX_FRAC = 0.45
GOAL_LOST_TIMEOUT    = 10.0
GOAL_AVOID_MARGIN_M  = 0.5
GOAL_COOLDOWN        = 1.5
GOAL_NEAR_LOST_M     = 1.5

FOCAL_LENGTH_PX = 600            # tieu cu uoc tinh cho camera dien thoai ~720p

# Dich nhan sang tieng Viet (de hien thi va doc cho de hieu)
VIET_LABELS = {
    "person": "người", "car": "xe hơi", "motorcycle": "xe máy",
    "bicycle": "xe đạp", "bus": "xe buýt", "truck": "xe tải",
    "dog": "chó", "cat": "mèo", "bench": "ghế đá",
    "chair": "ghế", "dining table": "cái bàn", "door": "cửa",
    "stairs": "cầu thang", "fan": "quạt",
}
DANGER = {"person", "car", "motorcycle", "bus", "truck", "bench",
          "bicycle", "cat", "dog", "chair", "dining table", "door", "stairs", "fan"}
PRIORITY = {
    "car": 5, "motorcycle": 5, "bus": 5, "truck": 5,
    "stairs": 4, "bicycle": 4,
    "person": 3,
    "dog": 2, "cat": 2,
    "fan": 2,
    "door": 1, "bench": 1, "chair": 1, "dining table": 1,
}
KNOWN_HEIGHTS_CM = {
    "person": 170, "car": 150, "motorcycle": 110, "bus": 280, "truck": 250,
    "bicycle": 100, "dog": 50, "cat": 30, "bench": 90, "chair": 90,
    "dining table": 75, "door": 200, "stairs": 100, "fan": 40,
}

NAV_MESSAGES = {
    "nav_straight":   "Đường thông, tiếp tục đi thẳng",
    "nav_caution":    "Có vật cản phía trước, đi chậm lại",
    "nav_turn_left":  "Rẽ trái",
    "nav_turn_right": "Rẽ phải",
    "nav_blocked":    "Dừng lại! Xung quanh có vật cản",
}
_OBSTACLE_KEYS = {"nav_blocked", "nav_caution", "nav_turn_left", "nav_turn_right"}


# ===========================================================================
# Model chinh + phu (YOLO11n + door + stairs), gop ket qua
# ===========================================================================
class CombinedModel:
    """YOLO11n (chinh) + doors.pt + stairs.pt (phu). Model phu chay thua hon,
    co cache + bo phieu thoi gian + loc hinh hoc de chong bao nham/nhap nhay."""

    def __init__(self):
        self._main = YOLO("yolo11n.pt")
        self.names = dict(self._main.names)
        self._aux_counter = 0
        self._aux_cache = []          # list[Tensor n×6] ket qua door/stairs da cache
        self._aux_votes = {}          # name -> deque[bool] lich su co/khong

        self._door, self._door_id = self._load_aux("doors.pt", "door")
        self._stairs, self._stairs_id = self._load_aux("stairs.pt", "stairs")
        self._fan, self._fan_id = self._load_fan()
        self._fan_cycle = 0           # dem chu ky aux de chay fan thua hon (nang tren CPU)
        self._fan_cache = []          # giu rieng ket qua fan giua cac chu ky chua chay lai

    def _load_aux(self, path, name):
        if not os.path.exists(path):
            print(f"[YOLO] Khong thay {path}, bo qua phat hien {name}")
            return None, None
        mdl = YOLO(path)
        new_id = max(self.names.keys()) + 1
        self.names[new_id] = name
        print(f"[YOLO] Da nap model phu {name} ({path})")
        return mdl, new_id

    def _load_fan(self):
        """Nap YOLO-World de phat hien 'fan' (COCO khong co lop nay)."""
        if not FAN_ENABLE:
            return None, None
        try:
            from ultralytics import YOLOWorld
            mdl = YOLOWorld(FAN_WEIGHTS)
            mdl.set_classes(FAN_PROMPTS)     # moi prompt -> 1 lop, deu hieu la "fan"
            new_id = max(self.names.keys()) + 1
            self.names[new_id] = "fan"
            print(f"[YOLO] Da nap model phat hien quat ({FAN_WEIGHTS})")
            return mdl, new_id
        except Exception as e:
            print(f"[YOLO] Khong nap duoc model quat ({e}) -> bo qua phat hien fan")
            return None, None

    def _detect_fan(self, frame):
        """Chay YOLO-World cho fan (thua hon aux). Tra ve cache n×6 (cls=fan_id)."""
        if self._fan is None:
            return []
        self._fan_cycle += 1
        if self._fan_cycle % FAN_AUX_EVERY != 0:
            return self._fan_cache       # giu ket qua cu giua cac lan chay
        d = self._fan(frame, verbose=False, device=INFER_DEVICE, conf=FAN_CONF)[0].boxes.data.clone()
        votes = self._aux_votes.setdefault("fan", deque(maxlen=AUX_VOTE_M))
        votes.append(d.shape[0] > 0)
        if d.shape[0] > 0 and sum(votes) >= AUX_VOTE_K:
            d[:, 5] = self._fan_id       # mọi prompt fan -> cùng 1 nhãn "fan"
            self._fan_cache = [d]
        else:
            self._fan_cache = []
        return self._fan_cache

    def _refresh_aux(self, frame):
        self._aux_counter += 1
        if self._aux_counter % AUX_INTERVAL != 0:
            return
        H, W = frame.shape[:2]
        frame_area = float(H * W)
        new_cache = []
        for mdl, cls_id, name in ((self._door, self._door_id, "door"),
                                  (self._stairs, self._stairs_id, "stairs")):
            if mdl is None:
                continue
            conf = STAIRS_CONF if name == "stairs" else AUX_CONF
            d = mdl(frame, verbose=False, device=INFER_DEVICE, conf=conf)[0].boxes.data.clone()
            if name == "stairs" and d.shape[0] > 0:
                d = _filter_stairs(d, H, frame_area)
                if d.shape[0] > 1:   # gop nhieu box mo cua cung 1 cau thang thanh 1
                    d = d.new_tensor([[float(d[:, 0].min()), float(d[:, 1].min()),
                                       float(d[:, 2].max()), float(d[:, 3].max()),
                                       float(d[:, 4].max()), float(d[0, 5])]])
            present = d.shape[0] > 0
            votes = self._aux_votes.setdefault(name, deque(maxlen=AUX_VOTE_M))
            votes.append(present)
            if present and sum(votes) >= AUX_VOTE_K:
                d[:, 5] = cls_id
                new_cache.append(d)
        new_cache.extend(self._detect_fan(frame))
        self._aux_cache = new_cache

    def _merge_cached(self, results):
        for extra in self._aux_cache:
            results[0].boxes = _merge_boxes(results[0].boxes, extra)
        results[0].names = self.names
        return results

    def track(self, frame, **kwargs):
        results = self._main.track(frame, **kwargs)
        self._refresh_aux(frame)
        return self._merge_cached(results)


def _merge_boxes(base_boxes, extra_data):
    """Them extra_data (n×6) vao base_boxes (co the n×7 khi BoT-SORT dang track)."""
    if extra_data.shape[0] == 0:
        return base_boxes
    base_data = base_boxes.data
    if base_data.shape[0] == 0:
        base_boxes.data = extra_data
        return base_boxes
    if base_data.shape[1] == 7 and extra_data.shape[1] == 6:
        pad = torch.full((extra_data.shape[0], 1), -1.0,
                         device=extra_data.device, dtype=extra_data.dtype)
        extra_data = torch.cat([extra_data[:, :4], pad, extra_data[:, 4:]], dim=1)
    base_boxes.data = torch.cat([base_data, extra_data], dim=0)
    return base_boxes


def _filter_stairs(d, H, frame_area):
    """Giu box cau thang hop ly ve hinh hoc (rong, nam thap, du lon)."""
    keep = []
    for row in d:
        x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        w, h = x2 - x1, y2 - y1
        if w <= 1 or h <= 1:
            continue
        if (w / h >= STAIRS_MIN_AR and y2 / H >= STAIRS_MIN_BOTTOM
                and (w * h) / frame_area >= STAIRS_MIN_AREA_FRAC):
            keep.append(row)
    return torch.stack(keep) if keep else d[:0]


# ===========================================================================
# Uoc luong do sau Metric3Dv2 (tuy chon - thieu thi dung focal-length)
# ===========================================================================
_depth_device = "cuda" if torch.cuda.is_available() else "cpu"
_depth_model = None
_depth_lock = threading.Lock()
_DEPTH_MEAN = torch.tensor([123.675, 116.28, 103.53]).float().view(3, 1, 1).to(_depth_device)
_DEPTH_STD  = torch.tensor([58.395, 57.12, 57.375]).float().view(3, 1, 1).to(_depth_device)


def _load_depth_model():
    global _depth_model
    cache = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub",
                         "yvanyin_metric3d_main")
    if not os.path.isdir(cache):
        print("[Depth] Khong thay cache Metric3D -> dung uoc luong focal-length")
        return
    try:
        print("[Depth] Dang tai Metric3Dv2 (metric depth)...")
        m = torch.hub.load(cache, "metric3d_vit_small", pretrain=True, source="local")
        _depth_model = m.to(_depth_device).eval()
        print(f"[Depth] San sang ({_depth_device.upper()})")
    except Exception as e:
        print(f"[Depth] Tai that bai ({e}) -> dung uoc luong focal-length")
        _depth_model = None


def _estimate_depth(frame_bgr):
    """Tra ve depth map float32 cung kich thuoc frame, gia tri = met."""
    h_orig, w_orig = frame_bgr.shape[:2]
    rgb_origin = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    input_size = (616, 1064)
    scale = min(input_size[0] / h_orig, input_size[1] / w_orig)
    rgb = cv2.resize(rgb_origin, (int(w_orig * scale), int(h_orig * scale)),
                     interpolation=cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]
    pad_h, pad_w = input_size[0] - h, input_size[1] - w
    pad_h_half, pad_w_half = pad_h // 2, pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half,
                             pad_w_half, pad_w - pad_w_half,
                             cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    rgb_t = torch.from_numpy(rgb.transpose((2, 0, 1))).to(_depth_device).float()
    rgb_t = ((rgb_t - _DEPTH_MEAN) / _DEPTH_STD)[None]

    with _depth_lock:
        with torch.no_grad():
            pred_depth, _, _ = _depth_model.inference({"input": rgb_t})

    pred_depth = pred_depth.squeeze()
    pred_depth = pred_depth[pad_info[0]: pred_depth.shape[0] - pad_info[1],
                            pad_info[2]: pred_depth.shape[1] - pad_info[3]]
    pred_depth = torch.nn.functional.interpolate(
        pred_depth[None, None], (h_orig, w_orig), mode="bilinear", align_corners=False
    ).squeeze()
    pred_depth = pred_depth * (FOCAL_LENGTH_PX * scale / 1000.0)
    pred_depth = torch.clamp(pred_depth, 0, 50)
    return pred_depth.cpu().numpy().astype(np.float32)


def _sample_depth(depth_map, cx, cy, patch=20):
    h, w = depth_map.shape
    y1, y2 = max(0, cy - patch // 2), min(h, cy + patch // 2)
    x1, x2 = max(0, cx - patch // 2), min(w, cx + patch // 2)
    roi = depth_map[y1:y2, x1:x2].flatten()
    valid = roi[(roi > 0.05) & (roi < 40.0)]
    return float(np.median(valid)) if len(valid) >= 4 else float(depth_map[cy, cx])


def _depth_nav(depth_map, prev_key=None):
    """Dan duong free-space tu depth map: chia FOV thanh N cot, lai ve phia thoang nhat.

    Uu tien tuyet doi vat NGAY TRUOC MAT: khi cot giua qua gan thi bao re/dung
    ngay, khong cho qua nhu truoc.
    """
    H, W = depth_map.shape
    # lay dai giua-duoi khung (noi vat can than the o tam tay/chan xuat hien)
    band = depth_map[int(0.30 * H):int(0.92 * H), :]
    N = 9
    col_w = max(1, W // N)
    col_min = np.full(N, 30.0, dtype=np.float32)
    for i in range(N):
        seg = band[:, i * col_w:(i + 1) * col_w]
        valid = seg[(seg > 0.1) & (seg < 30.0)]
        if valid.size > 8:
            col_min[i] = float(np.percentile(valid, 5))   # 5% gan nhat trong cot (nhay vat can)

    center_clear = float(col_min[N // 2 - 1:N // 2 + 2].min())  # 3 cot giua
    left_clear   = float(col_min[:N // 2].max())
    right_clear  = float(col_min[N // 2 + 1:].max())
    nearest      = float(col_min.min())

    # Vat can truoc mat: gan TUYET DOI, hoac gan TUONG DOI hon han hai ben
    # (cuu canh khi do sau bi lech ti le vi camera khong hieu chuan).
    obstacle_ahead = center_clear < NAV_SLOW_M
    if (not obstacle_ahead and center_clear < NAV_REL_M
            and center_clear < min(left_clear, right_clear) * NAV_REL_RATIO):
        obstacle_ahead = True

    if obstacle_ahead:
        # CO vat can ngay phia truoc -> tim ben thoang de re, neu khong thi dung
        if left_clear >= NAV_SLOW_M and left_clear >= right_clear:
            key = "nav_turn_left"
        elif right_clear >= NAV_SLOW_M:
            key = "nav_turn_right"
        elif center_clear < NAV_STOP_M:
            key = "nav_blocked"
        else:
            key = "nav_caution"
    elif center_clear >= NAV_CLEAR_M:
        key = "nav_straight"
    elif left_clear > right_clear + 0.5 and left_clear >= NAV_SLOW_M:
        key = "nav_turn_left"
    elif right_clear > left_clear + 0.5 and right_clear >= NAV_SLOW_M:
        key = "nav_turn_right"
    else:
        key = "nav_caution"

    # giu huong re de khoi dao trai-phai lien tuc
    if prev_key in ("nav_turn_left", "nav_turn_right") and key == "nav_straight" \
            and center_clear < NAV_CLEAR_M + 0.5:
        key = prev_key
    return key, nearest


def _semantic_nav_fallback(tracked, frame_w):
    """Khong co depth: lai ne dua tren box YOLO + focal-length."""
    if not tracked:
        return "nav_straight", 30.0
    nearest_m = 30.0
    third = frame_w / 3
    left_block = right_block = center_block = False
    for det in tracked:
        x1, y1, x2, y2, label = det[0], det[1], det[2], det[3], det[4]
        box_h = max(y2 - y1, 1)
        real_h = KNOWN_HEIGHTS_CM.get(label, 150)
        dist_m = max(0.1, (real_h * FOCAL_LENGTH_PX) / (box_h * 100.0))
        cx = (x1 + x2) / 2
        nearest_m = min(nearest_m, dist_m)
        if dist_m < NAV_SLOW_M:
            if cx < third:        left_block = True
            elif cx > 2 * third:  right_block = True
            else:                 center_block = True
    if not center_block and nearest_m >= NAV_CLEAR_M:
        return "nav_straight", nearest_m
    if center_block and left_block and right_block and nearest_m < NAV_STOP_M:
        return "nav_blocked", nearest_m
    if center_block:
        if not left_block:  return "nav_turn_left", nearest_m
        if not right_block: return "nav_turn_right", nearest_m
    return "nav_caution", nearest_m


# ===========================================================================
# Tien ich huong / khoang cach / cau noi
# ===========================================================================
def get_direction(x1, x2, frame_width, prev=None):
    ratio = ((x1 + x2) / 2) / frame_width
    if prev == "bên trái" and ratio < 0.42:  return "bên trái"
    if prev == "bên phải" and ratio > 0.58:  return "bên phải"
    if ratio < 0.33: return "bên trái"
    if ratio > 0.67: return "bên phải"
    return "phía trước"


def get_distance_focal(y1, y2, label):
    box_h = float(y2 - y1)
    if box_h < 10:
        return ""
    dist_cm = (KNOWN_HEIGHTS_CM.get(label, 150) * FOCAL_LENGTH_PX) / box_h
    if dist_cm < 150:   return "rất gần"
    elif dist_cm < 250: return "khoảng 2 mét"
    elif dist_cm < 400: return "khoảng 3 mét"
    elif dist_cm < 600: return "khoảng 5 mét"
    return ""


def get_distance_with_depth(x1, y1, x2, y2, label, depth_map):
    if depth_map is not None:
        cx = min(depth_map.shape[1] - 1, max(0, int((x1 + x2) / 2)))
        cy = min(depth_map.shape[0] - 1, max(0, int((y1 + y2) / 2)))
        d = _sample_depth(depth_map, cx, cy)
        if d < 1.5:   return "rất gần"
        elif d < 2.5: return "khoảng 2 mét"
        elif d < 4.0: return "khoảng 3 mét"
        elif d < 6.0: return "khoảng 5 mét"
        return ""
    return get_distance_focal(y1, y2, label)


def build_message(label, direction, distance):
    name = VIET_LABELS.get(label, label)
    return f"Có {name} {distance} {direction}, chú ý!" if distance else f"Có {name} {direction}"


def build_approach_message(label, direction):
    name = VIET_LABELS.get(label, label)
    return f"Cẩn thận! {name} đang tiến lại {direction}!"


# ===========================================================================
# Dan duong: free-space va homing toi muc tieu
# ===========================================================================
def _set_alert(state, message, danger=False, obj_label=None, src=None, nav_key=None):
    """Cap nhat canh bao moi nhat (trinh duyet se doc bang Web Speech)."""
    state["latest_alert"] = {
        "message": message,
        "timestamp": time.time(),
        "danger": danger,
        "obj_label": obj_label,
    }
    if src is not None:
        state["last_nav_spoken"] = time.time()
        state["last_nav_message"] = message
        state["last_nav_src"] = src
        if nav_key is not None:
            state["last_nav_key"] = nav_key


def _nav_guidance(state, tracked, frame_width, now):
    depth_map = state.get("depth_map")
    if depth_map is not None:
        key, dist = _depth_nav(depth_map, prev_key=state.get("last_nav_key"))
    else:
        key, dist = _semantic_nav_fallback(tracked, frame_width)
    msg = NAV_MESSAGES[key]
    # "dung lai" la canh bao khan -> phan ung ngay, khong bi cooldown/chong lap chan
    urgent = key == "nav_blocked"
    if not urgent and now - state.get("last_nav_spoken", 0) < NAV_COOLDOWN:
        return
    if (not urgent and msg == state.get("last_nav_message", "") and state.get("last_nav_src") == "nav"
            and now - state.get("last_nav_spoken", 0) < NAV_REPEAT_COOLDOWN):
        return
    danger = key in ("nav_blocked", "nav_caution")
    _set_alert(state, msg, danger=danger, src="nav", nav_key=key)
    print(f"[NAV] {msg} (gan nhat={dist:.1f}m)")


def _goal_dist_m(det, label, depth_map):
    if depth_map is not None:
        cx = min(depth_map.shape[1] - 1, max(0, int((det[0] + det[2]) / 2)))
        cy = min(depth_map.shape[0] - 1, max(0, int((det[1] + det[3]) / 2)))
        return _sample_depth(depth_map, cx, cy)
    box_h = max(det[3] - det[1], 1)
    return (KNOWN_HEIGHTS_CM.get(label, 150) * FOCAL_LENGTH_PX) / (box_h * 100.0)


def _emit_goal(state, msg, now, nav_key=None):
    if (msg == state.get("last_nav_message", "") and state.get("last_nav_src") == "goal"
            and now - state.get("last_nav_spoken", 0) < NAV_REPEAT_COOLDOWN):
        return
    _set_alert(state, msg, src="goal", nav_key=nav_key)


def _clear_goal(state, msg):
    state["goal_target"] = None
    state["goal_locked_tid"] = None
    state["goal_last_dir"] = None
    _set_alert(state, msg, src="goal")


def _goal_guidance(state, tracked, frame_width, frame_height, now, depth_map):
    label = state.get("goal_target")
    if not label:
        return
    if now - state.get("last_nav_spoken", 0) < GOAL_COOLDOWN:
        return
    name = VIET_LABELS.get(label, label)
    cands = [d for d in tracked if d[4] == label]

    if not cands:
        last_dist = state.get("goal_last_dist")
        if last_dist is not None and last_dist < GOAL_NEAR_LOST_M:
            _clear_goal(state, f"Đã tới {name}")
        elif now - state.get("goal_last_seen", state.get("goal_started", now)) > GOAL_LOST_TIMEOUT:
            _clear_goal(state, f"Không tìm thấy {name} quanh đây, đã dừng dẫn đường")
        elif state.get("goal_last_dir") == "bên phải":
            _emit_goal(state, f"Chưa thấy {name}, quay chậm sang phải để tìm", now)
        else:
            _emit_goal(state, f"Chưa thấy {name}, quay chậm sang trái để tìm", now)
        return

    locked = state.get("goal_locked_tid")
    target = None
    if locked is not None and locked >= 0:
        target = next((d for d in cands if d[6] == locked), None)
    if target is None:
        target = min(cands, key=lambda d: _goal_dist_m(d, label, depth_map))
        if target[6] >= 0:
            state["goal_locked_tid"] = target[6]
    state["goal_last_seen"] = now

    x1, y1, x2, y2 = target[0], target[1], target[2], target[3]
    dist_m = _goal_dist_m(target, label, depth_map)
    state["goal_last_dist"] = dist_m
    box_frac = (y2 - y1) / frame_height
    direction = get_direction(x1, x2, frame_width, state.get("goal_last_dir"))
    state["goal_last_dir"] = direction

    if depth_map is not None:
        key, obs_d = _depth_nav(depth_map, prev_key=state.get("last_nav_key"))
    else:
        others = [d for d in tracked if d is not target]
        key, obs_d = _semantic_nav_fallback(others, frame_width)
    if key in _OBSTACLE_KEYS and obs_d < dist_m - GOAL_AVOID_MARGIN_M:
        _emit_goal(state, NAV_MESSAGES[key], now, nav_key=key)
        return

    arrived = (dist_m < GOAL_ARRIVE_M) if depth_map is not None else (box_frac > GOAL_ARRIVE_BOX_FRAC)
    if arrived:
        _clear_goal(state, f"Đã tới {name}")
        return

    if direction == "phía trước":  msg = f"{name} ngay trước mặt, đi tới"
    elif direction == "bên trái":  msg = f"{name} bên trái, rẽ trái một chút"
    else:                          msg = f"{name} bên phải, rẽ phải một chút"
    _emit_goal(state, msg, now)


_GOAL_TRIGGERS = ("dẫn", "đưa tôi", "đưa tao", "đi tới", "đi đến",
                  "tới chỗ", "đến chỗ", "tìm đường", "chỉ đường")
_GOAL_CANCELS = ("dừng dẫn", "hủy dẫn", "huỷ dẫn", "thôi dẫn",
                 "tắt dẫn đường", "dừng tìm", "ngừng dẫn")


def _word_in(phrase, text):
    return re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text) is not None


def _parse_goal_command(text, current_goal=None):
    t = (text or "").lower().strip()
    if any(_word_in(c, t) for c in _GOAL_CANCELS):
        return ("stop", None)
    if not any(_word_in(tr, t) for tr in _GOAL_TRIGGERS):
        return None
    for lbl, vn in sorted(VIET_LABELS.items(), key=lambda kv: -len(kv[1])):
        if _word_in(vn, t):
            return ("start", lbl)
    if current_goal:
        return ("start", current_goal)
    return ("need_target", None)


# ===========================================================================
# OCR (EasyOCR) - tai luoi, chay offline, khong can API key
# ===========================================================================
_ocr_reader = None
_ocr_lock = threading.Lock()


def get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        with _ocr_lock:
            if _ocr_reader is None:
                import easyocr
                print("[OCR] Dang tai EasyOCR (vi, en)...")
                _ocr_reader = easyocr.Reader(["vi", "en"], gpu=torch.cuda.is_available())
                print("[OCR] San sang.")
    return _ocr_reader


def _ocr_preprocess(img):
    """Phong to + tang tuong phan de doc chu nho ro hon."""
    h, w = img.shape[:2]
    if max(h, w) < 1600:                       # anh nho -> phong to len ~1600px
        scale = 1600 / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    # tang tuong phan nhe bang CLAHE tren kenh sang
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        pass
    return img


def run_ocr(img):
    """Doc chu va sap xep lai theo thu tu doc (tren->duoi, trai->phai).

    Tra ve chuoi cac dong cach nhau bang ' / ' de doc tu nhien.
    """
    reader = get_ocr()
    proc = _ocr_preprocess(img)
    # detail=1 de lay vi tri + do tin cay; width_ths cao de gop chu thanh tu/cum;
    # mag_ratio phong to noi bo giup bat chu nho; paragraph=False de tu sap xep.
    results = reader.readtext(
        proc, detail=1, paragraph=False,
        text_threshold=0.5, low_text=0.3, link_threshold=0.4,
        mag_ratio=1.5, width_ths=0.8, ycenter_ths=0.6, add_margin=0.15,
    )
    items = []
    for box, text, conf in results:
        t = (text or "").strip()
        if not t or conf < 0.25:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        yc = sum(ys) / 4.0
        xc = sum(xs) / 4.0
        hh = max(ys) - min(ys)
        items.append({"t": t, "yc": yc, "xc": xc, "h": hh})
    if not items:
        return ""
    items.sort(key=lambda it: it["yc"])
    # gom thanh tung dong: cung dong neu tam y gan nhau (< 0.7 chieu cao chu)
    avg_h = sum(it["h"] for it in items) / len(items)
    line_gap = max(avg_h * 0.7, 12)
    lines, cur, base_y = [], [], items[0]["yc"]
    for it in items:
        if it["yc"] - base_y > line_gap and cur:
            lines.append(cur); cur = []
        cur.append(it); base_y = it["yc"]
    if cur:
        lines.append(cur)
    out = []
    for ln in lines:
        ln.sort(key=lambda it: it["xc"])           # trai -> phai trong dong
        out.append(" ".join(it["t"] for it in ln))
    return ", ".join(s for s in out if s.strip())


# ===========================================================================
# Hoi dap AI - OpenRouter Vision (tuy chon, can OPENROUTER_API_KEY)
# ===========================================================================
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
if not OPENROUTER_API_KEY:
    print("[AI] CANH BAO: chua co OPENROUTER_API_KEY (.env) -> che do hoi dap AI se loi.")

_FREE_VISION_MODELS = [
    "meta-llama/llama-4-maverick:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
]
_model_exhausted = {}

_SYSTEM_PROMPT = (
    "Bạn là mắt AI cho người khiếm thị, trả lời bằng tiếng Việt rõ ràng, đầy đủ, đúng trọng tâm. "
    "Khi người dùng hỏi quang cảnh/trước mặt có gì: nêu các vật, người, chướng ngại vật chính "
    "kèm vị trí tương đối (bên trái, bên phải, phía trước, xa, gần) trong 2-3 câu ngắn gọn — "
    "đừng trả lời cụt một hai chữ, cũng đừng kể lể dài dòng. "
    "Đọc chữ nếu có; hướng dẫn di chuyển trái/phải/thẳng/dừng; cảnh báo nguy hiểm. "
    "Không bịa khi không chắc; không tả màu sắc hay ngoại hình rườm rà. "
    "Không từ chối, không hỏi lại — đi thẳng vào nội dung."
)


def _is_exhausted(model):
    return (time.time() - _model_exhausted.get(model, 0)) < 86400


def _resize_image_b64(image_b64, max_side=512):
    raw = base64.b64decode(image_b64)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def _call_one_model(model, messages):
    payload = {"model": model, "messages": messages, "max_tokens": 180, "temperature": 0.3}
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://smarteye.local",
        "X-Title": "SmartEye Pro",
    }
    resp = http_requests.post(OPENROUTER_URL, json=payload, headers=headers, timeout=25)
    if not resp.ok:
        print(f"[AI] {model} HTTP {resp.status_code}: {resp.text[:200]}")
        resp.raise_for_status()
    return (resp.json()["choices"][0]["message"]["content"] or "").strip()


def query_ai(image_b64, user_text, history=None):
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    image_b64 = _resize_image_b64(image_b64)
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": user_text},
        ],
    })
    errors = []
    for m in _FREE_VISION_MODELS:
        if _is_exhausted(m):
            continue
        try:
            text = _call_one_model(m, messages)
            if text:
                return text
        except Exception as e:
            s = str(e)
            if "429" in s or "404" in s:
                _model_exhausted[m] = time.time()
            errors.append(f"{m}: {e}")
            print(f"[AI] {m} -> {e}")
    raise Exception(f"Tat ca model deu loi: {errors}")


# ===========================================================================
# Trang thai 1 thiet bi + worker suy luan
# ===========================================================================
_state = None
_state_lock = threading.Lock()


def get_state():
    global _state
    with _state_lock:
        if _state is None:
            print("[Model] Khoi tao CombinedModel...")
            _state = {
                "model": CombinedModel(),
                "infer_queue": queue.Queue(maxsize=1),
                "latest_alert": {"message": "", "timestamp": 0, "danger": False, "obj_label": None},
                "active_labels": [],
                "mode": "yolo",
                "nav_mode": False,
                "frame_count": 0,
                "last_spoken": {},
                "last_seen": {},
                "last_direction": {},
                "area_history": {},
                "depth_history": {},
                "last_approach_spoken": {},
                "depth_map": None,
                "_depth_ctr": 0,
                "last_nav_spoken": 0,
                "last_nav_message": "",
                "last_nav_src": None,
                "last_nav_key": None,
                "latest_frame_b64": None,
                "conversation_history": [],
                "goal_target": None,
                "goal_locked_tid": None,
                "goal_started": 0,
                "goal_last_seen": 0,
                "goal_last_dir": None,
                "goal_last_dist": None,
            }
            threading.Thread(target=_inference_worker, daemon=True).start()
        return _state


def _inference_worker():
    state = _state
    while True:
        frame = state["infer_queue"].get()
        try:
            _do_inference(state, frame)
        except Exception as e:
            import traceback
            print(f"[ERROR] {e}")
            traceback.print_exc()


def _do_inference(state, frame):
    dev_model = state["model"]
    frame_height, frame_width = frame.shape[:2]
    results = dev_model.track(frame, classes=DANGER_CLASS_IDS, verbose=False,
                              device=INFER_DEVICE, persist=True, tracker="botsort.yaml")
    now = time.time()

    # Do sau: dan duong/homing thi cap nhat day hon de phan ung kip vat can
    state["_depth_ctr"] += 1
    nav_active = state.get("nav_mode", False) or state.get("goal_target")
    depth_every = DEPTH_INTERVAL_NAV if nav_active else DEPTH_INTERVAL
    depth_fresh = _depth_model is not None and state["_depth_ctr"] % depth_every == 0
    if depth_fresh:
        try:
            state["depth_map"] = _estimate_depth(frame)
        except Exception as e:
            print(f"[Depth] Loi: {e}")
            depth_fresh = False

    # Thu thap detection nguy hiem
    raw_dets = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = dev_model.names[cls_id]
        if label not in DANGER:
            continue
        if label not in AUX_LABELS and float(box.conf[0]) < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        area = (x2 - x1) * (y2 - y1)
        tid = int(box.id[0]) if box.id is not None else -1
        raw_dets.append([x1, y1, x2, y2, label, area, tid])
        state["last_seen"][label] = now

    tracked = sorted(raw_dets, key=lambda d: (PRIORITY.get(d[4], 0), d[5]), reverse=True)
    state["active_labels"] = list({d[4] for d in tracked})

    def _det_key(d):
        return f"t{d[6]}" if d[6] >= 0 else f"aux_{d[4]}"
    alive_keys = {_det_key(d) for d in tracked}

    # area_history (fallback khi chua co depth)
    area_history = state["area_history"]
    for k in list(area_history.keys()):
        if k not in alive_keys:
            del area_history[k]
    for det in tracked:
        if det[6] < 0:
            continue
        area_history.setdefault(_det_key(det), deque(maxlen=TRACK_WINDOW)).append(det[5])

    # depth_history (tin hieu chinh phat hien tien lai)
    depth_map = state.get("depth_map")
    depth_history = state["depth_history"]
    for k in list(depth_history.keys()):
        if k not in alive_keys:
            del depth_history[k]
    if depth_fresh and depth_map is not None:
        dh, dw = depth_map.shape
        for det in tracked:
            cx = min(dw - 1, max(0, int((det[0] + det[2]) / 2)))
            cy = min(dh - 1, max(0, int((det[1] + det[3]) / 2)))
            d_m = _sample_depth(depth_map, cx, cy)
            if 0.2 < d_m < 30.0:
                depth_history.setdefault(_det_key(det), deque(maxlen=TRACK_WINDOW)).append(d_m)

    last_spoken = state["last_spoken"]
    last_approach_spoken = state["last_approach_spoken"]
    last_direction = state["last_direction"]

    alive_tids = {str(d[6]) for d in tracked if d[6] >= 0}
    for k in list(last_direction.keys()):
        if k not in alive_tids:
            del last_direction[k]
    for lbl in list(last_spoken.keys()):
        if now - state["last_seen"].get(lbl, 0) > 2.0:
            del last_spoken[lbl]

    # --- Vat dang tien lai (uu tien cao nhat) ---
    approach_fired = False
    for det in tracked:
        x1, y1, x2, y2, label, area, tid = det
        key = _det_key(det)
        approaching = False
        dhist = depth_history.get(key)
        if dhist is not None and len(dhist) >= TRACK_WINDOW:
            rises = sum(1 for i in range(len(dhist) - 1) if dhist[i + 1] > dhist[i] + 0.03)
            approaching = (dhist[-1] <= dhist[0] * APPROACH_DEPTH_RATIO
                           and dhist[-1] <= dhist[0] - APPROACH_MIN_DROP_M
                           and rises <= 1 and dhist[-1] < APPROACH_MAX_DIST_M)
        elif depth_map is None:
            ahist = area_history.get(key)
            if ahist is not None and len(ahist) >= TRACK_WINDOW:
                growth = sum(1 for i in range(len(ahist) - 1) if ahist[i + 1] > ahist[i] * 1.05)
                approaching = growth >= TRACK_WINDOW - 2 and ahist[-1] > ahist[0] * APPROACH_RATIO
        if approaching and now - last_approach_spoken.get(label, 0) > APPROACH_COOLDOWN:
            tid_key = str(tid)
            direction = get_direction(x1, x2, frame_width, last_direction.get(tid_key))
            last_direction[tid_key] = direction
            msg = build_approach_message(label, direction)
            _set_alert(state, msg, danger=True, obj_label=label)
            last_approach_spoken[label] = now
            last_spoken[label] = now
            print(f"[TIEP CAN] {msg}")
            approach_fired = True
            break

    if approach_fired:
        pass
    elif state.get("goal_target"):
        _goal_guidance(state, tracked, frame_width, frame_height, now, depth_map)
    elif state.get("nav_mode", False):
        _nav_guidance(state, tracked, frame_width, now)
    else:
        for det in tracked:
            x1, y1, x2, y2, label, _, tid = det
            tid_key = str(tid)
            direction = get_direction(x1, x2, frame_width, last_direction.get(tid_key))
            last_direction[tid_key] = direction
            if now - last_spoken.get(label, 0) > COOLDOWN:
                distance = get_distance_with_depth(x1, y1, x2, y2, label, depth_map)
                msg = build_message(label, direction, distance)
                _set_alert(state, msg, danger=label in ("car", "motorcycle", "bus", "truck", "stairs"),
                           obj_label=label)
                last_spoken[label] = now
                print(f"[YOLO] {msg}")
                break


def _alert_response(state):
    alert = dict(state["latest_alert"])
    alert["active_labels"] = state.get("active_labels", [])
    alert["nav_mode"] = state.get("nav_mode", False)
    alert["goal"] = state.get("goal_target")
    return jsonify(alert)


# ===========================================================================
# Routes
# ===========================================================================
@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image"}), 400
    state = get_state()
    state["latest_frame_b64"] = data['image']

    # Che do AI khong chay YOLO (tru khi dang dan toi muc tieu)
    if state.get("mode") == "ai" and not state.get("goal_target"):
        return _alert_response(state)

    state["frame_count"] += 1
    if state["frame_count"] % FRAME_SKIP != 0:
        return _alert_response(state)

    img_data = base64.b64decode(data['image'].split(',')[1])
    frame = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)

    q = state["infer_queue"]
    if q.full():
        try: q.get_nowait()
        except queue.Empty: pass
    q.put(frame)
    return _alert_response(state)


@app.route("/alert_state")
def alert_state():
    return _alert_response(get_state())


@app.route("/mode", methods=["POST"])
def set_mode():
    state = get_state()
    new_mode = (request.json or {}).get("mode", "yolo")
    if new_mode not in ("yolo", "ai"):
        return jsonify({"error": "Invalid mode"}), 400
    if new_mode != state.get("mode"):
        state["latest_alert"] = {"message": "", "timestamp": 0, "danger": False, "obj_label": None}
        state["last_spoken"].clear()
    state["mode"] = new_mode
    return jsonify({"mode": new_mode})


@app.route("/nav_mode", methods=["POST"])
def toggle_nav_mode():
    state = get_state()
    state["nav_mode"] = not state.get("nav_mode", False)
    on = state["nav_mode"]
    msg = "Bật chế độ dẫn đường" if on else "Tắt chế độ dẫn đường"
    _set_alert(state, msg, danger=False)
    return jsonify({"nav_mode": on})


@app.route("/ai_query", methods=["POST"])
def ai_query():
    data = request.json or {}
    user_text = data.get("query", "").strip()
    if not user_text:
        return jsonify({"error": "No query"}), 400
    state = get_state()

    # Lenh dan duong toi muc tieu -> xu ly cuc bo, khoi goi LLM
    cmd = _parse_goal_command(user_text, state.get("goal_target"))
    if cmd:
        action, label = cmd
        if action == "need_target":
            msg = "Bạn muốn tôi dẫn đường tới đâu? Hãy nói tên vật, ví dụ: cửa, ghế, hoặc cầu thang."
        elif action == "stop":
            state["goal_target"] = None
            state["goal_locked_tid"] = None
            msg = "Đã tắt dẫn đường tới mục tiêu"
        else:
            name = VIET_LABELS.get(label, label)
            resuming = (state.get("goal_target") == label)
            state["goal_target"] = label
            state["goal_locked_tid"] = None
            state["goal_started"] = time.time()
            state["goal_last_seen"] = time.time()
            state["goal_last_dir"] = None
            state["goal_last_dist"] = None
            state["last_nav_spoken"] = 0
            state["last_nav_message"] = ""
            msg = (f"Tiếp tục dẫn đường tới {name}" if resuming
                   else f"Bắt đầu dẫn đường tới {name}")
        now = time.time()
        state["latest_alert"] = {"message": msg, "timestamp": now, "danger": False, "obj_label": None}
        print(f"[GOAL] lenh: {msg!r}")
        return jsonify({"response": msg, "goal": action, "timestamp": now})

    image_b64 = data.get("image") or state.get("latest_frame_b64")
    if not image_b64:
        return jsonify({"error": "Chưa có hình ảnh từ camera"}), 400
    history = state.get("conversation_history", [])[-6:]
    try:
        response_text = query_ai(image_b64, user_text, history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    if not response_text:
        return jsonify({"error": "AI trả về rỗng"}), 500

    conv = state["conversation_history"]
    conv.append({"role": "user", "content": user_text})
    conv.append({"role": "assistant", "content": response_text})
    if len(conv) > 20:
        state["conversation_history"] = conv[-20:]

    now = time.time()
    state["latest_alert"] = {"message": response_text, "timestamp": now, "danger": False, "obj_label": None}
    return jsonify({"response": response_text, "timestamp": now})


@app.route("/ai_history", methods=["DELETE"])
def clear_ai_history():
    get_state()["conversation_history"] = []
    return jsonify({"cleared": True})


@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    """Doc chu trong anh (EasyOCR, offline, vi+en)."""
    try:
        data = request.get_json(force=True)
        raw = data["image"].split(",", 1)[-1]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(raw), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"ok": False, "error": "Không giải mã được ảnh"}), 400
        text = run_ocr(img)
        return jsonify({"ok": True, "text": text,
                        "sentence": text if text else "Không tìm thấy chữ nào."})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


_last_sos = {}

# --- Cau hinh gui email SOS (dat trong .env) ---
SMTP_HOST    = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.environ.get("SMTP_PORT", "465"))
SMTP_USER    = os.environ.get("SMTP_USER", "")          # gmail dung de GUI (vd ngovinhyb@gmail.com)
SMTP_PASS    = os.environ.get("SMTP_PASS", "")          # App Password 16 ky tu cua gmail do
SOS_EMAIL_TO = os.environ.get("SOS_EMAIL_TO", "ngovinhyb@gmail.com")  # nguoi nhan


def send_sos_email(lat, lng, maps_url, reason="SOS", acc=None):
    """Gui email canh bao (chay nen, khong chan response). Thieu cau hinh -> bo qua."""
    if not (SMTP_USER and SMTP_PASS):
        print("[SOS][Email] Chua cau hinh SMTP_USER/SMTP_PASS trong .env -> KHONG gui email")
        return
    import smtplib, ssl
    from email.message import EmailMessage
    try:
        acc_line = f"Sai so     : khoang +/- {round(acc)} m\n" if acc else ""
        msg = EmailMessage()
        msg["Subject"] = "🆘 SmartEye SOS - Yeu cau cuu tro"
        msg["From"] = SMTP_USER
        msg["To"] = SOS_EMAIL_TO
        msg.set_content(
            f"Nguoi dung SmartEye vua kich hoat canh bao ({reason}).\n\n"
            f"Vi tri GPS : {lat}, {lng}\n"
            f"{acc_line}"
            f"Ban do     : {maps_url}\n"
            f"Thoi gian  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"Hay lien he hoac toi ngay vi tri tren."
        )
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=15) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print(f"[SOS][Email] Da gui canh bao toi {SOS_EMAIL_TO}")
    except Exception as e:
        print(f"[SOS][Email] Loi gui email: {e}")


@app.route("/api/sos", methods=["POST"])
def api_sos():
    """Nhan toa do GPS, log, gui email cho nguoi than va tra ve link Google Maps."""
    try:
        data = request.get_json(force=True)
        lat, lng = float(data["lat"]), float(data["lng"])
        reason = (data.get("reason") or "SOS")
        try:
            acc = float(data.get("acc")) if data.get("acc") is not None else None
        except (TypeError, ValueError):
            acc = None
        maps_url = f"https://www.google.com/maps?q={lat},{lng}"
        _last_sos.update({"lat": lat, "lng": lng, "acc": acc, "url": maps_url, "time": time.time()})
        print(f"[SOS] Yeu cau cuu tro ({reason})! {maps_url}" + (f" (+/-{round(acc)}m)" if acc else ""))
        # Gui email o luong rieng de khong lam cham phan hoi cho nguoi dung
        threading.Thread(target=send_sos_email, args=(lat, lng, maps_url, reason, acc), daemon=True).start()
        email_on = bool(SMTP_USER and SMTP_PASS)
        note = "Đã gửi vị trí cho người thân." if email_on else "Đã ghi nhận vị trí (chưa bật gửi email)."
        return jsonify({"ok": True, "maps_url": maps_url, "email": email_on, "message": note})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/last-sos")
def api_last_sos():
    return jsonify({"ok": True, "data": _last_sos})


# ===========================================================================
# Main
# ===========================================================================
def get_lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


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
        name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"SmartEye")])
        cert = (x509.CertificateBuilder()
                .subject_name(name).issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
                .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
                .add_extension(x509.SubjectAlternativeName([
                    x509.DNSName(u"localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]), critical=False)
                .sign(key, hashes.SHA256()))
        with open(key_path, "wb") as f:
            f.write(key.private_bytes(serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption()))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        print(f"[SSL] Da tao cert moi: {cert_path}")
        return cert_path, key_path
    except ImportError:
        print("[SSL] Thieu 'cryptography', dung adhoc cert")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="SmartEye Pro AI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-ssl", action="store_true",
                        help="Tat HTTPS (chi test tren laptop bang localhost)")
    parser.add_argument("--no-preload", action="store_true",
                        help="Khong tai truoc model khi khoi dong")
    args = parser.parse_args()

    _load_depth_model()
    if not args.no_preload:
        get_state()  # nap YOLO + dung worker ngay

    ip = get_lan_ip()
    use_ssl = not args.no_ssl
    scheme = "https" if use_ssl else "http"
    print("=" * 64)
    print("  SmartEye Pro - Doi mat thu hai cho nguoi khiem thi")
    print("=" * 64)
    print(f"  May tinh nay : {scheme}://localhost:{args.port}")
    print(f"  Dien thoai   : {scheme}://{ip}:{args.port}  (cung mang wifi)")
    if use_ssl:
        print("  * HTTPS tu ky -> trinh duyet canh bao, bam 'Van tiep tuc'.")
    else:
        print("  * Mo bang dien thoai can HTTPS (bo --no-ssl) de camera/mic hoat dong.")
    print("=" * 64)

    if use_ssl:
        cert_f, key_f = _ensure_cert()
        ssl_ctx = (cert_f, key_f) if cert_f else 'adhoc'
    else:
        ssl_ctx = None
    app.run(host=args.host, port=args.port, ssl_context=ssl_ctx, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
