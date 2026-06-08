# SmartEyes-AIOT

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/framework-Flask%20%2F%20FastAPI-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

Hệ thống AIoT hỗ trợ người khiếm thị bằng công nghệ nhận diện vật thể và cảnh báo bằng giọng nói.


**smartEyes-AI** là một hệ thống trí tuệ nhân tạo ứng dụng mô hình điện toán biên (Edge Computing) tận dụng tối đa phần cứng sẵn có. Hệ thống biến bất kỳ chiếc điện thoại thông minh nào thành "mắt và tai" (thu thập dữ liệu hình ảnh, âm thanh) và sử dụng máy tính cá nhân làm "bộ não" trung tâm để xử lý AI theo thời gian thực thông qua nền tảng Web Application.

---

## 📌 Tóm Tắt Dự Án (Project Abstract)

Việc di chuyển và định hướng trong không gian kín như lớp học đối với robot tự hành hoặc người khiếm thị đòi hỏi một hệ thống nhận diện vật cản nhanh và chính xác. **smartEYE-AI** thu hẹp phạm vi hoạt động chuyên biệt trong **môi trường lớp học** với các mục tiêu cốt lõi:
* **Đối tượng nhận diện chính (YOLOv8):** Tập trung học và phát hiện chính xác 4 yếu tố môi trường bao gồm: **Con người (Human)**, **Bàn học (Desk)**, **Bậc thang (Stairs)**, **Cánh cửa (Door)**, **Tường (Wall)**, và **Đường thẳng/Vạch kẻ đường (Lines)** để xác định lối đi an toàn.
* **Cơ chế hoạt động:**
    * Camera điện thoại stream luồng video thời gian thực lên Server Flask/FastAPI trên laptop thông qua giao thức WebSocket (độ trễ thấp).
    * Laptop chạy mô hình **YOLOv8 (Ultralytics)** để bounding box vật cản và tính toán khoảng cách/nguy cơ va chạm.
    * Nếu phát hiện vật cản quá gần hoặc lệch khỏi đường thẳng an toàn, server lập tức gửi tín hiệu hạ lệnh cho điện thoại phát âm thanh cảnh báo (ví dụ: *"Chú ý có bàn phía trước"*, *"Chuẩn bị tới bậc thang"*).
---

## ⚙️ Kiến Trúc Hệ Thống (System Architecture)
```text
[ Smartphone: Eye/Ear ] --(Stream Video Frames qua WebSockets)--> [ Laptop Server: Brain ]
           ▲                                                                 │
           └──────────────────(Hạ lệnh phát Audio Cảnh báo)──────────────────┘
---

1.  **Frontend (Mobile Web):** HTML5 Camera API (`getUserMedia`), Socket.io-client gửi frame ảnh dạng Base64/Binary, Web Audio API đảm nhận phát âm thanh cảnh báo.
2.  **Backend (Laptop):** Python, Flask-SocketIO nhận diện luồng ảnh liên tục, chuyển đổi sang định dạng OpenCV.
3.  **AI Core:** Thư viện `ultralytics` chạy mô hình **YOLOv8** (sử dụng phiên bản `yolov8n.pt` - Nano để tối ưu hóa tốc độ trên laptop cá nhân không có GPU rời).
```
## 📌 Tính năng (2 chế độ)
# 1. Chế độ cảnh báo
-  Nhận diện & Bám đuổi: Sử dụng Tracker BoT-SORT để theo dõi quỹ đạo di chuyển của vật thể, phân loại mức độ ưu tiên nguy hiểm (Xe cộ, Con người > Bàn ghế > Cửa, Cầu thang).
-  Cảnh báo va chạm chủ động: Theo dõi biến thiên diện tích Bounding Box (APPROACH_RATIO = 1.40). Nếu phát hiện vật thể tăng kích thước nhanh trong $4/5$ khung hình liên tiếp, hệ thống lập tức phát cảnh báo vật thể đang tiến lại gần.
-  Dẫn đường thông minh (VFH+ Mode): Khi kích hoạt, hệ thống sẽ chuyển sang chế độ phân tích không gian số để tìm lối đi an toàn nhất.

# 2. Chế độ an toàn
-  Chuyển đổi linh hoạt: Phục vụ trong không gian tĩnh (trong nhà, cửa hàng) khi người dùng cần tương tác sâu với môi trường xung quanh.
-  Hỏi đáp không gian: Người dùng gửi câu lệnh bằng giọng nói, hệ thống sử dụng Faster-Whisper để dịch văn bản, chụp khung hình hiện tại và gửi tới các Mô hình ngôn ngữ lớn thị giác (VLM) để phân tích chi tiết vật thể, đọc văn bản hoặc tìm đồ vật theo yêu cầu.



# 📌 Mô hình và thư viện
Tên Mô Hình / Thuật Toán|	Vai Trò Trong Hệ Thống|	Thư Viện Chính|	Ghi Chú
YOLOv11 Nano (yolo11n.pt)|	Phát hiện vật cản nền tảng (80 lớp COCO)|	ultralytics|	Tối ưu hóa tốc độ chạy trên CPU/GPU onboard
Custom YOLO Ensembles|	Phát hiện chuyên biệt doors.pt và stairs.pt|	torch, ultralytics|	Merge Tensor song song với luồng chính (_merge_boxes)
BoT-SORT Tracker|	Định danh vật thể và tính toán vector chuyển động|	ultralytics|	Duy trì ID vật thể qua các khung hình (persist=True)
Metric3Dv2 (metric3d_vit_small)|	Ước lượng bản đồ chiều sâu đơn nhãn (Metric Depth)|	torch.hub|	Tính khoảng cách vật lý thực tế bằng mét nhờ focal length PX
VFH+ (Vector Field Histogram+)|	Xây dựng biểu đồ phân vùng 180° và tìm lối đi an toàn|	numpy, math|	Thuật toán tránh vật cản tự hành phân chia thành 36 bins
Faster-Whisper (small)|	Nhận diện giọng nói local độ chính xác cao|	faster-whisper|	Tích hợp bộ lọc VAD (Voice Activity Detection)
Edge-TTS (vi-VN-HoaiMyNeural)|	Tổng hợp tiếng nói cảnh báo tiếng Việt tự nhiên|	edge_tts, asyncio|	Tăng tốc độ đọc rate=+15%, có cơ chế tạo trước âm thanh (pregenerate_audio)
VLM (OpenRouter API)|	Phân tích ngữ cảnh sâu thông qua mô hình thị giác|	requests|	Chiến lược fallback luân phiên: Llama-4-Scout / Nemotron-12b


# 📌 Hướng Dẫn Tiếp Cận & Triển Khai
Do đặc thù các trình duyệt di động hiện nay bắt buộc phải có kết nối HTTPS bảo mật thì mới cho phép kích hoạt Camera (getUserMedia), chúng ta sẽ sử dụng Ngrok để tạo một đường hầm bảo mật từ Laptop ra Internet cho điện thoại kết nối.
# 1. Chuẩn bị môi trường và cài đặt
Yêu cầu hệ thống cài đặt sẵn Python 3.9 trở lên (Khuyến khích máy có card đồ họa NVIDIA để chạy mượt mà Metric3D và Whisper).
- Cài đặt các gói thư viện phụ thuộc:
```text
pip install flask flask-cors ultralytics faster-whisper edge-tts opencv-python numpy torch cryptography requests cv2
```
(Lưu ý: Đảm bảo bạn đã đặt các file trọng số bổ sung doors.pt và stairs.pt ở cùng thư mục chạy script nếu cần nhận diện nâng cao.)
# 2. Khởi chạy Local Server
Hệ thống tích hợp sẵn cơ chế tự động tạo chứng chỉ SSL nội bộ (cert.pem, key.pem). Chạy lệnh sau để khởi động server:
```text
python app.py
```
(Server sẽ chạy mặc định tại cổng 5000. Hãy ghi lại địa chỉ IP Local hiển thị trên terminal (Ví dụ: https://192.168.62.197:5000).)
# 3. Cấu hình Đường hầm HTTPS với Ngrok
Mở một terminal mới và chạy lệnh Ngrok để trỏ thẳng vào cổng Flask đang chạy dưới giao thức HTTPS:
```text
ngrok http https://localhost:5000
```
(Nếu bạn cấu hình chạy server không có SSL bằng cờ --no-ssl, hãy dùng lệnh: ngrok http 5000)
# 4. Kết nối Thiết bị
- Dành cho Người dùng (Điện thoại): Sử dụng điện thoại quét mã hoặc truy cập trực tiếp link Ngrok kèm ID thiết bị để vào giao diện camera:
  - Thiết bị 1: https://xxxx-xxx-xxx.ngrok-free.app/phone?id=1
  - Thiết bị 2: https://xxxx-xxx-xxx.ngrok-free.app/phone?id=2
- Dành cho Giám sát viên (PC/Laptop): Truy cập link sau trên máy tính để xem màn hình điều khiển trung tâm và camera trực quan:
  - Giao diện Monitor: https://xxxx-xxx-xxx.ngrok-free.app/monitor



  
# 📌 Đồng sáng lập
| **Nguyễn Việt Tiến** |
**Ngô Quang Vinh** |
**Nguyễn Hoàng Hải** |
**Trương Nhật Nam** |
