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
# 1. Chế độ an toàn
-  Nhận diện vật thể khi cần thiết
-  Sử dụng khi người dùng đến các nơi an toàn như trong cửa hàng hoặc ở nhà
-  Người dùng sẽ hỏi vị trí của đồ vật, nếu nhìn thấy, AI sẽ báo cho người dùng

# 2. Chế độ cảnh báo
-  Sử dụng khi người dùng tham gia giao thông, hoạt động ở ngoài
-  Cảnh báo bằng giọng nói khi có vật cản hoặc xe cộ nguy hiểm
-  Phát hiện các biển báo và đọc thông tin biển báo cho người dùng



# 📌 Mô hình và thư viện

# 📌 Hướng dẫn tiếp cận

# 📌 Đồng sáng lập
| **Nguyễn Việt Tiến** |
**Ngô Quang Vinh** |
**Nguyễn Hoàng Hải** |
**Trương Nhật Nam** |
