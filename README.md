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

# 📌 Tính năng chính

-  Quét vật thể bằng camera điện thoại
-  Gửi dữ liệu hình ảnh qua WiFi
-  Nhận diện vật thể bằng AI
-  Cảnh báo bằng giọng nói
-  Phát hiện vật cản nguy hiểm
-  Giao diện thân thiện trên điện thoại
-  Ứng dụng công nghệ AIoT thực tế


# 📌 Mô hình và thư viện

# 📌 Hướng dẫn tiếp cận

# 📌 Đồng sáng lập
| **Nguyễn Việt Tiến** |
**Ngô Quang Vinh** |
**Nguyễn Hoàng Hải** |
**Trương Nhật Nam** |
