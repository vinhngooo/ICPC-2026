# SmartEye - AIoT System

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)
[![Framework](https://img.shields.io/badge/framework-Flask-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

Hệ thống AIoT "Mắt thần" hỗ trợ người khiếm thị bằng công nghệ nhận diện vật thể, ước lượng khoảng cách và cảnh báo bằng giọng nói.

*SmartEye* ứng dụng mô hình điện toán biên (Edge Computing). Hệ thống sử dụng điện thoại thông minh làm thiết bị thu thập hình ảnh/nhận lệnh giọng nói/phát âm thanh, và sử dụng máy tính cá nhân (Server Local) làm "bộ não" trung tâm xử lý AI theo thời gian thực.

---

## 📌 Tóm Tắt Dự Án (Project Abstract)

Hệ thống được thiết kế để hỗ trợ định hướng và nhận biết môi trường xung quanh cho người khiếm thị. Phiên bản *SmartEye Pro* tập trung vào hiệu năng thực dụng, đẩy các tác vụ nặng về phía Server và tối ưu hóa phía Client (điện thoại).

* *Kiến trúc phân tán:*
    * *Frontend (Mobile/Trình duyệt):* Chịu trách nhiệm hoàn toàn về luồng thu thập ảnh (Camera), xử lý âm thanh (Web Speech API để đọc cảnh báo và nhận diện giọng nói tiếng Việt), và kích hoạt rung/SOS.
    * *Backend (Laptop/PC):* Xử lý hình ảnh gửi lên qua HTTP POST (Base64). Sử dụng các mô hình AI để nhận diện vật thể, đo độ sâu, đọc chữ, tính toán đường đi và trả về văn bản cảnh báo.
* *Tối ưu hóa phần cứng:*
    * *Các thành phần nặng (Metric3D, EasyOCR, AI Vision) được thiết kế suy biến mượt mà (Graceful Degradation) – nếu máy tính không đủ phần cứng hoặc thiếu cấu hình, hệ thống vẫn hoạt động trơn tru dựa trên cốt lõi YOLO.

---

## ⚙️ Tính Năng Nổi Bật

### 1. Chế độ Di chuyển & Cảnh báo (YOLO + Depth Navigation)
* *Nhận diện & Bám đuổi đa mô hình:* Kết hợp YOLO11n (COCO classes), mô hình Custom (Cửa, Cầu thang) và YOLO-World (Phát hiện Quạt điện). Sử dụng BoT-SORT để theo dõi quỹ đạo.
* *Cảnh báo tiến lại gần (Approach Warning):* Sử dụng lịch sử bản đồ độ sâu (Depth Map) hoặc biến thiên diện tích Bounding Box để phát hiện vật thể đang tiến lại gần (nguy cơ va chạm cao) và phát cảnh báo lập tức.
* *Dẫn đường tự do (Free-space Navigation):* Chia trường nhìn (FOV) thành 9 cột dựa trên bản đồ độ sâu từ Metric3Dv2. Đánh giá độ thoáng để đưa ra chỉ dẫn (Rẽ trái, Rẽ phải, Đi thẳng, Dừng lại).
* *Dẫn đường tới mục tiêu (Homing):* Người dùng có thể ra lệnh (VD: "Đưa tôi tới cái ghế"). Hệ thống sẽ khóa mục tiêu và chỉ đường chi tiết liên tục cho đến khi tiếp cận.

### 2. Chế độ Trợ lý Không gian (Vision AI & OCR)
* *Hỏi đáp bối cảnh:* Người dùng đặt câu hỏi bằng giọng nói, hệ thống chụp khung hình gửi lên các Mô hình Ngôn ngữ Lớn thị giác (Vision LLM qua OpenRouter API) để phân tích chi tiết môi trường trước mặt.
* *Đọc văn bản ngoại tuyến (Offline OCR):* Tích hợp EasyOCR để trích xuất văn bản từ hình ảnh biển báo, tài liệu, nhãn chai lọ,... mà không cần kết nối internet.

### 3. Tính năng An toàn (SOS)
* Gửi tín hiệu cấp cứu kèm Tọa độ GPS, đường dẫn Google Maps và độ sai số hiện tại.
* Tự động gửi Email thông báo khẩn cấp tới người thân (thông qua giao thức SMTP backend).

---

## 📌 Các Mô Hình & Thư Viện Cốt Lõi

| Tên Mô Hình / Thuật Toán | Vai Trò Trong Hệ Thống | Thư Viện / API | Ghi Chú |
| ------------- | ------------- | ------------- | ------------- |
| *YOLO11 Nano* (yolo11n.pt) | Nhận diện vật cản nền tảng (người, xe cộ, bàn ghế...) | ultralytics | Tối ưu hóa tốc độ chạy trên CPU/GPU onboard |
| *Custom YOLO & YOLO-World* | Phát hiện chuyên biệt: Cửa, Cầu thang, Quạt điện | ultralytics, torch | Có cơ chế bỏ phiếu (voting) để chống nhấp nháy tín hiệu |
| *BoT-SORT Tracker* | Định danh vật thể và duy trì ID qua các khung hình | ultralytics | persist=True |
| *Metric3Dv2* (metric3d_vit_small) | Ước lượng bản đồ chiều sâu đơn nhãn (Metric Depth) | torch.hub | Trích xuất độ sâu vật lý (mét) để tính toán va chạm |
| *EasyOCR* | Trích xuất và đọc văn bản (Tiếng Việt + Tiếng Anh) | easyocr, cv2 | Xử lý hoàn toàn Local (Offline) |
| *Vision LLM* (Llama-4/Nemotron) | Phân tích ngữ cảnh sâu và trả lời câu hỏi phức tạp | requests | Gọi qua OpenRouter API (Chiến lược fallback luân phiên) |
| *Web Speech API* | Text-to-Speech & Speech-to-Text | Native Browser API | Chạy phía Client, loại bỏ gánh nặng cho Server |

---

## 🚀 Hướng Dẫn Cài Đặt & Triển Khai

Hệ thống tích hợp sẵn cơ chế tự tạo chứng chỉ SSL (cryptography) để cho phép trình duyệt di động truy cập Camera qua mạng LAN (HTTPS) mà không cần bắt buộc dùng Ngrok.

### 1. Chuẩn Bị Môi Trường
Yêu cầu hệ thống cài đặt sẵn *Python 3.9+* (Khuyến khích máy có card đồ họa NVIDIA để chạy mượt mà Metric3D và EasyOCR).

Cài đặt các thư viện phụ thuộc:
```bash
pip install flask flask-cors ultralytics opencv-python numpy torch cryptography requests easyocr python-dotenv
```
(Đảm bảo bạn đã đặt các file trọng số doors.pt và stairs.pt ở cùng thư mục chứa source code)

### 2. Thiết Lập File Môi Trường (.env)
Tạo một file .env ở thư mục gốc và điền các thông tin sau:

Đoạn mã
API Key để dùng tính năng Hỏi đáp AI (Vision LLM)
```bash
OPENROUTER_API_KEY=your_openrouter_key_here
```
Cấu hình Email gửi thông báo SOS (Tùy chọn)
```bash
SMTP_HOST=smtp.gmail.com
SMTP_PORT=465
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_16_char_app_password
SOS_EMAIL_TO=relative_email@gmail.com
```
### 3. Khởi Chạy Server
Khởi động hệ thống trên máy tính:
Bash
python app.py
Dùng qua mạng LAN (Mặc định có HTTPS): Truy cập theo địa chỉ hiển thị trên terminal (Ví dụ: https://192.168.x.x:5000). Trình duyệt sẽ cảnh báo chứng chỉ tự cấp, bạn chọn "Nâng cao" -> "Vẫn tiếp tục".

Dùng Local trên máy tính: Chạy lệnh python app.py --no-ssl để truy cập qua http://localhost:5000.

### 4. Kết Nối Điện Thoại (Sử Dụng Thực Tế)
Điện thoại và máy tính phải kết nối cùng một mạng Wi-Fi (hoặc điện thoại phát 4G cho máy tính).

Dùng điện thoại truy cập vào địa chỉ HTTPS IP của máy tính hiển thị trên Terminal.

Cấp quyền truy cập Camera, Microphone và Vị trí (Location) khi trình duyệt yêu cầu.

## 📌 Đồng sáng lập
| Nguyễn Việt Tiến | Ngô Quang Vinh | Nguyễn Hoàng Hải | Trương Nhật Nam |
