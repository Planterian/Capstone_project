# Kết nối Camera KV260

Hoàn toàn **CÓ THỂ dùng camera của laptop**, nhưng cách thức truyền dữ liệu và kiến trúc phần mềm sẽ khác một chút so với việc cắm Webcam rời trực tiếp vào bo mạch KV260.

---

##### 💡 So sánh 2 cách triển khai:

###### 1. Phương án 1: Dùng Webcam rời cắm trực tiếp vào KV260 (Chuẩn Edge AI & Đơn giản nhất)
* **Luồng xử lý**: KV260 tự mở camera bằng OpenCV (`cv2.VideoCapture(0)`), tự lấy ảnh, đẩy xuống FPGA PL tính toán và xuất màn hình.
* **Ưu điểm**: Đơn giản, không phụ thuộc vào Laptop, trễ truyền nhận bằng 0.

###### 2. Phương án 2: Dùng Camera có sẵn trên Laptop (Truyền qua mạng LAN)
Nếu chưa có Webcam rời và muốn tận dụng camera của Laptop, bạn vẫn làm được bằng mô hình **Client - Server qua Mạng LAN**:
* **Luồng xử lý**:
    1. **Trên Laptop**: Chạy một đoạn script Python đơn giản đọc camera laptop bằng OpenCV, sau đó **stream (gửi) khung hình qua mạng Wi-Fi/LAN** (dùng Socket, RTSP, hoặc HTTP) tới địa chỉ IP của KV260.
    2. **Trên KV260**: Script Python trên KV260 nhận dữ liệu khung hình từ mạng LAN, nạp vào bộ nhớ CMA, kích hoạt FPGA PL tăng tốc AI, sau đó gửi kết quả (nhãn/bounding box) ngược lại Laptop hoặc hiển thị ra màn hình.
* **Nhược điểm**: Phát sinh thêm một chút độ trễ (Latency) do truyền tải dữ liệu qua mạng nội bộ và cần viết thêm vài dòng code stream video trên Laptop.

---

##### 🛠️ Mẹo hay: Dùng Điện Thoại làm Webcam cho KV260 (Không tốn chi phí)
Nếu không muốn viết thêm code stream video qua mạng từ Laptop, bạn có thể biến chính **chiếc Điện Thoại (Android / iPhone)** của mình thành một chiếc USB Webcam rời cho KV260:
1. Cài ứng dụng biến điện thoại thành webcam (như **DroidCam**, **Iriun Webcam** hoặc **IP Webcam**).
2. Cắm dây cáp sạc USB từ điện thoại vào cổng USB của KV260 (hoặc bắt chung Wi-Fi).
3. Linux trên KV260 sẽ nhận diện điện thoại như một thiết bị `/dev/video0` chuẩn UVC, giúp bạn chạy code OpenCV trực tiếp trên KV260 mà **không cần mua thêm Webcam rời**.
