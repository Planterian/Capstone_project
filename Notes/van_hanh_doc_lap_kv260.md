# Vận hành Độc lập KV260

**Vẫn CẦN Laptop/PC trong quá trình lập trình và phát triển**, nhưng **KHÔNG CẦN Laptop khi hệ thống chạy demo/thực tế (Standalone)**.

---

##### 1. Tại sao VẪN CẦN Laptop/PC khi phát triển?
* **Tổng hợp vi mạch (Vivado Synthesis & Implementation)**: Để biên dịch các file mã nguồn **SystemVerilog RTL** thành file phần cứng **Bitstream (`.bit`)**, bạn bắt buộc phải dùng phần mềm **AMD Vivado** chạy trên Laptop/PC x86 (cấu hình khuyến nghị CPU 4-8 nhân, RAM 16GB+). Chip ARM PS trên KV260 không đủ tài nguyên để tự chạy Vivado.
* **Soạn thảo mã nguồn & Kiểm thử (IDE & Simulation)**: Bạn viết code RTL, script Python trên VS Code, chạy mô phỏng Cocotb / Verilator trên Laptop trước khi đẩy xuống phần cứng.
* **Giao tiếp & Điều khiển KV260**: Laptop đóng vai trò là Terminal để điều khiển KV260 từ xa qua **SSH**, **Jupyter Notebook**, hoặc **VS Code Remote Development** (thông qua cáp mạng LAN hoặc USB-C).

---

##### 2. Khi nào thì KHÔNG CẦN Laptop nữa?
Khi bạn đã hoàn tất lập trình và nạp file `.bit` + script Python vào thẻ nhớ của KV260, bo mạch **Kria KV260 có thể vận hành hoàn toàn độc lập (Standalone Embedded System)**:
1. **Cắm Webcam USB** trực tiếp vào cổng USB trên KV260.
2. **Cắm màn hình** trực tiếp vào cổng HDMI / DisplayPort của KV260.
3. **Cấp nguồn cho KV260**: Lõi ARM PS tự khởi động Ubuntu/PYNQ, nạp Bitstream lên FPGA PL, đọc hình ảnh từ Webcam và xuất kết quả nhận diện real-time lên màn hình **mà không cần nối với bất kỳ chiếc Laptop nào**.
