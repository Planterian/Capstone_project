# Remote VCU118 — hướng dẫn trên client

## Phạm vi và giới hạn truy cập

**Client VCU118 chỉ được nạp FPGA qua hw_server và gửi/nhận dữ liệu UART qua TCP. Không được đăng nhập, truy cập file hoặc điều khiển PC host.**

- Được dùng `100.124.168.26:3122` trong Vivado Hardware Manager để nạp `.bit` cho VCU118.
- Được dùng `100.124.168.26:4002` để gửi dữ liệu tới FPGA và nhận UART trả về.
- Không dùng SSH, RDP/xrdp, SCP/SFTP, remote desktop hoặc shell trên PC host; không khởi động, dừng, sửa dịch vụ hay cấu hình host.
- Không dùng quyền truy cập VCU118 để truy cập KV260 hoặc các dịch vụ khác trên host.
- Khi có lỗi dịch vụ/cáp/nguồn, báo quản trị viên. Client không tự quản trị host.

Đây là quy định sử dụng. Tài liệu không xác nhận ACL/firewall đã được giới hạn tương ứng; quản trị viên cần áp dụng quyền mạng để client VCU118 chỉ tới được hai cổng trên. Tách port/serial tự nó không tạo phân quyền người dùng.

## Thông tin kết nối

| Mục | Giá trị |
|---|---|
| Host Tailscale | `100.124.168.26` |
| JTAG | TCP `3122` |
| UART | TCP raw `4002` |
| Serial JTAG | `210308BA534D` |
| FPGA | `xcvu9p_0` |
| Part project mẫu | `xcvu9p-flga2104-2L-e` |
| Board preset | `xilinx.com:vcu118:part0:3.0` |

Cập nhật: 23/10/2026. Client Windows, PC host Ubuntu tại lab.

## Chuẩn bị client

- Tailscale: thiết bị được cấp quyền truy cập các dịch vụ cần dùng.
- Vivado 2026.1 để thiết kế/build, hoặc Vivado Lab 2026.1 khi chỉ nạp/debug bitstream có sẵn. Cấu hình host hiện dùng 2026.1.
- Board nối USB vào host; client không cần cable drivers cho kết nối remote này.
- Host phải bật, có mạng và không Sleep/Suspend. Dịch vụ chạy nền, không phụ thuộc cửa sổ terminal; khóa màn hình vẫn dùng được.
- Không ghi mật khẩu hoặc nội dung license vào README chia sẻ.

Python 3 và `vcu118_uart_client.py` cần cho terminal UART mẫu. Không cần Vitis cho thiết kế Verilog UART thuần.

## License

Máy tự build XCVU9P phải có license hỗ trợ chip và phiên bản Vivado đang dùng. License node-locked gắn với PC build, không phải PC host làm cầu nối. Chỉ nạp bằng Vivado Lab/hw_server không cần license build.

Trang quản lý license: [AMD Product Licensing](https://account.amd.com/en/forms/license/license-form.html).

## Kiểm tra mạng

```powershell
tailscale ping 100.124.168.26
Test-NetConnection 100.124.168.26 -Port 3122
Test-NetConnection 100.124.168.26 -Port 4002
```

Cổng cần dùng phải có `TcpTestSucceeded : True`. TCP thông chưa chứng minh FPGA có bitstream UART đúng.

## Kết nối Vivado

Mở Hardware Manager → Tcl Console:

```tcl
open_hw_manager
connect_hw_server -url 100.124.168.26:3122
get_hw_targets
open_hw_target [get_hw_targets *210308BA534D]
get_hw_devices
```

Nếu đang nối server khác, chạy `disconnect_hw_server` trước khi kết nối. Kết quả mong đợi: `xcvu9p_0`.

### Nạp bitstream

1. Trong Hardware Manager, kiểm tra đúng server 3122 và đúng target `210308BA534D`.
2. Chuột phải `xcvu9p_0` → **Program Device**.
3. Chọn file `.bit` nằm trên PC client; không cần chép file sang host bằng tay.
4. Chỉ chọn `.ltx` tương ứng nếu thiết kế có các debug probe cần file đó. Bài UART hiện tại không có ILA, không cần `.ltx`.
5. Giữ JTAG clock 15 MHz như lần thử thành công; nếu lỗi giao tiếp có thể thử giảm còn 5 MHz.
6. Chọn **Program**, kiểm tra kết quả thành công.

Project mẫu `vcu_uart_test` dùng part `xcvu9p-flga2104-2L-e`, board preset `xilinx.com:vcu118:part0:3.0`, top `vcu118_uart_top`. File mẫu được tạo tại:

```text
C:/Users/KTMT/Vivado_Project/vcu_uart_test/vcu_uart_test.runs/impl_1/vcu118_uart_top.bit
```

Đây là đường dẫn trên PC đã tạo project; client khác phải chọn vị trí file của mình. Nạp `.bit` vào FPGA không đồng nghĩa ghi vào flash khởi động; sau mất nguồn thường cần nạp lại nếu chưa có cấu hình boot từ flash phù hợp.

## UART tương tác

### Thông số cố định

| Mục | Giá trị |
|---|---|
| TCP host | `100.124.168.26` |
| TCP port | `4002` |
| UART vật lý trên host | 115200 baud, 8 data bits, no parity, 1 stop bit |
| Flow control | Không dùng |
| USB board | Cổng UART J4, riêng với cổng USB JTAG |
| Kênh CP2105 | SCI/Standard, interface `if01` nối FPGA |
| Giới hạn kết nối | Một client TCP tại một thời điểm |

Chương trình client dùng TCP thuần, không phải Telnet/RFC2217. Client không cần COM port và không cần đặt baud trên PC; baud đã cố định tại bridge host và phải khớp RTL FPGA.

### Chạy Python trong Downloads

Đặt file `vcu118_uart_client.py` vào Downloads trên Windows rồi mở PowerShell:

```powershell
cd "$env:USERPROFILE\Downloads"
py .\vcu118_uart_client.py
```

Nếu không có lệnh `py`, thử:

```powershell
python .\vcu118_uart_client.py
```

Không cần `pip install` hoặc pyserial. Nếu cả hai lệnh không có, cài Python 3 trước.

Ví dụ:

```text
Da ket noi VCU118 UART: 100.124.168.26:4002
Gui > hello
FPGA > OK
Gui > test 123
FPGA > OK
Gui > /quit
```

- Nhập nội dung, nhấn Enter; chương trình tự thêm LF (`\n`).
- Bitstream mẫu trả `OK\r\n` cho mỗi dòng kết thúc bằng LF; nó chưa xử lý nội dung thành phép tính hay lệnh khác.
- Dùng từng dòng và chờ phản hồi. `/quit`, `/exit` hoặc Ctrl+C để thoát.
- Chương trình chờ phản hồi tối đa 5 giây mặc định; quá hạn/lỗi sẽ đóng kết nối. Có thể tăng bằng `py .\vcu118_uart_client.py --timeout 10`.
- Đóng mọi terminal/client khác đang dùng 4002 trước khi mở chương trình.
- Chương trình mẫu gửi và đọc theo dòng; với giao thức nhị phân hoặc dữ liệu phát liên tục cần thay chương trình client tương ứng. Bridge host vẫn chuyển nguyên byte.

### Quy ước phần cứng cho project mới

| Tín hiệu theo phía FPGA | Pin | I/O standard |
|---|---|---|
| UART RX — nhận từ CP2105 | AW25 | LVCMOS18 |
| UART TX — gửi tới CP2105 | BB21 | LVCMOS18 |
| Clock mẫu 300 MHz P/N | G31 / F31 | DIFF_SSTL12 |

Project mẫu dùng Clocking Wizard tạo 100 MHz cấp UART. Nếu đổi clock cấp UART, phải đổi tham số `CLK_HZ` tương ứng để baud vẫn đúng.

Giữ đúng pin UART, baud và định dạng 8N1 thì không cần sửa host khi đổi project. Client có thể cần đổi bộ mã hóa/giải mã nếu giao thức ứng dụng thay đổi. Thiết kế nên giữ UART TX ở mức nghỉ cao khi reset; tránh gửi lệnh trong lúc nạp lại FPGA.

## Xử lý lỗi

| Hiện tượng | Hành động trên client |
|---|---|
| Không tới được host/cổng dịch vụ | Kiểm tra Tailscale và quyền được cấp; báo quản trị viên nếu vẫn lỗi |
| Không thấy target | Kiểm tra đúng port 3122 và serial; nhờ quản trị viên kiểm tra nguồn/cáp/dịch vụ |
| Kết nối TCP nhưng không có OK | Kiểm tra đã nạp đúng bitstream, gửi LF, đúng clock/baud/pin RTL |
| TCP bị từ chối hoặc đóng ngay | Đóng client UART khác; bridge chỉ cho một client |
| Timeout | Đóng chương trình rồi kiểm tra lại thiết kế; báo quản trị viên nếu nghi lỗi bridge |
| Mất phản hồi sau mất nguồn | Nạp lại bitstream nếu chưa cấu hình flash boot |

## Trạng thái kiểm chứng

Đã nạp bitstream VCU118 và kiểm chứng PowerShell gửi dòng qua TCP 4002, FPGA trả `OK`. Python client đã được kiểm thử với server mô phỏng; chưa ghi nhận xác nhận chạy Python với board thật trong tài liệu nguồn. Chưa kiểm chứng tự phục hồi trong mọi tình huống reboot/rút cáp.
