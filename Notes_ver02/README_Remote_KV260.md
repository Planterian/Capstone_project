# Remote KV260 — hướng dẫn trên client

## Thông tin kết nối

| Mục | Giá trị |
|---|---|
| PC host Tailscale | `100.124.168.26` |
| JTAG KV260 | `100.124.168.26:3121` |
| Serial JTAG | `XFL1E32DB5TUA` |
| Thiết bị | `xck26_0`, `arm_dap_1` |
| IP LAN KV260 | `192.168.1.179` |
| Username KV260 | `ubuntu` |
| Hostname đã ghi nhận | `kv260-XFL1E32DB5TU` |
| SSH jump host | `ktmt@100.124.168.26` — chỉ cho tài khoản được cấp quyền KV260 |

`192.168.1.179` là địa chỉ LAN, không mặc nhiên truy cập được từ nhà. Hướng dẫn dùng SSH jump qua host cho người có quyền KV260. Quyền này không áp dụng cho client chỉ được cấp VCU118.

Cập nhật: 23/10/2026. Client Windows, PC host Ubuntu tại lab.

## Chuẩn bị client

- Tailscale: thiết bị được cấp quyền truy cập các dịch vụ cần dùng.
- Vivado 2026.1 để thiết kế/build, hoặc Vivado Lab 2026.1 khi chỉ nạp/debug bitstream có sẵn. Cấu hình host hiện dùng 2026.1.
- Board nối USB vào host; client không cần cable drivers cho kết nối remote này.
- Host phải bật, có mạng và không Sleep/Suspend. Dịch vụ chạy nền, không phụ thuộc cửa sổ terminal; khóa màn hình vẫn dùng được.
- Không ghi mật khẩu hoặc nội dung license vào README chia sẻ.

Cài OpenSSH Client (`ssh -V` để kiểm tra). Python điều khiển board chạy trên Ubuntu ARM của KV260; không phải Python UART VCU118 trên Windows.

## License

Theo cấu hình đã ghi nhận, KV260 được sử dụng với Vivado BASIC 2026.1. Kiểm tra thêm license nếu dùng IP có bản quyền. Vivado Lab/hw_server dùng cho nạp/debug không cần license build.

## Kiểm tra mạng và JTAG

```powershell
tailscale ping 100.124.168.26
Test-NetConnection 100.124.168.26 -Port 3121
```

Mở Hardware Manager → Tcl Console:

```tcl
open_hw_manager
connect_hw_server -url 100.124.168.26:3121
get_hw_targets
open_hw_target [get_hw_targets *XFL1E32DB5TUA]
get_hw_devices
```

Nếu đang kết nối server khác, chạy `disconnect_hw_server` trước. Kết quả đã kiểm chứng: `xck26_0 arm_dap_1`.

**KV260 đang chạy Ubuntu:** trước khi nạp PL cần đối chiếu bitstream với overlay/driver, AXI, clock/reset và điều khiển quạt đang dùng. Không reset PS, tải ELF hay chạy FSBL theo quy trình bare-metal một cách tùy tiện. Tài liệu xác nhận kết nối JTAG, chưa xác nhận bitstream mới tương thích với Linux đang chạy.

## SSH vào Ubuntu trên KV260

Chỉ dùng tài khoản và quyền jump host do quản trị viên cấp. Từ PowerShell:

```powershell
ssh -J ktmt@100.124.168.26 ubuntu@192.168.1.179
```

Lệnh mở phiên shell trên KV260 qua host. Nếu dùng mật khẩu, có thể được hỏi mật khẩu host rồi mật khẩu board; mật khẩu được cung cấp riêng. Lần đầu cần đối chiếu fingerprint với quản trị viên.

Xác nhận board sau đăng nhập:

```bash
whoami
hostname
uname -m
python3 --version
```

Giá trị đã ghi nhận: `ubuntu`, `kv260-XFL1E32DB5TU`, `aarch64`, `Python 3.10.12`. Chạy chương trình Python của board bằng `python3 ten_file.py`; dùng `exit` để thoát. IP LAN do người dùng cung cấp ngày 23/10/2026, chưa được thử kết nối lại trong lần sửa tài liệu này.

## Những phần đã chuẩn bị

- Host có dịch vụ `hw-server-kv260.service`, chạy nền và lọc serial KV260 trên port 3121.
- Board không cần Tailscale khi dùng đường SSH jump qua host.
- Chưa cấu hình UART–TCP riêng cho KV260 trong tài liệu này.
- Khi cần kiểm tra hoặc sửa dịch vụ host, liên hệ quản trị viên; việc dùng jump host không mặc nhiên cấp quyền quản trị PC host.

## Xử lý lỗi

| Hiện tượng | Cách kiểm tra |
|---|---|
| Không tới được host | Tailscale, quyền tailnet, nguồn/mạng host |
| Port 3121 không thông | Báo quản trị viên kiểm tra dịch vụ và quyền mạng |
| Không thấy target | Đúng serial/port; nhờ kiểm tra nguồn và cáp USB |
| SSH lỗi trước khi vào board | Phân biệt lỗi xác thực/quyền jump host với lỗi SSH tới KV260 |
| Host thông nhưng KV260 không SSH được | Nhờ kiểm tra nguồn, IP LAN, mạng và SSH của board |
| KV260 mất phản hồi sau thời gian idle | Báo quản trị viên kiểm tra log/kernel và cấu hình idle trên board |

## Ghi chú ổn định KV260

KV260 từng có kernel `5.15.0-1080-xilinx-zynqmp` bị soft lockup, stack liên quan BPF JIT và đồng bộ CPU. Chưa kết luận nguyên nhân gốc; không quy lỗi cho Tailscale.

Đã thử tắt riêng `state1 = cpu-sleep-0` trên cả bốn CPU, giữ `state0 = WFI`. Sau thử nghiệm, board có uptime 2 giờ 13 phút và không thấy log soft/hard lockup hoặc RCU stall trong lần kiểm tra đó. Đây là bằng chứng khả quan, không phải bảo đảm không bao giờ treo.

Dịch vụ `/etc/systemd/system/kv260-disable-deep-idle.service` trên **KV260** đã được enable và chạy thành công. Cần kiểm chứng thiết lập sau reboot bằng:

```bash
systemctl status kv260-disable-deep-idle.service --no-pager
grep -H . /sys/devices/system/cpu/cpu[0-9]*/cpuidle/state1/disable
uptime
sudo journalctl -b -k --no-pager | grep -Ei 'soft lockup|hard lockup|rcu.*stall'
```

Cả bốn giá trị `disable` phải là `1`. Tắt trạng thái nghỉ sâu có thể tăng điện năng/nhiệt nhàn rỗi. Chưa có kết quả kiểm chứng sau reboot trong hồ sơ phiên này.

## Trạng thái kiểm chứng

Đã nhận JTAG `xck26_0` và `arm_dap_1`; dịch vụ host chạy nền. Chưa xác nhận đầy đủ khả năng tự phục hồi sau reboot/rút cáp, cũng chưa xác nhận bitstream mới chạy cùng Linux. Kiểm tra ổn định sau reboot còn cần thực hiện như ghi chú trên.
