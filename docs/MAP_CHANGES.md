# Thay đổi map SUMO (2026-11-08)

Tài liệu cho team review. Mạng mới: `sumo/map/hcmc.net.xml.gz`, dựng bằng `build_network.py`.
Map OSM cũ (`osm.net.xml.gz`) giữ lại, chỉ dùng làm nguồn hình học.

## 1. Vì sao phải sửa

| Vấn đề ở map OSM cũ | Hậu quả |
|---|---|
| Một ngã tư bị tách thành 2–5 nút đèn cách nhau vài mét (15 đèn cho 5 ngã tư) | Một ngã tư thật có nhiều đèn chạy độc lập |
| Kế hoạch đèn của các mảnh không phối hợp (cùng chu kỳ 90 s, lệch pha 0) | Có hướng không bao giờ xanh suốt đường qua nút, xe kẹt giữa hai đèn |
| Đoạn nối giữa các mảnh chỉ dài 0,2–5 m | Không có chỗ chứa xe; teleport ngay cả với đèn cố định (heavy: 288 teleport, ATT 516 s) |
| Hàng trăm hẻm, đường dịch vụ; có người đi bộ | Nhiễu, chạy chậm, không thuộc phạm vi nghiên cứu |
| Xe máy chỉ có `vClass`, không có thông số chiều ngang | Xe máy xếp hàng dọc như ô tô nhỏ |

## 2. Đặc tả đã được xác nhận

### Hạ tầng
- 6 nút giao, mỗi nút **một nút SUMO và một đèn**. Vị trí giữ theo map cũ.
- **Lý Thường Kiệt, Bắc Hải:** dải phân cách cứng, mỗi chiều 2 làn × 3,5 m (≈ 3 ô tô / 2 xe tải dàn ngang).
- **Các đường còn lại** (Thành Thái, Tô Hiến Thành, Lữ Gia, Đường Số 2, Đồng Nai): vạch sơn, mỗi chiều 2 làn × 3,0 m (cả hai chiều ≈ 5 ô tô / 4 xe tải).
- **Hẻm 319A:** giữ lại vì là một nhánh của J0, mỗi chiều 1 làn × 3,0 m.
- Bỏ toàn bộ hẻm và đường nhỏ khác. Bỏ người đi bộ. Không quay đầu.
- Hình dạng đường lấy theo map OSM. Nhánh biên cắt ở 400 m; nhánh biên ngắn kéo dài thẳng đến khoảng 150 m.

### Luật rẽ và đèn ("tự do" = không có đèn, luôn được đi, nhường xe đang có đèn xanh)

**J0 – Lý Thường Kiệt × Bắc Hải** (LTK Bắc–Nam, Bắc Hải hướng đông, Hẻm 319A hướng tây)
| Từ | Rẽ trái | Đi thẳng | Rẽ phải |
|---|---|---|---|
| Bắc Hải | vào LTK Nam: **đèn** | vào 319A: **cấm** | vào LTK Bắc: tự do |
| LTK Bắc | vào Bắc Hải: **đèn** | LTK Nam: tự do | vào 319A: tự do |
| LTK Nam | vào 319A: **cấm** | LTK Bắc: **đèn** | vào Bắc Hải: tự do |
| Hẻm 319A | tự do | tự do | tự do |

Pha: (1) Bắc Hải rẽ trái · (2) LTK Bắc rẽ trái · (3) LTK Nam đi thẳng.

**J1 – Lý Thường Kiệt × Đường Số 2** (ngã ba, Đường Số 2 nhánh tây): mỗi đầu vào một đèn điều khiển mọi hướng.
Pha: (1) LTK hai chiều, rẽ trái nhường xe ngược chiều · (2) Đường Số 2.

**J2 – Lý Thường Kiệt × Tô Hiến Thành × Lữ Gia** (LTK Bắc–Nam, Lữ Gia tây, THT đông)
| Từ | Đèn điều khiển | Tự do |
|---|---|---|
| LTK Bắc | đi thẳng + rẽ trái vào THT | rẽ phải vào Lữ Gia |
| Lữ Gia | đi thẳng THT + rẽ trái vào LTK Bắc | rẽ phải vào LTK Nam |
| LTK Nam | mọi hướng (kể cả rẽ phải) | – |
| THT | đi thẳng Lữ Gia + rẽ trái vào LTK Nam | rẽ phải vào LTK Bắc |

Pha: (1) Lữ Gia + THT · (2) LTK hai chiều. Rẽ trái nhường xe ngược chiều.

**J3 – Thành Thái × Tô Hiến Thành** (TT Bắc–Nam, THT Đông–Tây): đèn điều khiển mọi hướng, riêng THT tây rẽ phải vào TT Nam tự do.
Pha: (1) THT hai chiều · (2) TT hai chiều. Rẽ trái nhường xe ngược chiều.

**J4 – Tô Hiến Thành × Đồng Nai:** nằm ở rìa nên không là agent. Giữ đèn cố định mặc định do netconvert tạo.

**J5 – Bắc Hải × Thành Thái** (Bắc Hải Đông–Tây, Thành Thái nhánh Nam)
| Từ | Đèn điều khiển | Tự do |
|---|---|---|
| Bắc Hải đông | đi thẳng (ô tô) + rẽ trái vào TT (mọi xe) | xe máy đi thẳng (làn phải) |
| Bắc Hải tây | mọi hướng, kể cả rẽ phải vào TT | – |
| Thành Thái | rẽ trái vào Bắc Hải tây (mọi xe) + rẽ phải (ô tô) | xe máy rẽ phải vào Bắc Hải đông (làn phải) |

Pha: (1) Bắc Hải đông · (2) Thành Thái · (3) Bắc Hải tây.
Đèn riêng cho xe máy được mô hình bằng kết nối theo loại xe từ làn phải.

### Đèn
- Pha được suy ra tự động: các nhóm đèn không xung đột được gom vào cùng một pha, theo ma trận xung đột của SUMO.
- **Thời gian là tạm thời:** 30 s xanh + 3 s vàng mỗi pha. Sẽ thay bằng kế hoạch đo thực tế (mục C1).

### Phương tiện và nhu cầu
- Xe máy: dài 1,9 m, rộng 0,7 m, minGap 0,5 m, minGapLat 0,3 m, `latAlignment=arbitrary`, mô hình sublane (`lateral-resolution` 0,4 m), tau 1,0 s.
- Các loại xe khác: thông số mặc định của SUMO.
- Mật độ sinh xe giữ nguyên như cũ (xe/km/giờ, 1×/2×/3×), bỏ người đi bộ.
- **Thêm kịch bản `test-heavy`** (12×, khoảng 8.046 xe/giờ) để kiểm tra sức chịu tải của các thuật toán. Mạng cố ý bị quá tải: đèn cố định cho ATT 455 s, hàng chờ trung bình 373 xe, khoảng 1.600 xe chưa tới đích sau 1 giờ. Đã thử 8× (mạng vẫn xả kịp), 12× và 16× (tắc toàn mạng, mỗi episode khoảng 5 phút).

## 3. Giả định cần team xác nhận

1. **Nhánh Bắc Hải phía đông J5 không có trong dữ liệu OSM.** Đã thêm một đoạn thẳng dài 250 m nối tiếp hướng Bắc Hải.
2. **Hẻm 319A được giữ** dù mục B3 là xóa hẻm, vì nó là một nhánh của J0 trong mô tả.
3. **Cấu trúc pha J0 và J5 tách riêng từng nhóm đèn** vì các luồng có đèn đều xung đột nhau. J1–J3 dùng 2 pha, rẽ trái nhường xe ngược chiều.
4. **Lượng xe giảm** so với map cũ (light/medium/heavy: 672 / 1 343 / 2 012 xe/giờ so với 1 433 / 2 862 / 4 292) vì mạng ngắn hơn mà mật độ tính theo km giữ nguyên. Kịch bản heavy hiện chưa kẹt. Nếu cần kẹt hơn, tăng `DENSITY_SCALE` trong `build_trips.py`.
5. **Va chạm:** chỉ tính khi xe chồng lên nhau thật (`collision.mingap-factor 0`), phù hợp cho xe máy chạy sát nhau.

## 4. Kiểm tra (seed 0, 3600 s)

| Kịch bản | Map cũ, đèn cố định | Map mới, đèn cố định | Map mới, MaxPressure |
|---|---|---|---|
| light | ATT 156 s, 36 teleport | ATT 183 s, 0 teleport | ATT 185 s, 1 teleport |
| medium | ATT 184 s, 73 teleport | ATT 182 s, 1 teleport | ATT 193 s, 4 teleport |
| heavy | ATT 516 s, 288 teleport | ATT 185 s, 2 teleport | ATT 202 s, 4 teleport |
| test-heavy (mới) | – | ATT 455 s, 55 teleport | ATT 403 s, 48 teleport |

ATT của hai map không so sánh trực tiếp được (khác mạng và khác lượng xe). Điểm cần xem là teleport gần như về 0.
