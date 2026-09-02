# 🏆 AIC 2026 - HỆ THỐNG TRUY VẤN VIDEO ĐA PHƯƠNG THỨC (MULTIMODAL RETRIEVAL)

> **Tài liệu hướng dẫn triển khai nội bộ dành cho Đồng Đội & AI Assistant (Antigravity / Cursor / Claude Code)**.  
> *Hệ thống phục vụ cuộc thi Ho Chi Minh City AI Challenge (AIC 2026) - Hạng mục Known-Item Search (KIS) & Visual Question Answering (QA).*

---

## 📌 MỤC LỤC
1. [Tổng Quan Kiến Trúc](#-tổng-quan-kiến-trúc)
2. [Cấu Trúc Dữ Liệu `Data/` (Tài Nguyên Cần Nhận Riêng)](#-cấu-trúc-dữ-liệu-data-tài-nguyên-cần-nhận-riêng)
3. [🤖 Hướng Dẫn Dành Cho AI Agent (Triển Khai Tự Động)](#-hướng-dẫn-dành-cho-ai-agent-triển-khai-tự-động)
4. [🛠️ Hướng Dẫn Triển Khai Thủ Công (Từng Bước)](#️-hướng-dẫn-triển-khai-thủ-công-từng-bước)
5. [🖥️ Tính Năng Trên Giao Diện Web](#️-tính-năng-trên-giao-diện-web)
6. [❓ Xử Lý Sự Cố Thường Gặp (Troubleshooting)](#-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 🏛️ TỔNG QUAN KIẾN TRÚC

Hệ thống kết hợp mô hình AI thị giác - ngôn ngữ tiên tiến nhất cùng cơ sở dữ liệu vector tốc độ cao:
* **Visual Representation**: `Qwen/Qwen3-VL-Embedding-2B` (Vector không gian siêu cầu 2048D).
* **Object Detection & Spatial Filtering**: `RT-DETR` (80 lớp COCO với tọa độ hộp Bounding Box chi tiết).
* **Audio Transcript (Speech)**: `Whisper ASR` (Nhận diện giọng nói tiếng Việt/Anh).
* **Vector Engine**: `LanceDB` (Apache Arrow Columnar Format, tìm kiếm dưới $2\text{ms}$ trên 358,639 keyframes).
* **Full-Stack Web App**: Backend `FastAPI` (Python) + Frontend `Vanilla HTML/CSS/JS` tối ưu trải nghiệm thời gian thực.

---

## 📂 CẤU TRÚC DỮ LIỆU `Data/` (TÀI NGUYÊN CẦN NHẬN RIÊNG)

Để đảm bảo bảo mật và tránh làm nặng kho mã nguồn GitHub, toàn bộ dữ liệu nặng và cơ sở dữ liệu **KHÔNG ĐƯA LÊN GITHUB**.  
👉 **Đồng đội vui lòng liên hệ Trưởng nhóm để nhận file nén dữ liệu và giải nén vào thư mục `Data/` theo cấu trúc chuẩn sau:**

```
2026-AIC/
├── Data/                                    ← [THƯ MỤC CHỨA DỮ LIỆU]
│   ├── aic_lancedb/                         ← [QUAN TRỌNG NHẤT] Thư mục cơ sở dữ liệu Vector LanceDB (~4.38 GB)
│   │   ├── _versions/
│   │   └── keyframes.lance/                 ← Bảng dữ liệu chứa 358k vectors 2048D & nhãn COCO
│   │
│   ├── Custom_Keyframes/                    ← Thư mục chứa ảnh Keyframes để hiển thị preview trên Web
│   │   ├── L21_V001/                        ← Thư mục video
│   │   │   ├── 000001.jpg                   ← Ảnh trích xuất từ video
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── asr_mapping.json                     ← [Tùy chọn] Dữ liệu phụ đề / giọng nói bóc tách từ Whisper
│   └── video_fps_mapping.json               ← [Tùy chọn] Bảng tra cứu tốc độ khung hình gốc của từng video
│
├── code/
│   ├── local_retrieval/                     ← Engine tìm kiếm lõi (LanceDB, Qwen3-VL, Gemini, RT-DETR)
│   └── web/
│       ├── backend/                         ← Server FastAPI (main.py)
│       └── frontend/                        ← Giao diện Web Client (index.html, app.js, style.css)
│
├── .env                                     ← File biến môi trường chứa API Key (Tạo từ .env.example)
├── verify_setup.py                          ← Script chẩn đoán kiểm tra môi trường 1-click
├── run_app.bat                              ← Script khởi động 1-click cho Windows
├── run_app.sh                               ← Script khởi động 1-click cho Linux / macOS
└── requirements.txt                         ← Danh sách các thư viện Python cần cài đặt
```

---

## 🤖 HƯỚNG DẪN DÀNH CHO AI AGENT (TRIỂN KHAI TỰ ĐỘNG)

> **Gợi ý dành cho đồng đội dùng AI Coding Assistant (Google Antigravity, Cursor, Claude Code, GitHub Copilot Workspace):**  
> Hãy copy toàn bộ đoạn lệnh dưới đây và gửi thẳng vào khung chat của AI Agent:

```markdown
Bạn là trợ lý lập trình phụ trách triển khai dự án AIC 2026. Hãy đọc kỹ file README_VN.md và thực hiện các bước sau:
1. Kiểm tra môi trường Python (yêu cầu >= 3.10) và cài đặt dependencies từ `requirements.txt` vào một Virtualenv (venv).
2. Chạy script `python verify_setup.py` để kiểm tra sức khỏe hệ thống.
3. Nếu phát hiện thiếu file `.env`, hãy tự động tạo file `.env` từ `.env.example`.
4. Nếu kiểm tra đạt chuẩn, hãy khởi động FastAPI Server bằng lệnh `python code/web/backend/main.py` và hướng dẫn tôi mở trình duyệt tại `http://127.0.0.1:8000`.
```

---

## 🛠️ HƯỚNG DẪN TRIỂN KHAI THỦ CÔNG (TỪNG BƯỚC)

### Bước 1: Clone mã nguồn về máy
```bash
git clone <URL_REPO_GITHUB>
cd 2026-AIC
```

### Bước 2: Tạo môi trường ảo Python & Cài đặt thư viện
Khuyến nghị sử dụng Python `3.10`, `3.11` hoặc `3.12`:

* **Trên Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

* **Trên Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

### Bước 3: Đặt dữ liệu vào thư mục `Data/`
1. Nhận thư mục **`aic_lancedb`** từ Trưởng nhóm và đặt vào: `Data/aic_lancedb/`.
2. Nhận thư mục ảnh **`Custom_Keyframes`** và đặt vào: `Data/Custom_Keyframes/`.

### Bước 4: Cấu hình Khóa Gemini API (Tùy chọn)
1. Copy file `.env.example` thành file `.env`:
   * Windows: `copy .env.example .env`
   * Linux: `cp .env.example .env`
2. Mở file `.env` và điền khóa [Google AI Studio API](https://aistudio.google.com/):
   ```ini
   GEMINI_API_KEY=AIzaSy...
   ```

### Bước 5: Kiểm tra sức khỏe môi trường (Health Check)
Chạy script kiểm tra tự động:
```bash
python verify_setup.py
```
Nếu màn hình báo xanh `[ĐẠT - OK]` ở các mục chính, hệ thống của bạn đã sẵn sàng $100\%$!

### Bước 6: Khởi động hệ thống (1-Click)
* **Windows**: Nháy đúp chuột vào file **`run_app.bat`** (hoặc gõ `.\run_app.bat` trong Terminal).
* **Linux / macOS**: Chạy lệnh **`bash run_app.sh`**.

Trình duyệt sẽ tự động mở trang web tìm kiếm tại: **`http://127.0.0.1:8000`**.

---

## 🖥️ TÍNH NĂNG TRÊN GIAO DIỆN WEB

* 🔍 **Multi-Modal Text Search**: Tìm kiếm bằng câu văn tiếng Việt / tiếng Anh ("hai người nấu ăn trong bếp", "xe cứu thương trên đường cao tốc").
* 🖼️ **Multi-View Image Search**: Cho phép kéo thả hoặc nhấn `Ctrl + V` nạp nhiều góc chụp của cùng một vật thể để truy vấn chính xác.
* 📦 **Lọc đối tượng COCO & Bounding Box**: Vẽ trực tiếp hộp vùng chọn trên Lightbox hoặc lọc theo 80 lớp đối tượng (xe cộ, con người, bàn ghế, v.v.).
* 🚫 **Loại trừ nhanh Video (`🚫 Loại video này`)**: Nút bấm trực tiếp trên thẻ kết quả để làm mờ ngay các video không mong muốn và đồng bộ vào bộ lọc.
* ⏱️ **Trục thời gian Video Timeline**: Xem toàn bộ phân cảnh xung quanh khung hình tìm thấy để xác định thời điểm nộp bài KIS.
* 🏷️ **Thứ Hạng Thật (`#Rank`)**: Giữ nguyên vị trí xếp hạng gốc từ thuật toán khi áp dụng bộ lọc client-side.

---

## ❓ XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

#### 1. Lỗi `DLL load failed while importing _lancedb`:
* **Nguyên nhân**: Trên một số máy Windows có chính sách bảo mật (Application Control) hoặc thiếu thư viện C++ Redistributable.
* **Khắc phục**: Cài đặt gói [Microsoft Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) rồi khởi động lại máy.

#### 2. Máy không có card đồ họa NVIDIA (No GPU / CUDA):
* Hệ thống được lập trình với cơ chế **Zero-Crash Fallback**: Nếu không có GPU, mã nguồn sẽ tự động chuyển sang tính toán trên **CPU** mà không gây crash ứng dụng!

#### 3. Tìm kiếm ra kết quả nhưng ảnh preview bị vỡ hình:
* **Nguyên nhân**: Đường dẫn thư mục ảnh trong `Data/Custom_Keyframes` chưa đúng.
* **Khắc phục**: Kiểm tra lại xem cấu trúc thư mục có dạng `Data/Custom_Keyframes/[VIDEO_ID]/[FRAME_ID].jpg` chưa.
