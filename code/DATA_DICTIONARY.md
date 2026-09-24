# 📚 Từ Điển Dữ Liệu & Tài Nguyên Dự Án (Data Dictionary)

Tài liệu này đóng vai trò như một bản đồ tri thức (Knowledge Base) để các Agent AI và Developer hiểu rõ mục đích, cấu trúc và vị trí của các luồng dữ liệu, file index, model và module mã nguồn trong dự án AIC 2026.

> **Cập nhật lần cuối:** 2026-08-16 | Index chính: **Qwen3-VL-Embedding-2B (2048 dim)** + **Faster-Whisper ASR** + **RT-DETR Object**

---

## 1. Dữ Liệu Keyframe & Nền Tảng (Base Data)

* **[Custom Keyframes](file:///l:/Competitions/2026-AIC/Data/batch%201/Custom_Keyframes)**
  * **Mô tả:** Thư mục chứa toàn bộ ~358,639 frames đã cắt và lọc chất lượng (Laplacian blur + duplicate filtering).
  * **Đường dẫn:** `Data/batch 1/Custom_Keyframes/{video_id}/{frame_id:06d}.jpg` (Ví dụ: `L23_V007/003349.jpg`).
  * **Mục đích:** Nguồn ảnh gốc để Backend stream hiển thị trên Web Frontend và Lightbox.

* **[Video Gốc MP4](file:///l:/Competitions/2026-AIC/Data/batch%201/video)**
  * **Mô tả:** Thư mục chứa toàn bộ 873 file video gốc `.mp4` (từ `L21_V001.mp4` đến `L30_V096.mp4`).
  * **Đường dẫn:** `Data/batch 1/video/{video_id}.mp4` (Ví dụ: `L23_V007.mp4`).
  * **Mục đích:** Phục vụ streaming trực tiếp qua HTTP Range tới trình phát HTML5 Native Player trên Web.

* **[Media Info](file:///l:/Competitions/2026-AIC/Data/batch%201/media-info)**
  * **Mô tả:** Chứa thông tin metadata của video gốc (FPS, duration, resolution).

---

## 2. Hệ Thống Tìm Kiếm Đặc Trưng Chính (Qwen3-VL Multimodal Index)

* **[Qwen3-VL Index (NPY)](file:///l:/Competitions/2026-AIC/Data/usearch_qwen_index.npy)**
  * **Model sinh ra:** `Qwen/Qwen3-VL-Embedding-2B` (arXiv 2601.04720).
  * **Shape:** `(358,639 × 2048)` — dtype `float32` (~2.74 GB).
  * **Mô tả:** Ma trận vector đặc trưng của toàn bộ 358,639 keyframes sau khi qua ViT + LLM Transformer + Last Token Pooling (EOS).
  * **Mục đích:** Phục vụ so khớp Cosine Similarity thời gian thực với Text Query (hỗ trợ Tiếng Việt & Tiếng Anh).

* **[Qwen3-VL Mapping (JSON)](file:///l:/Competitions/2026-AIC/Data/usearch_qwen_index_mapping.json)**
  * **Số entries:** 358,639 entries (~8.6 MB).
  * **Format:** Mảng JSON `["L01_V001/000000.jpg", ..., "L30_V096/002335.jpg"]`.
  * **Mục đích:** Ánh xạ từ chỉ số hàng `i` trong ma trận vector NPY sang đường dẫn tương đối của file ảnh.

---

## 3. Hệ Thống Lời Thoại Âm Thanh (Faster-Whisper Large-V3 ASR)

* **[ASR Mapping File](file:///l:/Competitions/2026-AIC/Data/asr_mapping.json)**
  * **Model sinh ra:** `Faster-Whisper Large-v3` (từ `Data/batch 1/asr/*.json` gồm 873 file).
  * **Số đoạn thoại:** 16,609 segments câu thoại chuẩn ngữ nghĩa (1,362,943 từ, liên kết tới 330,695 keyframes, chiếm 92.2% CSDL).
  * **Dung lượng:** 19.64 MB.
  * **Format:** Mảng JSON:
    ```json
    [
      {
        "video_id": "L21_V001",
        "start": 4.43,
        "end": 40.45,
        "text": "Chào mừng quý vị đến với chương trình 60 giây của đài truyền hình thành phố Hồ Chí Minh...",
        "matched_keyframes": [
          "L21_V001/000173.jpg",
          "L21_V001/000230.jpg"
        ],
        "language": "vi"
      }
    ]
    ```
  * **Mục đích:** Cho phép tìm kiếm toàn văn theo lời thoại nhân vật, lọc trùng lặp theo phân đoạn thoại ASR (`seg_id`), và hiển thị transcript chi tiết trong Lightbox & Timeline.

* **[Video FPS Mapping File](file:///l:/Competitions/2026-AIC/Data/video_fps_mapping.json)**
  * **Mô tả:** Chứa FPS thực tế của toàn bộ 873 video (30.0, 29.97, 26.44, 25.0).
  * **Mục đích:** Đảm bảo mốc thời gian $t = \text{frame\_idx} / \text{FPS}$ chuẩn xác 100% khi tua video và khớp nối ASR.

---

## 4. Hệ Thống Phát Hiện Vật Thể (RT-DETR Object Detection)

* **[RT-DETR Multi-hot Index (NPY)](file:///l:/Competitions/2026-AIC/Data/usearch_rtdetr_index.npy)**
  * **Shape:** `(358,639 × 80)` — dtype `float16` (~57 MB).
  * **Mô tả:** Confidence score cao nhất của 80 lớp đối tượng COCO Dataset cho từng frame.
  * **Mục đích:** Bộ lọc Pre-filter lọc cứng hoặc kết hợp tính điểm với Visual score.

* **[RT-DETR Metadata (JSON)](file:///l:/Competitions/2026-AIC/Data/rtdetr_metadata.json)**
  * **Kích thước:** ~420 MB.
  * **Mô tả:** Danh sách Bounding Box `[x, y, w, h]`, confidence và class name cho từng keyframe.
  * **Mục đích:** Vẽ khung Bounding Boxes trực quan trên ảnh trong Lightbox Modal.

---

## 5. Cấu Trúc Mã Nguồn (Source Code Architecture)

```
code/
├── local_retrieval/                ← Core Python Logic (Chạy Local)
│   ├── config.py                   ← Cấu hình tập trung (Paths, Dimensions, Weights, API Key)
│   ├── text_encoder.py             ← Qwen3-VL Text Encoder (Instruct prefix chuẩn)
│   ├── asr_engine.py               ← ASR Speech Search & Segment Mapping
│   ├── search_engine.py            ← Search Engine Đa Phương Thức 4 Kênh + Smart Dedup
│   ├── reranker_engine.py          ← Two-Stage Video-Level N-Search Reranker
│   └── gemini_parser.py            ← Bộ phân tích đề bài thi tự động bằng Gemini AI
│
├── web/                            ← Giao Diện Người Dùng & Backend API
│   ├── backend/
│   │   └── main.py                 ← FastAPI Backend Server (REST API endpoints)
│   └── frontend/
│       ├── index.html              ← Giao diện Web 6 Module Cards
│       ├── style.css               ← Flat brutalist / Dark modern styling
│       └── app.js                  ← Client-side interactions, Live Preview, Lightbox
│
└── slurm/scripts/                  ← Slurm Compute Cluster Pipelines
    ├── python/                     ← Trích xuất Qwen3-VL, Faster-Whisper, RT-DETR
    └── bash/                       ← Job submission scripts (Part 1/2/3, Forward/Reverse)
```

---

## 6. Bảng Tra Cứu Trọng Số & Ngưỡng Mặc Định

| Tham số | Giá trị mặc định | Giải thích |
|---|---|---|
| `weight_visual` | `0.50` | Trọng số so khớp ngữ nghĩa hình ảnh Qwen3-VL |
| `weight_ocr` | `0.20` | Trọng số văn bản chữ xuất hiện trên video |
| `weight_asr` | `0.20` | Trọng số lời thoại nhân vật |
| `weight_object` | `0.10` | Trọng số đối tượng vật thể RT-DETR |
| `multi_channel_bonus` | `1.25` | Hệ số nhân thưởng khi frame thỏa mãn $\ge 2$ kênh |
| `visual_sim_thresh` | `0.90` | Ngưỡng Cosine Similarity để Smart Dedup gộp frame trùng |
| `top_k` ($K_1$) | `100` | Số lượng candidate frames quét ở vòng 1 |
| `top_v` ($N$) | `5` | Số lượng video có điểm cao nhất được chọn để Rerank |
| `k_per_video` ($M$) | `40` | Số lượng frames tối đa rút ra từ mỗi video trong Top $N$ |
