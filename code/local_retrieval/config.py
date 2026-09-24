import os

# Đường dẫn gốc của dự án (tính từ file này lùi ra 2 cấp: code/local_retrieval -> code -> project_root)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

# Tự động nạp cấu hình từ .env nếu tồn tại
env_path = os.path.join(PROJECT_ROOT, ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k and v and k not in os.environ:
                        os.environ[k] = v
    except Exception:
        pass

# --- LANCEDB VECTOR DATABASE ---
LANCEDB_PATH = os.path.join(DATA_DIR, "aic_lancedb")
USE_LANCEDB = True

# --- INDEX CHÍNH (Qwen3-VL-Embedding-2B) ---
# Ma trận embedding (358,639 × 2048), float32, 2.74 GB
QWEN_INDEX_PATH = os.path.join(DATA_DIR, "usearch_qwen_index.npy")
# Mapping: list string "VIDEO_ID/FRAME_ID.jpg", 358,639 entries
QWEN_MAPPING_PATH = os.path.join(DATA_DIR, "usearch_qwen_index_mapping.json")

# --- RT-DETR OBJECT DETECTION DATA ---
# Ma trận xác suất 80 lớp COCO (358,639 × 80), float16, 57.38 MB
RTDETR_INDEX_PATH = os.path.join(DATA_DIR, "usearch_rtdetr_index.npy")
# Metadata Bounding Boxes tọa độ pixel của từng keyframe (420.79 MB)
RTDETR_METADATA_PATH = os.path.join(DATA_DIR, "rtdetr_metadata.json")

# 80 lớp đối tượng chuẩn COCO
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Thư mục chứa ảnh keyframe (tự động phát hiện)
if os.path.exists(os.path.join(DATA_DIR, "Custom_Keyframes")):
    KF_DIR = os.path.join(DATA_DIR, "Custom_Keyframes")
elif os.path.exists(os.path.join(DATA_DIR, "batch 1", "Custom_Keyframes")):
    KF_DIR = os.path.join(DATA_DIR, "batch 1", "Custom_Keyframes")
else:
    KF_DIR = os.path.join(DATA_DIR, "Custom_Keyframes")

# Thư mục chứa video gốc .mp4 (tự động phát hiện)
if os.path.exists(os.path.join(DATA_DIR, "video")):
    VIDEO_DIR = os.path.join(DATA_DIR, "video")
elif os.path.exists(os.path.join(DATA_DIR, "batch 1", "video")):
    VIDEO_DIR = os.path.join(DATA_DIR, "batch 1", "video")
else:
    VIDEO_DIR = os.path.join(DATA_DIR, "video")

# --- MODEL ---
TEXT_MODEL = "Qwen/Qwen3-VL-Embedding-2B"
QWEN_RERANKER_MODEL = "Qwen/Qwen3-VL-Reranker-2B"
EMBEDDING_DIM = 2048

# --- RETRIEVAL PARAMS ---
TOP_K = 100
DEFAULT_OBJECT_THRESHOLD = 0.3

# --- ASR & VIDEO METADATA ---
ASR_MAPPING_PATH = os.path.join(DATA_DIR, "asr_mapping.json")
FPS_MAPPING_PATH = os.path.join(DATA_DIR, "video_fps_mapping.json")

# --- VIDEO SHOTS & MAPPING (Qwen3-VL Video 2048D) ---
VIDEO_SHOT_TABLE = "video_shots"
KF_SHOT_MAP_PATH = os.path.join(DATA_DIR, "kf_shot_map.json")
SHOT_METADATA_MAP_PATH = os.path.join(DATA_DIR, "shot_metadata_map.json")

# Thư mục chứa video phân cảnh đã cắt (.mp4 con)
if os.path.exists(os.path.join(DATA_DIR, "batch 1", "output-video-encode", "cut_videos")):
    CUT_VIDEOS_BASE = os.path.join(DATA_DIR, "batch 1", "output-video-encode", "cut_videos")
elif os.path.exists(os.path.join(DATA_DIR, "cut_videos")):
    CUT_VIDEOS_BASE = os.path.join(DATA_DIR, "cut_videos")
else:
    CUT_VIDEOS_BASE = os.path.join(DATA_DIR, "batch 1", "output-video-encode", "cut_videos")

# --- GEMINI API KEY (MẶC ĐỊNH CHO AIC 2026) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

