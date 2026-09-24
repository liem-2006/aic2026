"""
VideoSearchEngine — Load Qwen3-VL index và phục vụ truy vấn Đa Phương Thức nâng cao (Visual + OCR + ASR + RT-DETR Objects).

Hỗ trợ:
1. Qwen3-VL-Embedding (2.74 GB)
2. Whisper ASR Mapping (84k segments)
3. RT-DETR Object Detection Index (57 MB, 80 COCO classes)
4. RT-DETR Bounding Box Metadata (420 MB)
5. Bộ Lọc Trùng (Deduplication) Đa Phương Thức: "smart", "visual_only", "asr_only", "unique", "none"
"""
import os
import re
import json
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from usearch.index import Index

try:
    from . import config
    from .text_encoder import get_text_encoder
    from .asr_engine import get_asr_engine
except (ImportError, ValueError):
    import config
    from text_encoder import get_text_encoder
    from asr_engine import get_asr_engine


class VideoSearchEngine:
    def __init__(self):
        print("[SearchEngine] Đang khởi tạo...")
        t0 = time.time()

        # 1. Load mapping JSON
        if not os.path.exists(config.QWEN_MAPPING_PATH):
            raise FileNotFoundError(f"Không tìm thấy mapping: {config.QWEN_MAPPING_PATH}")

        with open(config.QWEN_MAPPING_PATH, "r", encoding="utf-8") as f:
            self.mapping: List[str] = json.load(f)

        # Tách mảng video_id để lọc nhanh bằng numpy
        self.video_ids = np.array([p.split("/")[0] for p in self.mapping])
        self.path_to_idx = {p: i for i, p in enumerate(self.mapping)}

        # 1.1 Nạp danh mục phân loại video (Category & Sub-category)
        self.video_categories: Dict[str, Dict[str, str]] = {}
        cat_path = os.path.join(config.DATA_DIR, "video_categories.json")
        if os.path.exists(cat_path):
            with open(cat_path, "r", encoding="utf-8") as f:
                self.video_categories = json.load(f)
            self.categories = np.array([self.video_categories.get(vid, {}).get("category", "") for vid in self.video_ids])
            self.sub_categories = np.array([self.video_categories.get(vid, {}).get("sub_category", "") for vid in self.video_ids])
        else:
            self.categories = np.array(["" for _ in self.video_ids])
            self.sub_categories = np.array(["" for _ in self.video_ids])

        print(f"[SearchEngine] Mapping nạp xong: {len(self.mapping):,} entries | Phân loại: {len(self.video_categories):,} videos")

        # 2. Khởi tạo Engine Tìm Kiếm Vector (LanceDB Disk-Native hoặc USearch In-Memory)
        self.use_lancedb = getattr(config, "USE_LANCEDB", False) and os.path.exists(getattr(config, "LANCEDB_PATH", ""))
        self.lancedb_client = None
        self.lancedb_table = None
        self.beit3_table = None
        self.beit3_encoder = None
        self.features = None
        self.feature_norms = None
        self.index = None

        if self.use_lancedb:
            import lancedb
            print(f"[SearchEngine] Đang nạp LanceDB Vector Database từ: {config.LANCEDB_PATH}...")
            self.lancedb_client = lancedb.connect(config.LANCEDB_PATH)
            self.lancedb_table = self.lancedb_client.open_table("keyframes")
            
            existing_tables = []
            try:
                existing_tables = self.lancedb_client.table_names()
            except Exception:
                try:
                    existing_tables = self.lancedb_client.list_tables().tables
                except Exception:
                    existing_tables = []

            if "keyframes_beit3" in existing_tables:
                self.beit3_table = self.lancedb_client.open_table("keyframes_beit3")
            
            # 2.1 Bảng video_shots (Qwen3-VL Video 2048D)
            self.shot_table = None
            shot_tbl_name = getattr(config, "VIDEO_SHOT_TABLE", "video_shots")
            if shot_tbl_name in existing_tables:
                self.shot_table = self.lancedb_client.open_table(shot_tbl_name)
                print(f"[SearchEngine] ✅ Nạp bảng '{shot_tbl_name}': {len(self.shot_table):,} items.")

            self.d = config.EMBEDDING_DIM
            print(f"[SearchEngine] ✅ LanceDB Engine nạp xong | Bảng 'keyframes': {len(self.lancedb_table):,} items.")

            # Nạp mmap ma trận embedding Qwen (0 MB RAM) để Cross-Scoring siêu tốc
            self.qwen_mmap = None
            if os.path.exists(config.QWEN_INDEX_PATH):
                self.qwen_mmap = np.load(config.QWEN_INDEX_PATH, mmap_mode="r")
        else:
            self.shot_table = None
            if not os.path.exists(config.QWEN_INDEX_PATH):
                raise FileNotFoundError(f"Không tìm thấy index: {config.QWEN_INDEX_PATH}")

            print(f"[SearchEngine] Đang nạp ma trận embedding (~2.74 GB, vui lòng chờ)...")
            self.features = np.load(config.QWEN_INDEX_PATH)  # shape (358639, 2048)

            self.d = self.features.shape[1]
            if self.d != config.EMBEDDING_DIM:
                raise ValueError(f"Chiều vector sai: {self.d} != {config.EMBEDDING_DIM}. Vui lòng kiểm tra lại file NPY.")

            # Tính trước chuẩn Norm L2 cho từng vector để tính Cosine Similarity siêu tốc
            self.feature_norms = np.linalg.norm(self.features, axis=1)
            self.feature_norms[self.feature_norms == 0] = 1.0

            # Khởi tạo USearch Index (Cosine Metric)
            self.index = Index(ndim=self.d, metric="cos", dtype="f32")
            self.index.add(np.arange(len(self.features)), self.features)

        # 3. Nạp ASR Engine
        self.asr_engine = get_asr_engine()
        self.asr_engine.load()

        # 4. Nạp RT-DETR Index (57 MB)
        self.rtdetr_index = None
        if os.path.exists(config.RTDETR_INDEX_PATH):
            print(f"[SearchEngine] Đang nạp RT-DETR Index (57 MB)...")
            self.rtdetr_index = np.load(config.RTDETR_INDEX_PATH)  # shape (358639, 80), float16
            print(f"[SearchEngine] ✅ RT-DETR Index (80 COCO classes) nạp xong.")

        # 5. Metadata Bounding Boxes (Nạp Lazy khi cần)
        self.rtdetr_metadata: Optional[Dict[str, Any]] = None

        # 6. Video FPS Mapping (Chính xác 100% cho từng video)
        self.fps_mapping: Dict[str, float] = {}
        if os.path.exists(config.FPS_MAPPING_PATH):
            with open(config.FPS_MAPPING_PATH, "r", encoding="utf-8") as f:
                self.fps_mapping = json.load(f)
            print(f"[SearchEngine] ✅ Đã nạp FPS Mapping cho {len(self.fps_mapping):,} videos.")

        # 7. Video Shot Mappings (Tra cứu O(1) KF <-> Shot)
        self.kf_shot_map: Dict[str, str] = {}
        kf_map_path = getattr(config, "KF_SHOT_MAP_PATH", "")
        if os.path.exists(kf_map_path):
            try:
                with open(kf_map_path, "r", encoding="utf-8") as f:
                    self.kf_shot_map = json.load(f)
                print(f"[SearchEngine] ✅ Đã nạp KF->Shot Mapping: {len(self.kf_shot_map):,} keys.")
            except Exception as e:
                print(f"[SearchEngine] ⚠️ Lỗi đọc kf_shot_map: {e}")

        self.shot_meta_map: Dict[str, Dict[str, Any]] = {}
        shot_meta_path = getattr(config, "SHOT_METADATA_MAP_PATH", "")
        if os.path.exists(shot_meta_path):
            try:
                with open(shot_meta_path, "r", encoding="utf-8") as f:
                    self.shot_meta_map = json.load(f)
                print(f"[SearchEngine] ✅ Đã nạp Shot Metadata: {len(self.shot_meta_map):,} shots.")
            except Exception as e:
                print(f"[SearchEngine] ⚠️ Lỗi đọc shot_meta_map: {e}")

        print(f"[SearchEngine] ✅ Khởi tạo xong trong {time.time() - t0:.2f}s | {len(self.mapping):,} frames sẵn sàng")

    def get_video_fps(self, video_id: str) -> float:
        """Lấy FPS thực tế của video, mặc định 25.0 nếu không tìm thấy."""
        clean_vid = video_id.strip()
        return float(self.fps_mapping.get(clean_vid, 25.0))

    def _ensure_metadata_loaded(self):
        """Nạp RT-DETR Bounding Box Metadata vào bộ nhớ nếu chưa nạp."""
        if self.rtdetr_metadata is None:
            if os.path.exists(config.RTDETR_METADATA_PATH):
                t_load = time.time()
                print("[SearchEngine] Đang nạp RT-DETR Metadata Bounding Boxes (420 MB)...")
                with open(config.RTDETR_METADATA_PATH, "r", encoding="utf-8") as f:
                    self.rtdetr_metadata = json.load(f)
                print(f"[SearchEngine] ✅ Nạp xong {len(self.rtdetr_metadata):,} entries Bounding Boxes trong {time.time() - t_load:.2f}s.")
            else:
                self.rtdetr_metadata = {}

    def get_boxes_for_keyframe(self, video_id: str, frame_id: str, raw_boxes_json: Optional[str] = None) -> List[Dict[str, Any]]:
        """Lấy danh sách bounding boxes của 1 keyframe cụ thể."""
        if raw_boxes_json:
            try:
                raw_boxes = json.loads(raw_boxes_json)
            except Exception:
                raw_boxes = []
        else:
            self._ensure_metadata_loaded()
            if not self.rtdetr_metadata:
                return []
            
            clean_vid = str(video_id).strip()
            clean_fid = f"{int(frame_id):06d}" if str(frame_id).strip().isdigit() else str(frame_id).strip().replace(".jpg", "")
            
            # Format trong rtdetr_metadata.json là "{video_id}_{frame_id}" (Ví dụ: "L21_V001_000410")
            raw_boxes = (
                self.rtdetr_metadata.get(f"{clean_vid}_{clean_fid}")
                or self.rtdetr_metadata.get(f"{clean_vid}/{clean_fid}.jpg")
                or self.rtdetr_metadata.get(f"{clean_vid}_{clean_fid}.jpg")
                or self.rtdetr_metadata.get(f"{clean_vid}/{clean_fid}")
                or []
            )

        formatted = []
        for b in raw_boxes:
            cid = b.get("class_id", -1)
            cname = config.COCO_CLASSES[cid] if 0 <= cid < len(config.COCO_CLASSES) else f"cls_{cid}"
            formatted.append({
                "class_id": cid,
                "class_name": cname,
                "confidence": round(float(b.get("confidence", 0.0)), 3),
                "box": [round(float(coord), 1) for coord in b.get("box", [])]
            })
        return formatted

    def get_keyframes_for_video(self, video_id: str) -> List[Dict[str, Any]]:
        """
        Lấy toàn bộ keyframes của 1 video theo đúng thứ tự thời gian tăng dần.
        Phục vụ tính năng Timeline Keyframe Strip (Chiến thuật 5).
        """
        clean_vid = video_id.strip()
        matching_paths = [p for p in self.mapping if p.startswith(f"{clean_vid}/")]
        if not matching_paths:
            # Thử tìm substring nếu user gõ v_id không chuẩn
            matching_paths = [p for p in self.mapping if clean_vid in p.split("/")[0]]

        # Sắp xếp theo frame_id dạng số nguyên
        def get_frame_num(p: str) -> int:
            try:
                return int(p.split("/")[1].replace(".jpg", ""))
            except (ValueError, IndexError):
                return 0

        matching_paths.sort(key=get_frame_num)

        timeline_frames = []
        for p in matching_paths:
            parts = p.split("/")
            v_id = parts[0]
            f_id = parts[1].replace(".jpg", "")
            fps = self.get_video_fps(v_id)
            try:
                sec = int(f_id) / fps
            except ValueError:
                sec = 0.0

            # Lấy thông tin ASR transcript mở rộng nếu có
            sp_exp = self.asr_engine.get_speech_for_keyframe_expanded(p, window_n=1)
            transcript = sp_exp.get("text", "")
            asr_prev = sp_exp.get("prev_text", "")
            asr_next = sp_exp.get("next_text", "")
            asr_all = sp_exp.get("all_context", "")
            boxes = self.get_boxes_for_keyframe(v_id, f_id)

            timeline_frames.append({
                "video_id": v_id,
                "frame_id": f_id,
                "keyframe": p,
                "timestamp_sec": round(sec, 2),
                "asr_transcript": transcript,
                "asr_prev": asr_prev,
                "asr_next": asr_next,
                "asr_all": asr_all,
                "boxes": boxes,
                "image_url": f"http://localhost:8000/images/{v_id}/{f_id}.jpg"
            })

        return timeline_frames

    def _search_shots(
        self,
        query_vec: np.ndarray,
        top_k: int = 200,
        effective_include: Optional[List[str]] = None,
        effective_exclude: Optional[List[str]] = None,
        category: str = "",
        sub_category: str = ""
    ) -> List[Dict[str, Any]]:
        """Truy vấn trực tiếp trên bảng video_shots (LanceDB) với bộ lọc pushdown."""
        if self.shot_table is None or query_vec is None:
            return []

        try:
            vec_list = query_vec.tolist() if isinstance(query_vec, np.ndarray) else query_vec
            q_builder = self.shot_table.search(vec_list)
            where_clauses = []

            if effective_include:
                exact_inc = [v for v in effective_include if "_" in v]
                prefix_inc = [v for v in effective_include if "_" not in v]
                inc_parts = []
                if exact_inc:
                    v_str = ", ".join([f"'{v}'" for v in exact_inc])
                    inc_parts.append(f"video_id IN ({v_str})")
                for pfx in prefix_inc:
                    inc_parts.append(f"video_id LIKE '{pfx}%'")
                if inc_parts:
                    where_clauses.append(" OR ".join(inc_parts))

            if effective_exclude:
                exact_exc = [v for v in effective_exclude if "_" in v]
                prefix_exc = [v for v in effective_exclude if "_" not in v]
                for v in exact_exc:
                    where_clauses.append(f"video_id != '{v}'")
                for pfx in prefix_exc:
                    where_clauses.append(f"NOT (video_id LIKE '{pfx}%')")

            if category and category.strip() and category != "all":
                where_clauses.append(f"category = '{category.strip()}'")
            if sub_category and sub_category.strip() and sub_category != "all":
                where_clauses.append(f"sub_category = '{sub_category.strip()}'")

            if where_clauses:
                full_where = " AND ".join([f"({c})" for c in where_clauses])
                q_builder = q_builder.where(full_where)

            df = q_builder.metric("cosine").limit(top_k).to_pandas()
            return df.to_dict("records")
        except Exception as e:
            print(f"[SearchEngine] ⚠️ Lỗi tìm kiếm video_shots: {e}")
            return []

    def search(
        self,
        query_text: str = "",
        ocr_query: str = "",
        speech_query: str = "",
        enable_asr: bool = True,
        object_class_id: Optional[int] = None,
        object_conf_thresh: float = 0.0,
        dedup_mode: str = "smart",         # "smart" | "visual_only" | "asr_only" | "unique" | "none"
        visual_sim_thresh: float = 0.90,   # Ngưỡng tương đồng vector trong cùng video (0.80 -> 0.98)
        enable_cross_video_dedup: bool = True,  # Bật/tắt khử trùng thị giác liên video (chống QC/Intro)
        cross_video_sim_thresh: float = 0.90,   # Ngưỡng tương đồng liên video
        video_id: str = "",
        include_videos: Optional[List[str]] = None,
        exclude_videos: Optional[List[str]] = None,
        category: str = "",
        sub_category: str = "",
        weight_visual: float = 1.0,
        weight_ocr: float = 1.0,
        weight_asr: float = 1.0,
        weight_object: float = 0.0,
        top_k: int = config.TOP_K,
        use_rrf: bool = True,
        rrf_k: int = 60,
        asr_window_n: int = 1,
        query_vec: Optional[np.ndarray] = None,
        query_image: Optional[Any] = None,
        image_weight: float = 0.5,
        rrf_use_visual: bool = True,       # Tín hiệu KF Visual tham gia RRF
        rrf_use_asr: bool = True,          # Tín hiệu ASR tham gia RRF
        rrf_use_shot: bool = True,         # Tín hiệu Video Shot tham gia RRF
        return_shots: bool = False,        # Trả về cả {results, shot_results}
        **kwargs
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Truy vấn đa phương thức kết hợp:
        Qwen3-VL (Visual + OCR + Image + Composed Image/Text) + Whisper (ASR) + RT-DETR (Object Detection).
        Hỗ trợ chế độ RRF và các bộ lọc Pushdown Nâng Cao (Tag Multi-Video Include/Exclude, Category & Sub-category).
        """
        t0 = time.time()
        has_visual = bool(query_text and query_text.strip())
        has_ocr = bool(ocr_query and ocr_query.strip())
        has_image = query_image is not None
        
        effective_speech = speech_query.strip() if (speech_query and speech_query.strip()) else (query_text.strip() if enable_asr else "")
        has_speech = bool(enable_asr and effective_speech)
        has_object = bool(object_class_id is not None and object_class_id >= 0 and self.rtdetr_index is not None)

        if not has_visual and not has_ocr and not has_speech and not has_object and not (video_id and video_id.strip()) and not include_videos and not category and query_vec is None and not has_image:
            return []

        # Chuẩn hóa danh sách video include và exclude (hỗ trợ nhập chuỗi nhiều ID/Prefix cách nhau bởi phẩy/khoảng trắng)
        effective_include = []
        if video_id and str(video_id).strip():
            for item in re.split(r"[,;\s]+", str(video_id).strip()):
                if item and item.upper() not in effective_include:
                    effective_include.append(item.upper())
        if include_videos:
            for v in include_videos:
                if isinstance(v, str):
                    for item in re.split(r"[,;\s]+", v.strip()):
                        if item and item.upper() not in effective_include:
                            effective_include.append(item.upper())

        effective_exclude = []
        if exclude_videos:
            for v in exclude_videos:
                if isinstance(v, str):
                    for item in re.split(r"[,;\s]+", v.strip()):
                        if item and item.upper() not in effective_exclude:
                            effective_exclude.append(item.upper())

        # --- BƯỚC 1: LỌC THEO VIDEO ID & BỘ LỌC BAN ĐẦU ---
        valid_mask = np.ones(len(self.mapping), dtype=bool)

        if effective_include:
            inc_mask = np.zeros(len(self.mapping), dtype=bool)
            for item in effective_include:
                if "_" in item:
                    inc_mask |= (self.video_ids == item)
                else:
                    inc_mask |= np.char.startswith(self.video_ids, item)
            valid_mask &= inc_mask

        if effective_exclude:
            exc_mask = np.zeros(len(self.mapping), dtype=bool)
            for item in effective_exclude:
                if "_" in item:
                    exc_mask |= (self.video_ids == item)
                else:
                    exc_mask |= np.char.startswith(self.video_ids, item)
            valid_mask &= ~exc_mask

        # Lọc theo Danh mục (Category) & Môn học (Sub-category)
        if category and category.strip() and category != "all":
            valid_mask &= (self.categories == category.strip())
        if sub_category and sub_category.strip() and sub_category != "all":
            valid_mask &= (self.sub_categories == sub_category.strip())

        # Lọc theo Object Confidence Threshold
        if has_object and object_conf_thresh > 0:
            obj_scores_all = self.rtdetr_index[:, object_class_id]
            valid_mask = valid_mask & (obj_scores_all >= object_conf_thresh)

        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return []

        # Bảng gom điểm theo keyframe path: path -> dict thông tin
        frame_candidates: Dict[str, Dict[str, Any]] = {}
        shot_cands: List[Dict[str, Any]] = []
        encoder = get_text_encoder()

        # --- BƯỚC 2A: KÊNH VISUAL SEARCH (QWEN3-VL 2048D ĐA HÌNH: CHỮ / ẢNH / KẾT HỢP) ---
        if has_visual or has_ocr or has_image or query_vec is not None:
            if query_vec is None:
                prompt_parts = []
                if has_visual:
                    prompt_parts.append(query_text.strip())
                if has_ocr:
                    prompt_parts.append(f"văn bản chữ trong ảnh: {ocr_query.strip()}")
                combined_text = ", ".join(prompt_parts)

                query_vec = encoder.encode_visual(
                    text=combined_text,
                    image=query_image,
                    image_weight=image_weight
                )

            if query_vec is not None and query_vec.shape[0] == self.d:
                # Tìm kiếm Video Shots song song
                if (rrf_use_shot or return_shots) and self.shot_table is not None:
                    shot_cands = self._search_shots(
                        query_vec=query_vec,
                        top_k=max(top_k * 5, 200),
                        effective_include=effective_include,
                        effective_exclude=effective_exclude,
                        category=category,
                        sub_category=sub_category
                    )

                search_k = min(len(self.mapping), max(top_k * 30, 2000))

                if self.use_lancedb and self.lancedb_table is not None:
                    q_builder = self.lancedb_table.search(query_vec)
                    where_clauses = []

                    # 1. Multi-video include filter (hỗ trợ cả exact ID và prefix)
                    if effective_include:
                        exact_inc = [v for v in effective_include if "_" in v]
                        prefix_inc = [v for v in effective_include if "_" not in v]
                        inc_parts = []
                        if exact_inc:
                            v_str = ", ".join([f"'{v}'" for v in exact_inc])
                            inc_parts.append(f"video_id IN ({v_str})")
                        for pfx in prefix_inc:
                            inc_parts.append(f"video_id LIKE '{pfx}%'")
                        if inc_parts:
                            where_clauses.append(" OR ".join(inc_parts))

                    # 2. Multi-video exclude filter (hỗ trợ cả exact ID và prefix)
                    if effective_exclude:
                        exact_exc = [v for v in effective_exclude if "_" in v]
                        prefix_exc = [v for v in effective_exclude if "_" not in v]
                        if exact_exc:
                            ex_str = ", ".join([f"'{v}'" for v in exact_exc])
                            where_clauses.append(f"video_id NOT IN ({ex_str})")
                        for pfx in prefix_exc:
                            where_clauses.append(f"video_id NOT LIKE '{pfx}%'")

                    # 3. Category filter
                    if category and category.strip() and category != "all":
                        where_clauses.append(f"category = '{category.strip()}'")

                    # 4. Sub-category filter (Môn học)
                    if sub_category and sub_category.strip() and sub_category != "all":
                        where_clauses.append(f"sub_category = '{sub_category.strip()}'")

                    if where_clauses:
                        sql_where = " AND ".join([f"({c})" for c in where_clauses])
                        q_builder = q_builder.where(sql_where, prefilter=True)
                    
                    hits = q_builder.limit(search_k).to_list()
                    for hit in hits:
                        kf_path = hit["id"]
                        if kf_path in self.path_to_idx and valid_mask[self.path_to_idx[kf_path]]:
                            idx = self.path_to_idx[kf_path]
                            dist = float(hit.get("_distance", 0.0))
                            sim = max(0.0, 1.0 - dist)
                            v_raw = hit.get("vector")
                            v_arr = np.array(v_raw, dtype=np.float32) if v_raw is not None else None
                            v_norm = float(np.linalg.norm(v_arr)) if v_arr is not None else 1.0
                            if v_norm == 0: v_norm = 1.0
                            
                            obj_scs = hit.get("obj_scores", [])
                            obj_score = float(obj_scs[object_class_id]) if (has_object and 0 <= object_class_id < len(obj_scs)) else 0.0

                            frame_candidates[kf_path] = {
                                "path": kf_path,
                                "idx": idx,
                                "score_visual": round(sim, 4),
                                "score_ocr": round(sim, 4) if has_ocr else 0.0,
                                "score_asr": 0.0,
                                "score_obj": round(obj_score, 4),
                                "speech_text": "",
                                "speech_start": 0.0,
                                "speech_end": 0.0,
                                "vector": v_arr,
                                "vec_norm": v_norm,
                                "boxes_json": hit.get("boxes_json")
                            }
                elif self.index is not None:
                    matches = self.index.search(query_vec, search_k)
                    for dist, idx in zip(matches.distances, matches.keys):
                        if valid_mask[idx]:
                            sim = max(0.0, 1.0 - float(dist))
                            kf_path = self.mapping[idx]
                            obj_score = float(self.rtdetr_index[idx, object_class_id]) if (has_object and self.rtdetr_index is not None) else 0.0
                            frame_candidates[kf_path] = {
                                "path": kf_path,
                                "idx": int(idx),
                                "score_visual": round(sim, 4),
                                "score_ocr": round(sim, 4) if has_ocr else 0.0,
                                "score_asr": 0.0,
                                "score_obj": round(obj_score, 4),
                                "speech_text": "",
                                "speech_start": 0.0,
                                "speech_end": 0.0
                            }

        # --- BƯỚC 2B: KÊNH ASR SPEECH SEARCH (HYBRID: QWEN3-VL SEMANTIC + TF-IDF LEXICAL) ---
        if has_speech:
            asr_results_dict: Dict[str, Dict[str, Any]] = {}

            # 1. Semantic Search qua Qwen3-VL Vector Embedding trên LanceDB asr_segments (nếu bảng có sẵn)
            if hasattr(self.asr_engine, "asr_table") and self.asr_engine.asr_table is not None:
                encoder = get_text_encoder()
                asr_vec = encoder.encode(effective_speech)
                sem_results = self.asr_engine.search_semantic(
                    query_vector=asr_vec,
                    video_id=video_id.strip() if video_id else "",
                    include_videos=include_videos,
                    top_k=max(top_k * 15, 800)
                )
                for item in sem_results:
                    asr_results_dict[item["path"]] = item

            # 2. Lexical Search qua TF-IDF N-gram (bắt chính xác từng từ khóa/tên riêng)
            lex_results = self.asr_engine.search_keyframes(
                query_speech=effective_speech,
                video_id=video_id.strip() if video_id else "",
                top_k=max(top_k * 15, 800)
            )
            for item in lex_results:
                kf_p = item["path"]
                if kf_p in asr_results_dict:
                    asr_results_dict[kf_p]["score_asr"] = max(asr_results_dict[kf_p]["score_asr"], item["score_asr"])
                else:
                    asr_results_dict[kf_p] = item

            asr_results = list(asr_results_dict.values())

            for item in asr_results:
                kf_path = item["path"]
                if kf_path in self.path_to_idx and valid_mask[self.path_to_idx[kf_path]]:
                    idx = self.path_to_idx[kf_path]
                    obj_score = float(self.rtdetr_index[idx, object_class_id]) if has_object else 0.0
                    if kf_path in frame_candidates:
                        frame_candidates[kf_path]["score_asr"] = item["score_asr"]
                        frame_candidates[kf_path]["speech_text"] = item["speech_text"]
                        frame_candidates[kf_path]["speech_start"] = item["speech_start"]
                        frame_candidates[kf_path]["speech_end"] = item["speech_end"]
                    else:
                        frame_candidates[kf_path] = {
                            "path": kf_path,
                            "idx": idx,
                            "score_visual": 0.0,
                            "score_ocr": 0.0,
                            "score_asr": item["score_asr"],
                            "score_obj": round(obj_score, 4),
                            "speech_text": item["speech_text"],
                            "speech_start": item["speech_start"],
                            "speech_end": item["speech_end"]
                        }

        # Nếu chỉ tìm theo Object
        if not has_visual and not has_ocr and not has_speech and has_object:
            top_obj_indices = np.argsort(-self.rtdetr_index[:, object_class_id])[:top_k * 5]
            for idx in top_obj_indices:
                if valid_mask[idx]:
                    kf_path = self.mapping[idx]
                    obj_score = float(self.rtdetr_index[idx, object_class_id])
                    if obj_score > 0:
                        frame_candidates[kf_path] = {
                            "path": kf_path,
                            "idx": int(idx),
                            "score_visual": 0.0,
                            "score_ocr": 0.0,
                            "score_asr": 0.0,
                            "score_obj": round(obj_score, 4),
                            "speech_text": "",
                            "speech_start": 0.0,
                            "speech_end": 0.0
                        }

        if not frame_candidates:
            return []

        # --- BƯỚC 2C: CROSS-SCORING THỊ GIÁC CHO CÁC CANDIDATES TỪ ASR/OCR/OBJECT ---
        # Khi một frame được mang vào bởi kênh ASR hoặc Object mà chưa có điểm visual (score_visual == 0.0),
        # ta tính tích vô hướng Cosine Similarity trực tiếp giữa query_vec và vector của frame đó để phục vụ RRF & hiển thị chuẩn xác
        if has_visual and query_vec is not None:
            for kf_p, cand in frame_candidates.items():
                if cand["score_visual"] <= 0.0:
                    cand_idx = cand["idx"]
                    if cand.get("vector") is not None:
                        v = cand["vector"]
                        norm = cand.get("vec_norm", 1.0)
                        sim = float(np.dot(v, query_vec)) / norm
                    elif self.qwen_mmap is not None and 0 <= cand_idx < len(self.qwen_mmap):
                        v = self.qwen_mmap[cand_idx]
                        norm = float(np.linalg.norm(v))
                        if norm == 0: norm = 1.0
                        sim = float(np.dot(v, query_vec)) / norm
                    elif self.features is not None and 0 <= cand_idx < len(self.features):
                        v = self.features[cand_idx]
                        norm = float(self.feature_norms[cand_idx])
                        sim = float(np.dot(v, query_vec)) / norm
                    else:
                        sim = 0.0

                    cand["score_visual"] = round(max(0.0, sim), 4)

        # --- BƯỚC 3: TÍNH ĐIỂM FUSION (RRF k HOẶC WEIGHTED SUM) ---
        candidates_list = list(frame_candidates.values())

        if use_rrf:
            k_val = max(1, int(rrf_k))
            
            # Xếp hạng từng kênh độc lập dựa trên tín hiệu được chọn
            if has_visual and rrf_use_visual:
                vis_cands = sorted([c for c in candidates_list if c["score_visual"] > 0], key=lambda x: x["score_visual"], reverse=True)
                for r_idx, c in enumerate(vis_cands):
                    c["rank_vis"] = r_idx + 1

            if has_ocr:
                ocr_cands = sorted([c for c in candidates_list if c["score_ocr"] > 0], key=lambda x: x["score_ocr"], reverse=True)
                for r_idx, c in enumerate(ocr_cands):
                    c["rank_ocr"] = r_idx + 1

            if has_speech and rrf_use_asr:
                asr_cands = sorted([c for c in candidates_list if c["score_asr"] > 0], key=lambda x: x["score_asr"], reverse=True)
                for r_idx, c in enumerate(asr_cands):
                    c["rank_asr"] = r_idx + 1

            if has_object:
                obj_cands = sorted([c for c in candidates_list if c["score_obj"] > 0], key=lambda x: x["score_obj"], reverse=True)
                for r_idx, c in enumerate(obj_cands):
                    c["rank_obj"] = r_idx + 1

            # Bảng tra thứ hạng Video Shot (nếu bật rrf_use_shot)
            shot_rank_map = {}
            if rrf_use_shot and shot_cands:
                for r_idx, sc in enumerate(shot_cands):
                    shot_rank_map[sc["shot_id"]] = r_idx + 1

            raw_rrf_scores = []
            for cand in candidates_list:
                score = 0.0
                if "rank_vis" in cand and has_visual and rrf_use_visual:
                    score += 1.0 / (k_val + cand["rank_vis"])
                if "rank_ocr" in cand and has_ocr:
                    score += 1.0 / (k_val + cand["rank_ocr"])
                if "rank_asr" in cand and has_speech and rrf_use_asr:
                    score += 1.0 / (k_val + cand["rank_asr"])
                if "rank_obj" in cand and has_object:
                    score += 1.0 / (k_val + cand["rank_obj"])

                # Điểm đóng góp từ Video Shot
                if rrf_use_shot and shot_rank_map:
                    cand_shot_id = self.kf_shot_map.get(cand["path"])
                    if not cand_shot_id:
                        p_parts = cand["path"].split("/")
                        cand_shot_id = self.kf_shot_map.get(f"{p_parts[0]}/{p_parts[1].replace('.jpg','')}")
                    if cand_shot_id and cand_shot_id in shot_rank_map:
                        score += 1.0 / (k_val + shot_rank_map[cand_shot_id])
                        cand["rank_shot"] = shot_rank_map[cand_shot_id]

                # Bonus cộng điểm cho frame khớp nhiều kênh đồng thời
                match_channels = (1 if cand["score_visual"] > 0.15 else 0) + \
                                 (1 if cand["score_asr"] > 0.15 else 0) + \
                                 (1 if cand["score_obj"] > 0.3 else 0) + \
                                 (1 if "rank_shot" in cand else 0)
                if match_channels >= 2:
                    score *= 1.25

                cand["raw_rrf"] = score
                raw_rrf_scores.append(score)

            # Chuẩn hóa điểm RRF về [0.0, 1.0]
            max_rrf = max(raw_rrf_scores) if raw_rrf_scores else 1.0
            if max_rrf <= 0:
                max_rrf = 1.0

            for cand in candidates_list:
                cand["final_score"] = round(cand["raw_rrf"] / max_rrf, 4)

        else:
            w_v = max(0.0, weight_visual) if has_visual else 0.0
            w_o = max(0.0, weight_ocr) if has_ocr else 0.0
            w_a = max(0.0, weight_asr) if has_speech else 0.0
            w_obj = max(0.0, weight_object) if has_object else 0.0

            total_weight = w_v + w_o + w_a + w_obj
            if total_weight == 0:
                total_weight = 1.0
                w_v = 1.0

            for cand in candidates_list:
                final_score = (
                    w_v * cand["score_visual"] +
                    w_o * cand["score_ocr"] +
                    w_a * cand["score_asr"] +
                    w_obj * cand["score_obj"]
                ) / total_weight

                # Bonus cộng điểm cho frame khớp nhiều kênh đồng thời
                match_channels = (1 if cand["score_visual"] > 0.15 else 0) + \
                                 (1 if cand["score_asr"] > 0.15 else 0) + \
                                 (1 if cand["score_obj"] > 0.3 else 0)
                if match_channels >= 2:
                    final_score *= 1.25

                cand["final_score"] = round(min(1.0, final_score), 4)

        # Sắp xếp theo điểm tổng giảm dần
        candidates_list.sort(key=lambda x: x["final_score"], reverse=True)

        # --- BƯỚC 4: BỘ LỌC TRÙNG (DEDUPLICATION) ---
        results: List[Dict[str, Any]] = []

        if dedup_mode == "unique":
            seen_videos = set()
            accepted_indices = []
            accepted_records = []
            for cand in candidates_list:
                cand_idx = cand["idx"]
                v_id = cand["path"].split("/")[0]
                if v_id in seen_videos:
                    continue
                
                # Khử trùng liên video (chống cùng 1 cảnh quảng cáo/intro xuất hiện ở video khác)
                if enable_cross_video_dedup and accepted_records:
                    is_cross_dup = False
                    for acc_rec in accepted_records:
                        cand_vec = cand.get("vector")
                        acc_vec = acc_rec.get("vector")
                        if cand_vec is not None and acc_vec is not None:
                            vec_sim = float(np.dot(cand_vec, acc_vec)) / (cand.get("vec_norm", 1.0) * acc_rec.get("vec_norm", 1.0))
                        elif self.features is not None:
                            raw_dot = float(np.dot(self.features[cand_idx], self.features[acc_rec["idx"]]))
                            vec_sim = raw_dot / (self.feature_norms[cand_idx] * self.feature_norms[acc_rec["idx"]])
                        else:
                            vec_sim = 0.0

                        if vec_sim >= cross_video_sim_thresh:
                            is_cross_dup = True
                            break
                    if is_cross_dup:
                        continue

                seen_videos.add(v_id)
                accepted_indices.append(cand_idx)
                accepted_records.append({
                    "idx": cand_idx,
                    "video_id": v_id,
                    "vector": cand.get("vector"),
                    "vec_norm": cand.get("vec_norm")
                })
                results.append(cand)
                if len(results) >= top_k:
                    break

        elif dedup_mode in ["smart", "visual_only", "asr_only"]:
            accepted_indices = []
            accepted_records = []

            for cand in candidates_list:
                cand_idx = cand["idx"]
                cand_path = cand["path"]
                cand_vid = cand_path.split("/")[0]
                cand_frame_str = cand_path.split("/")[1].replace(".jpg", "")
                cand_fps = self.get_video_fps(cand_vid)
                try:
                    cand_frame_num = int(cand_frame_str)
                    cand_sec = cand_frame_num / cand_fps
                except ValueError:
                    cand_sec = 0.0

                cand_seg_id = self.asr_engine.get_segment_id_for_keyframe(cand_path)

                is_duplicate = False

                for acc_rec in accepted_records:
                    acc_idx = acc_rec["idx"]
                    # 1. So sánh vector thị giác chuẩn Cosine Similarity
                    cand_vec = cand.get("vector")
                    acc_vec = acc_rec.get("vector")
                    if cand_vec is not None and acc_vec is not None:
                        vec_sim = float(np.dot(cand_vec, acc_vec)) / (cand.get("vec_norm", 1.0) * acc_rec.get("vec_norm", 1.0))
                    elif self.features is not None:
                        raw_dot = float(np.dot(self.features[cand_idx], self.features[acc_idx]))
                        vec_sim = raw_dot / (self.feature_norms[cand_idx] * self.feature_norms[acc_idx])
                    else:
                        vec_sim = 0.0

                    # 2. Khử trùng thị giác liên video (Cross-Video Dedup chống spam quảng cáo/intro)
                    if cand_vid != acc_rec["video_id"]:
                        if enable_cross_video_dedup and vec_sim >= cross_video_sim_thresh:
                            is_duplicate = True
                            break
                        continue

                    # 3. Khử trùng nội bộ trong cùng 1 video (Intra-Video Dedup)
                    is_visually_similar = vec_sim >= visual_sim_thresh
                    is_same_asr_segment = (cand_seg_id is not None and cand_seg_id == acc_rec["seg_id"])
                    time_diff = abs(cand_sec - acc_rec["time_sec"])
                    is_close_time = time_diff <= 5.0

                    if dedup_mode == "smart":
                        if is_visually_similar and (is_same_asr_segment or is_close_time):
                            is_duplicate = True
                            break
                    elif dedup_mode == "visual_only":
                        if is_visually_similar:
                            is_duplicate = True
                            break
                    elif dedup_mode == "asr_only":
                        if is_same_asr_segment or (is_close_time and is_visually_similar):
                            is_duplicate = True
                            break

                if not is_duplicate:
                    accepted_indices.append(cand_idx)
                    accepted_records.append({
                        "idx": cand_idx,
                        "video_id": cand_vid,
                        "time_sec": cand_sec,
                        "seg_id": cand_seg_id,
                        "vector": cand.get("vector"),
                        "vec_norm": cand.get("vec_norm")
                    })
                    results.append(cand)
                    if len(results) >= top_k:
                        break
        else:
            results = candidates_list[:top_k]

        # --- BƯỚC 5: ĐỊNH DẠNG ĐẦU RA & ĐÍNH KÈM BOUNDING BOXES ---
        formatted_results = []
        for r in results:
            parts = r["path"].split("/")
            v_id = parts[0]
            f_id = parts[1].replace(".jpg", "")

            # Trích xuất mốc giây
            fps = self.get_video_fps(v_id)
            try:
                frame_sec = int(f_id) / fps
            except ValueError:
                frame_sec = 0.0

            # Lấy danh sách Bounding Boxes từ metadata
            boxes = self.get_boxes_for_keyframe(v_id, f_id, raw_boxes_json=r.get("boxes_json"))

            # Lấy thông tin lời thoại thực tế của keyframe kèm ngữ cảnh mở rộng (±asr_window_n segments)
            sp_exp = self.asr_engine.get_speech_for_keyframe_expanded(r["path"], window_n=asr_window_n)
            transcript = sp_exp.get("text", "") if sp_exp.get("text") else r.get("speech_text", "")
            asr_st = sp_exp.get("start", 0.0) if sp_exp.get("start") else r.get("speech_start", 0.0)
            asr_et = sp_exp.get("end", 0.0) if sp_exp.get("end") else r.get("speech_end", 0.0)
            asr_prev = sp_exp.get("prev_text", "")
            asr_next = sp_exp.get("next_text", "")
            # Lấy thông tin Video Shot tương ứng với Keyframe này
            cand_shot_id = self.kf_shot_map.get(r["path"])
            if not cand_shot_id:
                p_parts = r["path"].split("/")
                cand_shot_id = self.kf_shot_map.get(f"{p_parts[0]}/{p_parts[1].replace('.jpg','')}")
            
            shot_meta = self.shot_meta_map.get(cand_shot_id, {}) if cand_shot_id else {}
            cut_file = shot_meta.get("cut_file", "")

            formatted_results.append({
                "video_id": v_id,
                "frame_id": f_id,
                "keyframe": r["path"],
                "score": r["final_score"],
                "visual_score": r["score_visual"],
                "ocr_score": r["score_ocr"],
                "asr_score": r["score_asr"],
                "obj_score": r.get("score_obj", 0.0),
                "timestamp_sec": round(frame_sec, 2),
                "asr_transcript": transcript,
                "asr_prev": asr_prev,
                "asr_next": asr_next,
                "asr_all": sp_exp.get("all_text", transcript),
                "asr_start": asr_st,
                "asr_end": asr_et,
                "boxes": boxes,
                # Thông tin Video Shot gắn liền với khoảnh khắc Top-K
                "shot_id": cand_shot_id,
                "cut_file": cut_file,
                "shot_start_sec": shot_meta.get("start_sec", 0.0),
                "shot_end_sec": shot_meta.get("end_sec", 0.0),
                "shot_duration_sec": shot_meta.get("duration_sec", 0.0),
                "shot_index": shot_meta.get("shot_index", 1),
                # Minh bạch thứ hạng từng kênh thành phần (Transparency Ranks)
                "rank_vis": r.get("rank_vis"),
                "rank_asr": r.get("rank_asr"),
                "rank_shot": r.get("rank_shot"),
                "rank_ocr": r.get("rank_ocr"),
                "rank_obj": r.get("rank_obj")
            })

        # Định dạng danh sách Video Shots (nếu return_shots=True)
        formatted_shot_results = []
        if return_shots:
            shot_scores = {}
            shot_best_kf = {}
            k_val = max(1, int(rrf_k))

            # 1. Điểm từ Video Shot vector search trực tiếp
            if rrf_use_shot and shot_cands:
                for r_idx, sc in enumerate(shot_cands):
                    s_id = sc["shot_id"]
                    shot_scores[s_id] = shot_scores.get(s_id, 0.0) + 1.0 / (k_val + r_idx + 1)

            # 2. Điểm từ Keyframe matches (Visual, ASR)
            for cand in candidates_list:
                s_id = self.kf_shot_map.get(cand["path"])
                if not s_id:
                    p_parts = cand["path"].split("/")
                    s_id = self.kf_shot_map.get(f"{p_parts[0]}/{p_parts[1].replace('.jpg','')}")
                if not s_id:
                    continue

                if s_id not in shot_best_kf or cand.get("final_score", 0) > shot_best_kf[s_id].get("final_score", 0):
                    shot_best_kf[s_id] = cand

                if use_rrf:
                    if rrf_use_visual and "rank_vis" in cand:
                        shot_scores[s_id] = shot_scores.get(s_id, 0.0) + 1.0 / (k_val + cand["rank_vis"])
                    if rrf_use_asr and "rank_asr" in cand:
                        shot_scores[s_id] = shot_scores.get(s_id, 0.0) + 1.0 / (k_val + cand["rank_asr"])
                else:
                    shot_scores[s_id] = max(shot_scores.get(s_id, 0.0), cand.get("final_score", 0.0))

            sorted_shots = sorted(shot_scores.items(), key=lambda x: x[1], reverse=True)
            max_s_score = sorted_shots[0][1] if sorted_shots else 1.0
            if max_s_score <= 0:
                max_s_score = 1.0

            for s_id, raw_s_score in sorted_shots[:top_k]:
                meta = self.shot_meta_map.get(s_id, {})
                v_id = meta.get("video_id", s_id.split("_shot_")[0])
                b_kf = shot_best_kf.get(s_id)

                b_kf_frame_id = ""
                b_kf_path = ""
                if b_kf:
                    b_kf_path = b_kf.get("path", "")
                    b_kf_frame_id = b_kf_path.split("/")[1].replace(".jpg", "") if "/" in b_kf_path else ""
                else:
                    anchor = meta.get("anchor_frame_id", 0)
                    b_kf_frame_id = str(anchor).zfill(6)
                    b_kf_path = f"{v_id}/{b_kf_frame_id}.jpg"

                cut_file = meta.get("cut_file", "")
                norm_score = round(raw_s_score / max_s_score, 4)

                formatted_shot_results.append({
                    "shot_id": s_id,
                    "video_id": v_id,
                    "shot_index": meta.get("shot_index", 1),
                    "start_sec": meta.get("start_sec", 0.0),
                    "end_sec": meta.get("end_sec", 0.0),
                    "duration_sec": meta.get("duration_sec", 0.0),
                    "anchor_frame_id": meta.get("anchor_frame_id", 0),
                    "cut_file": cut_file,
                    "score": norm_score,
                    "best_kf_frame_id": b_kf_frame_id,
                    "best_kf_path": b_kf_path
                })

        print(f"[SearchEngine] ⏱️ Truy vấn hoàn tất trong {time.time() - t0:.3f}s | Trả về {len(formatted_results)} KFs, {len(formatted_shot_results)} Shots.")
        if return_shots:
            return {
                "results": formatted_results,
                "shot_results": formatted_shot_results
            }
        return formatted_results

    def search_multi_scene(
        self,
        scenes: List[Dict[str, Any]],
        enable_visual: bool = True,
        enable_ocr: bool = True,
        enable_asr: bool = True,
        enable_object: bool = True,
        object_class_id: Optional[int] = None,
        object_conf_thresh: float = 0.0,
        weight_visual: float = 0.50,
        weight_ocr: float = 0.20,
        weight_asr: float = 0.20,
        weight_object: float = 0.10,
        dedup_mode: str = "smart",
        visual_sim_thresh: float = 0.90,
        enable_cross_video_dedup: bool = True,
        cross_video_sim_thresh: float = 0.90,
        top_k: int = 100,
        top_v: int = 5,
        k_per_video: int = 40,
        use_rrf: bool = True,
        rrf_k: int = 60,
        trake_mode: bool = False,
        lambda_penalty: float = 0.001,
        alpha_fusion: float = 0.35,
        k_paths: int = 3,
        trake_top_k: int = 10,
        asr_window_n: int = 1,
        video_id: str = "",
        include_videos: Optional[List[str]] = None,
        exclude_videos: Optional[List[str]] = None,
        category: str = "",
        sub_category: str = "",
        rrf_use_visual: bool = True,
        rrf_use_asr: bool = True,
        rrf_use_shot: bool = True,
        return_shots: bool = True
    ) -> Dict[str, Any]:
        """
        Truy vấn Đa Phân Cảnh (Multi-Scene / TRAKE Temporal Query Decomposition & Video Voting).
        Tích hợp thuật toán TARS Monotonic DP (Prefix-Maximum Recurrence & Alpha Fusion chuẩn AIC 2025/2026).
        Hỗ trợ toàn diện Multimodal + RRF Đa Tín Hiệu (Visual / ASR / Shot) và ánh xạ Shot cho từng sự kiện.
        """
        t0 = time.time()
        if not scenes:
            return {
                "top_videos": [],
                "scenes_results": [],
                "storyboard_sequences": [],
                "trake_submission_lines": [],
                "combined_results": [],
                "total_candidates": 0,
                "elapsed_sec": 0.0
            }

        num_scenes = len(scenes)
        scene_outputs = []
        video_scene_scores: Dict[str, Dict[int, float]] = {}

        # Tôn trọng kích thước Candidate Pool (top_k) do người dùng thiết lập
        effective_top_k = max(top_k, 10)
        effective_top_v = max(top_v, trake_top_k if trake_mode else top_v, 1)

        encoder = get_text_encoder()
        scene_query_vecs = []

        # 1. Quét độc lập từng phân cảnh S_i với toàn bộ thiết lập động hiện tại
        for s_idx, sc in enumerate(scenes):
            sc_vis = sc.get("visual_query", "").strip() if enable_visual else ""
            sc_ocr = sc.get("ocr_query", "").strip() if enable_ocr else ""
            sc_asr = sc.get("speech_query", "").strip() if enable_asr else ""

            prompt_parts = []
            if sc_vis:
                prompt_parts.append(sc_vis)
            if sc_ocr:
                prompt_parts.append(f"văn bản chữ trong ảnh: {sc_ocr}")
            sc_prompt = ", ".join(prompt_parts)
            sc_vec = encoder.encode(sc_prompt) if sc_prompt else None
            scene_query_vecs.append(sc_vec)

            # Nhận diện object class nếu phân cảnh có chỉ định hoặc lấy theo filter chung
            sc_obj_dict = sc.get("object_filter")
            sc_obj_id = sc_obj_dict.get("class_id") if (enable_object and sc_obj_dict) else object_class_id
            sc_obj_thresh = sc_obj_dict.get("min_confidence", object_conf_thresh) if (enable_object and sc_obj_dict) else object_conf_thresh

            search_out = self.search(
                query_text=sc_vis,
                ocr_query=sc_ocr,
                speech_query=sc_asr,
                enable_asr=enable_asr,
                object_class_id=sc_obj_id if enable_object else None,
                object_conf_thresh=sc_obj_thresh if enable_object else 0.0,
                weight_visual=weight_visual,
                weight_ocr=weight_ocr,
                weight_asr=weight_asr,
                weight_object=weight_object if enable_object else 0.0,
                top_k=effective_top_k,
                dedup_mode=dedup_mode,
                visual_sim_thresh=visual_sim_thresh,
                enable_cross_video_dedup=enable_cross_video_dedup,
                cross_video_sim_thresh=cross_video_sim_thresh,
                video_id=video_id,
                include_videos=include_videos,
                exclude_videos=exclude_videos,
                category=category,
                sub_category=sub_category,
                use_rrf=use_rrf,
                rrf_k=rrf_k,
                asr_window_n=asr_window_n,
                query_vec=sc_vec,
                rrf_use_visual=rrf_use_visual,
                rrf_use_asr=rrf_use_asr,
                rrf_use_shot=rrf_use_shot,
                return_shots=return_shots
            )

            if isinstance(search_out, dict):
                sc_results = search_out.get("results", [])
                sc_shots = search_out.get("shot_results", [])
            else:
                sc_results = search_out
                sc_shots = []

            scene_outputs.append({
                "scene_idx": sc.get("scene_idx", s_idx + 1),
                "name": sc.get("name", f"Cảnh {s_idx + 1}"),
                "visual_query": sc_vis,
                "results": sc_results,
                "shot_results": sc_shots
            })

            # Tích lũy điểm số theo video cho phân cảnh này
            for r in sc_results:
                vid = r["video_id"]
                if vid not in video_scene_scores:
                    video_scene_scores[vid] = {}
                video_scene_scores[vid][s_idx] = max(video_scene_scores[vid].get(s_idx, 0.0), float(r["score"]))

        # 2. Giao thoa Video & Video Voting
        video_joint_scores: Dict[str, float] = {}
        for vid, s_dict in video_scene_scores.items():
            matched_scenes_count = len(s_dict)
            base_score = sum(s_dict.values())

            # Hệ số thưởng Giao thoa: Video xuất hiện ở TẤT CẢ các phân cảnh được nhân thưởng cao nhất
            if matched_scenes_count == num_scenes and num_scenes > 1:
                bonus_mult = 1.60
            elif matched_scenes_count > 1:
                bonus_mult = 1.0 + 0.20 * (matched_scenes_count - 1)
            else:
                bonus_mult = 0.80 if trake_mode else 1.0

            video_joint_scores[vid] = base_score * bonus_mult

        top_videos = sorted(video_joint_scores.keys(), key=lambda v: video_joint_scores[v], reverse=True)[:effective_top_v]

        # 3. Trích xuất frames chi tiết cho từng video trong Top N theo từng phân cảnh
        refined_scenes_results = []
        all_combined_frames = []

        for s_idx, sc in enumerate(scenes):
            sc_vis = sc.get("visual_query", "").strip() if enable_visual else ""
            sc_ocr = sc.get("ocr_query", "").strip() if enable_ocr else ""
            sc_asr = sc.get("speech_query", "").strip() if enable_asr else ""
            sc_vec = scene_query_vecs[s_idx]

            sc_obj_dict = sc.get("object_filter")
            sc_obj_id = sc_obj_dict.get("class_id") if (enable_object and sc_obj_dict) else object_class_id
            sc_obj_thresh = sc_obj_dict.get("min_confidence", object_conf_thresh) if (enable_object and sc_obj_dict) else object_conf_thresh

            sc_frames_for_top_videos = []
            for vid in top_videos:
                vid_res = self.search(
                    query_text=sc_vis,
                    ocr_query=sc_ocr,
                    speech_query=sc_asr,
                    enable_asr=enable_asr,
                    object_class_id=sc_obj_id if enable_object else None,
                    object_conf_thresh=sc_obj_thresh if enable_object else 0.0,
                    video_id=vid,
                    weight_visual=weight_visual,
                    weight_ocr=weight_ocr,
                    weight_asr=weight_asr,
                    weight_object=weight_object if enable_object else 0.0,
                    top_k=k_per_video,
                    dedup_mode=dedup_mode,
                    visual_sim_thresh=visual_sim_thresh,
                    enable_cross_video_dedup=enable_cross_video_dedup,
                    cross_video_sim_thresh=cross_video_sim_thresh,
                    use_rrf=use_rrf,
                    rrf_k=rrf_k,
                    query_vec=sc_vec
                )
                for item in vid_res:
                    item["scene_idx"] = sc.get("scene_idx", s_idx + 1)
                    item["scene_name"] = sc.get("name", f"Cảnh {s_idx + 1}")
                sc_frames_for_top_videos.extend(vid_res)

            # Sắp xếp theo score trong từng phân cảnh
            sc_frames_for_top_videos.sort(key=lambda x: x["score"], reverse=True)

            refined_scenes_results.append({
                "scene_idx": sc.get("scene_idx", s_idx + 1),
                "name": sc.get("name", f"Cảnh {s_idx + 1}"),
                "visual_query": sc_vis,
                "results": sc_frames_for_top_videos
            })
            all_combined_frames.extend(sc_frames_for_top_videos)

        # 4. Xây dựng Chuỗi Storyboard Ghép Cảnh theo Video bằng TARS Monotonic DP
        storyboard_sequences = []
        trake_submission_lines = []

        for v_rank, vid in enumerate(top_videos):
            vid_scenes_frames = []
            for s_idx in range(len(scenes)):
                frames_for_s = [f for f in refined_scenes_results[s_idx]["results"] if f["video_id"] == vid]
                frames_for_s.sort(key=lambda x: x["timestamp_sec"])
                vid_scenes_frames.append(frames_for_s)

            global_vid_score = video_joint_scores.get(vid, 0.0)
            
            # Chạy thuật toán TARS Monotonic DP (Prefix-Maximum Recurrence)
            candidate_paths = self._tars_monotonic_dp(
                vid_scenes_frames=vid_scenes_frames,
                lambda_penalty=lambda_penalty,
                alpha=alpha_fusion,
                global_score=global_vid_score,
                k_paths=k_paths if trake_mode else 1
            )

            for path_idx, (seq_items, seq_fused_score, is_strictly_chronological) in enumerate(candidate_paths):
                if not seq_items:
                    continue

                # Format chi tiết chuỗi storyboard
                formatted_items = []
                for i, f in enumerate(seq_items):
                    delta_t = None
                    if i < len(seq_items) - 1:
                        delta_t = round(seq_items[i + 1]["timestamp_sec"] - f["timestamp_sec"], 1)
                    
                    item_dict = dict(f)
                    item_dict["delta_t_to_next"] = delta_t

                    # Ánh xạ Video Shot tương ứng cho event này
                    s_id = self.kf_shot_map.get(f.get("keyframe", ""))
                    if not s_id:
                        s_id = self.kf_shot_map.get(f"{f.get('video_id', '')}/{f.get('frame_id', '')}")
                    if s_id:
                        meta = self.shot_meta_map.get(s_id, {})
                        item_dict["shot_id"] = s_id
                        item_dict["shot_index"] = meta.get("shot_index", 1)
                        item_dict["shot_start_sec"] = meta.get("start_sec", 0.0)
                        item_dict["shot_end_sec"] = meta.get("end_sec", 0.0)
                        item_dict["shot_duration_sec"] = meta.get("duration_sec", 0.0)
                        item_dict["cut_file"] = meta.get("cut_file", "")
                        item_dict["anchor_frame_id"] = meta.get("anchor_frame_id", f.get("frame_id"))

                    formatted_items.append(item_dict)

                # Format dòng CSV nộp bài chuẩn: <video_id>,<F_E1>,<F_E2>,...
                frame_ids_str = ",".join(str(int(f["frame_id"])) for f in seq_items)
                sub_line = f"{vid},{frame_ids_str}"

                storyboard_sequences.append({
                    "video_id": vid,
                    "rank": v_rank + 1,
                    "path_idx": path_idx + 1,
                    "joint_score": round(seq_fused_score, 4),
                    "is_chronological": is_strictly_chronological,
                    "submission_line": sub_line,
                    "scenes": formatted_items
                })

        storyboard_sequences.sort(key=lambda x: x["joint_score"], reverse=True)

        # Cập nhật lại rank sau khi sort
        for r_idx, seq in enumerate(storyboard_sequences):
            seq["rank"] = r_idx + 1
            if seq["submission_line"] not in trake_submission_lines:
                trake_submission_lines.append(seq["submission_line"])

        # 5. Gộp toàn bộ kết quả và loại bỏ frame trùng lặp giữa các phân cảnh
        unique_combined = []
        seen_paths = set()
        for f in all_combined_frames:
            if f["keyframe"] not in seen_paths:
                seen_paths.add(f["keyframe"])
                unique_combined.append(f)

        unique_combined.sort(key=lambda x: x["score"], reverse=True)

        elapsed = round(time.time() - t0, 3)
        print(f"[SearchEngine] ⏱️ Đa phân cảnh (TRAKE={trake_mode}) hoàn tất trong {elapsed}s | Top videos: {top_videos[:5]} | {len(storyboard_sequences)} storyboard rows | {len(unique_combined)} frames.")

        return {
            "top_videos": top_videos,
            "storyboard_sequences": storyboard_sequences[:trake_top_k if trake_mode else len(storyboard_sequences)],
            "trake_submission_lines": trake_submission_lines[:trake_top_k if trake_mode else 100],
            "scenes_results": refined_scenes_results,
            "combined_results": unique_combined,
            "total_candidates": len(unique_combined),
            "elapsed_sec": elapsed
        }

    def search_by_image(
        self,
        query_image,
        query_text: str = "",
        ocr_query: str = "",
        speech_query: str = "",
        enable_asr: bool = True,
        object_class_id: Optional[int] = None,
        object_conf_thresh: float = 0.0,
        dedup_mode: str = "smart",
        visual_sim_thresh: float = 0.90,
        enable_cross_video_dedup: bool = True,
        cross_video_sim_thresh: float = 0.90,
        video_id: str = "",
        include_videos: Optional[List[str]] = None,
        exclude_videos: Optional[List[str]] = None,
        category: str = "",
        sub_category: str = "",
        weight_visual: float = 1.0,
        weight_ocr: float = 1.0,
        weight_asr: float = 1.0,
        weight_object: float = 0.5,
        image_weight: float = 0.5,
        top_k: int = config.TOP_K,
        use_rrf: bool = True,
        rrf_k: int = 60,
        asr_window_n: int = 1,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Tìm kiếm hình ảnh kết hợp đa phương thức (Composed Multimodal Retrieval: Image + Text).
        Sử dụng Qwen3-VL (2048D) và hỗ trợ 100% toàn bộ tính năng:
        - Image + Text Composed Vector (Qwen3-VL 2048D)
        - OCR Query & Text matching
        - ASR Speech Query & TF-IDF Semantic search
        - RT-DETR Object Detection & Bounding Boxes
        - Weighted Fusion & RRF Fusion
        - Intra-video & Cross-video Deduplication
        - SQL Pushdown Filters (Tags Include/Exclude, Category & Sub-category)
        """
        t0 = time.time()

        results = self.search(
            query_text=query_text,
            ocr_query=ocr_query,
            speech_query=speech_query,
            enable_asr=enable_asr,
            object_class_id=object_class_id,
            object_conf_thresh=object_conf_thresh,
            dedup_mode=dedup_mode,
            visual_sim_thresh=visual_sim_thresh,
            enable_cross_video_dedup=enable_cross_video_dedup,
            cross_video_sim_thresh=cross_video_sim_thresh,
            video_id=video_id,
            include_videos=include_videos,
            exclude_videos=exclude_videos,
            category=category,
            sub_category=sub_category,
            weight_visual=weight_visual,
            weight_ocr=weight_ocr,
            weight_asr=weight_asr,
            weight_object=weight_object,
            top_k=top_k,
            use_rrf=use_rrf,
            rrf_k=rrf_k,
            asr_window_n=asr_window_n,
            query_image=query_image,
            image_weight=image_weight
        )

        print(f"[SearchEngine] 🖼️ Multimodal Search hoàn tất trong {time.time() - t0:.3f}s | Trả về {len(results)} kết quả.")
        return results

    def _tars_monotonic_dp(
        self,
        vid_scenes_frames: List[List[Dict[str, Any]]],
        lambda_penalty: float = 0.001,
        alpha: float = 0.35,
        global_score: float = 1.0,
        k_paths: int = 3
    ) -> List[Tuple[List[Dict[str, Any]], float, bool]]:
        """
        Thuật toán TARS Monotonic DP (Prefix-Maximum Recurrence) cho bài toán TRAKE:
        dp[s, j] = Sim[s, j] + max_{k < j} (dp[s-1, k] - lambda * (time_j - time_k))
        Final Score = alpha * GlobalScore + (1 - alpha) * DPScore
        
        Hỗ trợ trích xuất Top-k đa dạng (diverse paths) cho cùng một video.
        """
        num_s = len(vid_scenes_frames)
        if num_s == 0:
            return []

        # Kiểm tra nếu có cảnh nào trống không có frame -> Fallback lấy frame cao nhất của từng cảnh
        has_empty_scene = any(len(sf) == 0 for sf in vid_scenes_frames)
        if has_empty_scene:
            fallback_seq = []
            for sf in vid_scenes_frames:
                if sf:
                    fallback_seq.append(max(sf, key=lambda x: x["score"]))
            if fallback_seq:
                avg_score = sum(float(f["score"]) for f in fallback_seq) / max(1, len(fallback_seq))
                fused = alpha * global_score + (1.0 - alpha) * avg_score
                return [(fallback_seq, fused, False)]
            return []

        # dp[s][j] = list of (score, prev_frame_idx)
        # Khởi tạo Scene 0
        dp = [[] for _ in range(num_s)]
        for j, f in enumerate(vid_scenes_frames[0]):
            sim_score = float(f["score"])
            dp[0].append([(sim_score, -1)])  # list of best paths ending at (0, j)

        # Tiến hành DP với Prefix-Max và Lambda Penalty
        for s in range(1, num_s):
            curr_frames = vid_scenes_frames[s]
            prev_frames = vid_scenes_frames[s - 1]
            dp[s] = [[] for _ in range(len(curr_frames))]

            for j, curr_f in enumerate(curr_frames):
                curr_sim = float(curr_f["score"])
                curr_t = curr_f["timestamp_sec"]

                # Tìm các bước chuyển hợp lệ từ scene s-1 thỏa mãn t_prev < t_curr
                candidate_transitions = []
                for k, prev_f in enumerate(prev_frames):
                    prev_t = prev_f["timestamp_sec"]
                    if curr_t > prev_t:
                        # Tính penalty theo khoảng cách thời gian giữa 2 sự kiện
                        delta_t = curr_t - prev_t
                        penalty = lambda_penalty * delta_t
                        
                        for prev_path_score, _ in dp[s - 1][k]:
                            trans_score = prev_path_score + curr_sim - penalty
                            candidate_transitions.append((trans_score, k))

                if candidate_transitions:
                    # Sắp xếp lấy top transitions cho vị trí (s, j)
                    candidate_transitions.sort(key=lambda x: x[0], reverse=True)
                    dp[s][j] = candidate_transitions[:k_paths]
                else:
                    # Nếu không có frame nào trước đó có t < curr_t -> Fallback không ràng buộc thời gian
                    for k, prev_f in enumerate(prev_frames):
                        for prev_path_score, _ in dp[s - 1][k]:
                            trans_score = prev_path_score + curr_sim * 0.5 - 0.2  # Phạt nặng vì sai thứ tự
                            candidate_transitions.append((trans_score, k))
                    if candidate_transitions:
                        candidate_transitions.sort(key=lambda x: x[0], reverse=True)
                        dp[s][j] = candidate_transitions[:1]

        # Thu thập tất cả các endpoint ở Scene cuối cùng
        last_scene_candidates = []
        for j in range(len(vid_scenes_frames[num_s - 1])):
            for score, prev_idx in dp[num_s - 1][j]:
                last_scene_candidates.append((score, j, prev_idx))

        if not last_scene_candidates:
            fallback_seq = [max(sf, key=lambda x: x["score"]) for sf in vid_scenes_frames if sf]
            avg_score = sum(float(f["score"]) for f in fallback_seq) / max(1, len(fallback_seq))
            fused = alpha * global_score + (1.0 - alpha) * avg_score
            return [(fallback_seq, fused, False)]

        # Sắp xếp điểm DP giảm dần
        last_scene_candidates.sort(key=lambda x: x[0], reverse=True)

        results = []
        seen_seq_hashes = set()

        for dp_score, last_j, last_prev in last_scene_candidates[:k_paths * 2]:
            # Truy vết chuỗi từ Scene N-1 về Scene 0
            seq = [vid_scenes_frames[num_s - 1][last_j]]
            curr_prev = last_prev

            is_chronological = True
            for s in range(num_s - 2, -1, -1):
                if curr_prev >= 0 and curr_prev < len(vid_scenes_frames[s]):
                    prev_f = vid_scenes_frames[s][curr_prev]
                    seq.append(prev_f)
                    
                    # Tìm tiếp parent của prev_f
                    if dp[s][curr_prev]:
                        curr_prev = dp[s][curr_prev][0][1]
                    else:
                        curr_prev = -1
                else:
                    # Fallback nếu mất dấu
                    if vid_scenes_frames[s]:
                        seq.append(vid_scenes_frames[s][0])
                    is_chronological = False

            seq.reverse()
            
            # Kiểm tra tính tuần tự thời gian nghiêm ngặt
            for i in range(len(seq) - 1):
                if seq[i]["timestamp_sec"] >= seq[i + 1]["timestamp_sec"]:
                    is_chronological = False
                    break

            seq_hash = tuple(f["frame_id"] for f in seq)
            if seq_hash in seen_seq_hashes:
                continue
            seen_seq_hashes.add(seq_hash)

            # Điểm dung hợp Alpha Fusion chuẩn TARS
            avg_seq_score = sum(float(f["score"]) for f in seq) / max(1, len(seq))
            if is_chronological:
                avg_seq_score *= 1.25  # Thưởng thêm cho chuỗi hoàn hảo
            
            fused_score = alpha * global_score + (1.0 - alpha) * avg_seq_score
            results.append((seq, fused_score, is_chronological))

            if len(results) >= k_paths:
                break

        return results


# Singleton Instance
_search_engine_instance: Optional[VideoSearchEngine] = None


def get_search_engine() -> VideoSearchEngine:
    global _search_engine_instance
    if _search_engine_instance is None:
        _search_engine_instance = VideoSearchEngine()
    return _search_engine_instance

