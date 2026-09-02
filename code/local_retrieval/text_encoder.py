"""
TextEncoder — Dùng Qwen/Qwen3-VL-Embedding-2B để encode text query.
Tuân thủ 100% chuẩn bài báo và tài liệu của tác giả Qwen Team:
Sử dụng Instruct Prefix ("Instruct: Given a search query, retrieve relevant images\\nQuery: ")
để kích hoạt đúng không gian vector Text-to-Image Contrastive Space 2048 dim.
"""
import os
import numpy as np
from typing import Optional, List, Dict, Any, Union
from sentence_transformers import SentenceTransformer

try:
    from . import config
except (ImportError, ValueError):
    import config

# Tắt warning song song hóa của tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Tiền tố Task Instruction chính thức từ bài báo Qwen3-VL-Embedding
OFFICIAL_INSTRUCT_PROMPT = "Instruct: Given a search query, retrieve relevant images\nQuery: "


class TextEncoder:
    """
    Nhúng text query thành vector 2048 chiều bằng Qwen/Qwen3-VL-Embedding-2B.
    Singleton — chỉ load model 1 lần duy nhất vào VRAM/RAM.
    """

    def __init__(self, model_name: str = None, instruct_prompt: str = OFFICIAL_INSTRUCT_PROMPT):
        self.model_name = model_name if model_name is not None else config.TEXT_MODEL
        self.instruct_prompt = instruct_prompt
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            print(f"[TextEncoder] Đang nạp model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            print(f"[TextEncoder] ✅ Nạp model thành công với Instruct Prefix chuẩn tác giả.")

    def encode(self, text: str) -> np.ndarray:
        """
        Nhận vào một chuỗi text query, áp dụng Instruct Prefix chuẩn tác giả Qwen,
        trả về numpy array 1D shape (2048,) đã được L2-normalize.
        """
        self._load()
        # Áp dụng chính xác Task Instruction Prefix của tác giả Qwen
        vec = self._model.encode(
            [text],
            prompt=self.instruct_prompt,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return vec[0].astype(np.float32)

    def _to_pil(self, img_input):
        """Chuyển đổi các định dạng đầu vào (path, ndarray, PIL) sang PIL.Image RGB."""
        from PIL import Image
        if isinstance(img_input, str):
            return Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input).convert("RGB")
        elif isinstance(img_input, Image.Image):
            return img_input.convert("RGB")
        else:
            raise ValueError(f"Định dạng ảnh không hỗ trợ: {type(img_input)}")

    def encode_image(self, image_input) -> np.ndarray:
        """
        Nhận vào 1 ảnh hoặc danh sách nhiều góc ảnh của cùng một đối tượng (Multi-View).
        Trả về numpy array 1D shape (2048,) đã được L2-normalize.
        Dùng chung không gian vector 2048D với 358,639 keyframes trong LanceDB.
        """
        self._load()
        if isinstance(image_input, (list, tuple)):
            if len(image_input) == 0:
                raise ValueError("Danh sách ảnh rỗng!")
            pil_images = [self._to_pil(img) for img in image_input]
            vecs = self._model.encode(
                pil_images,
                prompt_name="document",
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # Tính vector trung bình các góc nhìn và chuẩn hóa lại L2
            mean_vec = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-6:
                mean_vec = mean_vec / norm
            return mean_vec.astype(np.float32)
        else:
            image = self._to_pil(image_input)
            vec = self._model.encode(
                [image],
                prompt_name="document",
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return vec[0].astype(np.float32)

    def encode_composed_query(
        self,
        image_input=None,
        text_query: str = "",
        image_weight: float = 0.5
    ) -> np.ndarray:
        """
        Mã hóa truy vấn kết hợp Composed Multimodal Query (Image(s) + Text) trong Qwen3-VL 2048D:
        - Hỗ trợ cả 1 ảnh hoặc nhiều góc ảnh của cùng một đối tượng.
        - Sử dụng Composed Prompting kết hợp trọng số để tạo ra vector 2048D hội tụ cả bối cảnh hình ảnh lẫn yêu cầu ngôn ngữ.
        """
        has_img = image_input is not None and (not isinstance(image_input, (list, tuple)) or len(image_input) > 0)
        has_txt = bool(text_query and str(text_query).strip())

        if not has_img and not has_txt:
            raise ValueError("Cần ít nhất 1 trong 2: image_input hoặc text_query!")

        if has_img and not has_txt:
            return self.encode_image(image_input)

        if has_txt and not has_img:
            return self.encode(text_query)

        self._load()
        txt = str(text_query).strip()

        # 1. Composed Vector từ 1 hoặc nhiều ảnh
        if isinstance(image_input, (list, tuple)):
            pil_images = [self._to_pil(img) for img in image_input]
            prompt_str = f"Instruct: Given these multi-view query images of the object, retrieve video frames satisfying: {txt}\nQuery: "
            composed_vecs = self._model.encode(
                pil_images,
                prompt=prompt_str,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            v_composed = np.mean(composed_vecs, axis=0)
            norm_comp = np.linalg.norm(v_composed)
            if norm_comp > 1e-6:
                v_composed = v_composed / norm_comp
        else:
            image = self._to_pil(image_input)
            prompt_str = f"Instruct: Given this query image, retrieve video frames satisfying: {txt}\nQuery: "
            v_composed = self._model.encode(
                [image],
                prompt=prompt_str,
                convert_to_numpy=True,
                normalize_embeddings=True
            )[0]

        # 2. Text Vector riêng biệt
        v_text = self.encode(txt)

        # 3. Kết hợp có trọng số (Weighted Normalized Spherical Interpolation)
        w_img = max(0.0, min(1.0, float(image_weight)))
        w_txt = 1.0 - w_img
        v_fused = (w_img * v_composed) + (w_txt * v_text)
        norm = np.linalg.norm(v_fused)
        if norm > 1e-6:
            v_fused = v_fused / norm
        return v_fused.astype(np.float32)

    def encode_visual(
        self,
        text: str = "",
        image: Any = None,
        image_weight: float = 0.5
    ) -> Optional[np.ndarray]:
        """
        Hàm Đa Hình Trích Xuất Vector Thị Giác (Polymorphic Visual Encoder):
        - Hỗ trợ Text, 1 ảnh đơn lẻ hoặc danh sách nhiều góc ảnh của cùng đối tượng.
        """
        has_txt = bool(text and str(text).strip())
        has_img = image is not None and (not isinstance(image, (list, tuple)) or len(image) > 0)

        if not has_txt and not has_img:
            return None

        if has_txt and not has_img:
            return self.encode(text)

        if has_img and not has_txt:
            return self.encode_image(image)

        # Cả Chữ và Ảnh cùng tồn tại (Composed Multi-view Query):
        return self.encode_composed_query(
            image_input=image,
            text_query=text,
            image_weight=image_weight
        )


# Singleton instance
_instance: TextEncoder = None


def get_text_encoder() -> TextEncoder:
    global _instance
    if _instance is None:
        _instance = TextEncoder()
    return _instance
