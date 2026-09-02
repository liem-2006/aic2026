import os
import sys
import importlib

# Màu sắc console ANSI
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"

def print_header(title):
    print(f"\n{BLUE}{BOLD}{'=' * 75}{RESET}")
    print(f"{BLUE}{BOLD}🔍 {title}{RESET}")
    print(f"{BLUE}{BOLD}{'=' * 75}{RESET}")

def print_status(name, status, detail=""):
    if status == "OK":
        badge = f"{GREEN}[ĐẠT - OK]{RESET}"
    elif status == "WARN":
        badge = f"{YELLOW}[CẢNH BÁO]{RESET}"
    else:
        badge = f"{RED}[THIẾU / LỖI]{RESET}"
    print(f"{badge} {BOLD}{name:<28}{RESET} : {detail}")

def main():
    print_header("HỆ THỐNG KIỂM TRA MÔI TRƯỜNG AIC 2026 (HEALTH CHECK)")
    project_root = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(project_root, "Data")
    all_ok = True

    # 1. KIỂM TRA MÔI TRƯỜNG VENV & PYTHON
    venv_standard = os.path.join(project_root, "code", "local_retrieval", "venv")
    venv_root = os.path.join(project_root, "venv")
    if os.path.exists(venv_standard):
        print_status("Vị trí Virtualenv", "OK", "code/local_retrieval/venv (Chuẩn)")
    elif os.path.exists(venv_root):
        print_status("Vị trí Virtualenv", "OK", "venv/ (Gốc)")
    else:
        print_status("Vị trí Virtualenv", "WARN", "Chưa tìm thấy venv tại code/local_retrieval/venv")

    py_version = sys.version.split()[0]
    if sys.version_info >= (3, 10):
        print_status("Phiên bản Python", "OK", f"Python {py_version}")
    else:
        print_status("Phiên bản Python", "WARN", f"Python {py_version} (Khuyến nghị 3.10 trở lên)")

    # 2. KIỂM TRA THƯ VIỆN CỐT LÕI
    print_header("1. KIỂM TRA THƯ VIỆN PYTHON (DEPENDENCIES)")
    required_pkgs = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("sentence_transformers", "Sentence Transformers"),
        ("lancedb", "LanceDB Vector Engine"),
        ("pyarrow", "Apache Arrow Data Format"),
        ("fastapi", "FastAPI Web Backend"),
        ("uvicorn", "Uvicorn ASGI Server"),
        ("PIL", "Pillow Image Library"),
        ("av", "PyAV Video Stream Library"),
        ("google.generativeai", "Google Generative AI (Gemini)")
    ]

    missing_pkgs = []
    for mod_name, label in required_pkgs:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "Ready")
            print_status(label, "OK", f"v{ver}")
        except ImportError as e:
            print_status(label, "FAIL", f"Chưa cài đặt ({e})")
            missing_pkgs.append(mod_name)
            all_ok = False
        except Exception as e:
            print_status(label, "WARN", f"Cảnh báo nạp: {e}")

    # 3. KIỂM TRA PHẦN CỨNG & GPU
    print_header("2. KIỂM TRA PHẦN CỨNG & KHẢ NĂNG TĂNG TỐC ĐỒ HỌA")
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_status("Bộ tăng tốc GPU", "OK", f"{device_name} ({vram_gb:.1f} GB VRAM)")
        else:
            print_status("Bộ tăng tốc GPU", "WARN", "Không phát hiện GPU NVIDIA. Hệ thống sẽ tự động chạy trên CPU!")
    except Exception as e:
        print_status("Bộ tăng tốc GPU", "WARN", str(e))

    # 4. KIỂM TRA DỮ LIỆU & TÀI NGUYÊN BẢO MẬT (DATA DIRECTORIES)
    print_header("3. KIỂM TRA TÀI NGUYÊN DỮ LIỆU (CẦN NHẬN TỪ TRƯỞNG NHÓM)")
    
    # 4.1 LanceDB Database
    lancedb_path = os.path.join(data_dir, "aic_lancedb")
    if os.path.exists(lancedb_path):
        try:
            import lancedb
            db = lancedb.connect(lancedb_path)
            if "keyframes" in db.table_names():
                tbl = db.open_table("keyframes")
                count = len(tbl)
                print_status("Vector DB (aic_lancedb)", "OK", f"Đã nạp {count:,} keyframes thành công!")
            else:
                print_status("Vector DB (aic_lancedb)", "WARN", "Thư mục tồn tại nhưng thiếu bảng 'keyframes'.")
        except Exception as e:
            print_status("Vector DB (aic_lancedb)", "FAIL", f"Lỗi đọc LanceDB: {e}")
            all_ok = False
    else:
        print_status("Vector DB (aic_lancedb)", "FAIL", f"Không tìm thấy thư mục: {lancedb_path}")
        print(f"   {YELLOW}👉 Hướng dẫn: Liên hệ Trưởng nhóm để nhận thư mục 'aic_lancedb' và dán vào thư mục Data/{RESET}")
        all_ok = False

    # 4.2 Keyframes Folder
    kf_candidates = [
        os.path.join(data_dir, "Custom_Keyframes"),
        os.path.join(data_dir, "batch 1", "Custom_Keyframes")
    ]
    kf_found = any(os.path.exists(p) for p in kf_candidates)
    if kf_found:
        actual_kf = next(p for p in kf_candidates if os.path.exists(p))
        n_folders = len([d for d in os.listdir(actual_kf) if os.path.isdir(os.path.join(actual_kf, d))])
        print_status("Ảnh Keyframes (Visual)", "OK", f"Tìm thấy tại {os.path.basename(actual_kf)} ({n_folders} videos)")
    else:
        print_status("Ảnh Keyframes (Visual)", "WARN", "Chưa có ảnh Custom_Keyframes (Giao diện web vẫn tìm kiếm được nhưng không hiện ảnh preview).")
        print(f"   {YELLOW}👉 Hướng dẫn: Nhận thư mục 'Custom_Keyframes' và đặt vào Data/{RESET}")

    # 4.3 Gemini API Key
    sys.path.insert(0, os.path.join(project_root, "code", "local_retrieval"))
    try:
        import config
        if config.GEMINI_API_KEY:
            masked_key = config.GEMINI_API_KEY[:6] + "..." + config.GEMINI_API_KEY[-4:]
            print_status("Khóa Gemini API", "OK", f"Đã cấu hình ({masked_key})")
        else:
            print_status("Khóa Gemini API", "WARN", "Chưa cấu hình (Trợ lý AI phân tích truy vấn sẽ bị vô hiệu hóa).")
            print(f"   {YELLOW}👉 Hướng dẫn: Copy file .env.example thành .env và điền GEMINI_API_KEY của bạn vào.{RESET}")
    except Exception as e:
        print_status("Khóa Gemini API", "WARN", f"Lỗi kiểm tra: {e}")

    # 5. TỔNG KẾT VÀ HƯỚNG DẪN TIẾP THEO
    print_header("KẾT LUẬN & HƯỚNG DẪN TRIỂN KHAI")
    if all_ok:
        print(f"{GREEN}{BOLD}🎉 XUẤT SẮC! TOÀN BỘ MÔI TRƯỜNG ĐÃ ĐẠT CHUẨN 100%.{RESET}")
        print(f"\nBạn có thể khởi động hệ thống ngay lập tức bằng cách:")
        print(f"  • Trên Windows : Nháy đúp vào file {BOLD}run_app.bat{RESET} (hoặc chạy: .\\run_app.bat)")
        print(f"  • Trên Linux   : Chạy lệnh {BOLD}bash run_app.sh{RESET}")
        print(f"  • Thủ công     : python code/web/backend/main.py\n")
    else:
        print(f"{RED}{BOLD}⚠️ PHÁT HIỆN MỘT SỐ THÀNH PHẦN CẦN BỔ SUNG TRƯỚC KHI CHẠY:{RESET}")
        if missing_pkgs:
            print(f"\n1. Cài đặt các thư viện còn thiếu:")
            print(f"   {BOLD}pip install -r requirements.txt{RESET}")
        if not os.path.exists(lancedb_path):
            print(f"\n2. Bổ sung cơ sở dữ liệu Vector:")
            print(f"   Nhận file nén {BOLD}aic_lancedb.zip{RESET} từ Trưởng nhóm, giải nén vào thư mục {BOLD}Data/aic_lancedb/{RESET}")
        print()

if __name__ == "__main__":
    main()
