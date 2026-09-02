<div align="center">

# 🎬 AIC 2026: Multi-Modal Video Retrieval & Interactive Search Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20CUDA-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![LanceDB](https://img.shields.io/badge/LanceDB-Vector%20Engine-8A2BE2)](https://lancedb.github.io/lancedb/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An advanced multi-modal interactive video retrieval system built for the Ho Chi Minh City AI Challenge (AIC 2026), supporting Known-Item Search (KIS), Visual Question Answering (VQA), and Ad-hoc Video Search.*

[Tiếng Việt](README_VN.md) • [English](README.md) • [Architecture](#-system-architecture) • [Quickstart](#-quickstart) • [Features](#-key-features)

</div>

---

## 🌟 Highlights

* **High-Dimensional Visual Representation**: Powered by `Qwen/Qwen3-VL-Embedding-2B` producing 2048-dimensional hyperspherical vector embeddings.
* **Sub-Millisecond Retrieval Speed**: Powered by **LanceDB** vector engine using Apache Arrow columnar format, indexing over 350,000 keyframes with latency under $2\text{ms}$.
* **Hybrid Multimodal Signals**:
  * **Visual Semantics**: Dense neural vectors (Qwen3-VL + BEiT-3).
  * **Object Detection & Spatial Constraints**: RT-DETR across 80 COCO classes with pixel-level bounding boxes.
  * **Audio Transcription**: Whisper ASR spoken-word transcripts.
* **Interactive Exploration UI**:
  * Multi-view angle visual search (query with multiple perspectives of the same object).
  * On-canvas interactive Bounding Box filtering in Lightbox modal.
  * Video timeline scrubbing with intra-video semantic ranking.
  * Rapid client-side filtering preserving true global retrieval ranks (`#Rank`).

---

## 🏛️ System Architecture

```
                                  [ User Query ]
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
            [ Natural Language ]                      [ Image / Multi-View ]
            "two chefs in kitchen"                   [Front, Angle, Detail]
                    │                                         │
                    ▼                                         ▼
         [ Qwen3-VL Text Encoder ]               [ Qwen3-VL Vision Encoder ]
           (Instruct Prefix 2048D)                 (Mean/Max Pooling 2048D)
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                             [ LanceDB Vector Engine ]
                          (358,639 Keyframes / 2048D)
                                         │
                   ┌─────────────────────┼─────────────────────┐
                   ▼                     ▼                     ▼
          [ Cosine Similarity ]   [ COCO Detection ]    [ Audio ASR Match ]
             (Dense Vector)       (RT-DETR Bounding Box)   (Whisper Speech)
                   │                     │                     │
                   └─────────────────────┼─────────────────────┘
                                         ▼
                             [ Re-Ranking & Filtering ]
                           (Client Quick Exclude & Filter)
                                         │
                                         ▼
                         [ Interactive Web UI Dashboard ]
```

---

## 📂 Repository & Virtual Environment Structure

```
aic2026/
├── venv/                             # [Recommended Virtual Environment Location]
│   ├── Scripts/ (or bin/ on Linux)
│   └── Lib/site-packages/
│
├── Data/                             # [Confidential Data Root - Ignored on Git]
│   ├── aic_lancedb/                  # Embedded LanceDB table (~4.38 GB)
│   ├── Custom_Keyframes/             # Extracted keyframe images for preview
│   └── video/                        # Original source MP4 videos
│
├── code/
│   ├── local_retrieval/              # Core Search Engine & Model Encoders
│   │   ├── config.py                 # Central configurations & environment loaders
│   │   ├── search_engine.py          # Unified hybrid retrieval engine
│   │   ├── qwen_reranker.py          # Qwen3-VL Reranking Engine
│   │   ├── gemini_parser.py          # LLM Query decomposition & assistance
│   │   └── asr_engine.py             # Whisper transcript matching
│   └── web/
│       ├── backend/                  # FastAPI REST API server (main.py)
│       └── frontend/                 # Client UI (HTML5, Vanilla CSS, App.js)
│
├── query/
│   └── submission_generator.py       # Official KIS submission formatting engine
│
├── verify_setup.py                   # Automated system diagnostics script
├── run_app.bat                       # 1-Click launcher for Windows
├── run_app.sh                        # 1-Click launcher for Linux / macOS
├── requirements.txt                  # Python dependencies
└── .env.example                      # Environment variables template
```

---

## 🚀 Quickstart

### 1. Clone & Environment Setup

Create the Python virtual environment (Python >= 3.10) directly at the project root:

```bash
git clone https://github.com/liem-2006/aic2026.git
cd aic2026

# Create Virtual Environment at project root (venv/)
python -m venv venv

# Activate Virtual Environment
# On Windows:
.\venv\Scripts\activate
# On Linux / macOS:
source venv/bin/activate

# Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and specify optional keys (e.g., Google Gemini for query parsing):

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Setup Data Assets

Ensure the datasets obtained from team storage are placed in the `Data/` directory:
* Vector Database: `Data/aic_lancedb/`
* Image Previews: `Data/Custom_Keyframes/`

### 4. Verify System Health

Run the diagnostic script to verify hardware and data integrity:

```bash
python verify_setup.py
```

### 5. Launch the Web Application (1-Click)

* **Windows**: Double-click `run_app.bat` or run:
  ```powershell
  .\run_app.bat
  ```
* **Linux / macOS**:
  ```bash
  bash run_app.sh
  ```

Open your browser at **`http://127.0.0.1:8000`**.

---

## 🖥️ Key Features

### 1. Multi-View Image Queries
Users can paste multiple angles of the same target object simultaneously (`Ctrl+V` or file picker). The engine aggregates image feature vectors through hyperspherical normalization for view-invariant retrieval.

### 2. Client-Side Real-Time Filter & Revert
Instant search filtering on Top-K results with preservation of the original query rank (`#Rank`), allowing instantaneous reset without re-querying the database.

### 3. Rapid Video Exclusion (`🚫 Exclude Video`)
One-click exclusion of all results originating from a specific video ID, visually dimming excluded items and synchronizing with exclusion tags.

### 4. Interactive Bounding Box Filter
Draw rectangular regions directly onto keyframe images in the Lightbox viewer to isolate objects and query localized regions.

---

## 👥 Team & Acknowledgements

Developed for **AIC 2026 (Ho Chi Minh City AI Challenge)**.  
Architectural inspirations from:
* *Qwen3-VL-Embedding* (arXiv:2601.04720)
* *PraK V4 at Video Browser Showdown 2026* (MMM 2026)
* *LanceDB Embedded Vector Architecture*

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE). Confidential competition video data and proprietary vector indices are managed under private licenses.
