const API_URL = "http://localhost:8000/api/search";
const IMAGE_SEARCH_URL = "http://localhost:8000/api/search-image";
const PARSE_URL = "http://localhost:8000/api/parse-query";
const COCO_URL = "http://localhost:8000/api/coco-classes";
const TIMELINE_URL = "http://localhost:8000/api/video-timeline";
const RERANK_URL = "http://localhost:8000/api/rerank-top-videos";
const MULTI_SCENE_URL = "http://localhost:8000/api/search-multi-scene";
const QWEN_RERANK_CHAINS_URL = "http://localhost:8000/api/qwen-rerank-chains";
const QWEN_STATUS_URL = "http://localhost:8000/api/qwen-status";
const SUBMISSION_FILL_URL = "http://localhost:8000/api/submission/fill";
const SUBMISSION_SAVE_URL = "http://localhost:8000/api/submission/save-file";
const SUBMISSION_LIST_URL = "http://localhost:8000/api/submission/list-files";
const SUBMISSION_ZIP_URL = "http://localhost:8000/api/submission/export-zip";

// Image Query Elements
const cardImageQuery = document.getElementById("card-image-query");
const imageDropzone = document.getElementById("image-dropzone");
const imageDropzonePlaceholder = document.getElementById("image-dropzone-placeholder");
const imageFileInput = document.getElementById("image-file-input");
const imagePreviewContainer = document.getElementById("image-preview-container");
const imagePreviewImg = document.getElementById("image-preview-img");
const btnRemoveImageQuery = document.getElementById("btn-remove-image-query");
const btnSearchImageQuery = document.getElementById("btn-search-image-query");
let currentImageQueryBase64 = null;

// Submission Manager Elements
const cardSubmission = document.getElementById("card-submission");
const subDirPathInput = document.getElementById("sub-dir-path");
const btnSubRefreshDir = document.getElementById("btn-sub-refresh-dir");
const subFileListContainer = document.getElementById("sub-file-list-container");
const subFileCountSpan = document.getElementById("sub-file-count");
const subFileItemsContainer = document.getElementById("sub-file-items");

const subModePills = document.getElementById("sub-mode-pills");
let currentSubMode = "kis";

const subVideoIdInput = document.getElementById("sub-video-id");
const subKisQaFields = document.getElementById("sub-kis-qa-fields");
const subFrameAInput = document.getElementById("sub-frame-a");
const subFrameBInput = document.getElementById("sub-frame-b");
const subFrameA2Input = document.getElementById("sub-frame-a2");
const subFrameB2Input = document.getElementById("sub-frame-b2");

const subQaFields = document.getElementById("sub-qa-fields");
const subQaAnswerInput = document.getElementById("sub-qa-answer");

const subTrakeFields = document.getElementById("sub-trake-fields");
const subTrakeFramesInput = document.getElementById("sub-trake-frames");

const subEnableHedgingCheck = document.getElementById("sub-enable-hedging");
const subFileNameInput = document.getElementById("sub-file-name");
const btnSubFillPreview = document.getElementById("btn-sub-fill-preview");
const btnSubSaveFile = document.getElementById("btn-sub-save-file");

const subPreviewContainer = document.getElementById("sub-preview-container");
const subPreviewTotalSpan = document.getElementById("sub-preview-total");
const subPreviewStatusSpan = document.getElementById("sub-preview-status");
const subPreviewCode = document.getElementById("sub-preview-code");

const subZipOutputPathInput = document.getElementById("sub-zip-output-path");
const btnSubExportZip = document.getElementById("btn-sub-export-zip");
const lightboxSubBtn = document.getElementById("lightbox-sub-btn");

let latestGeneratedSubmissionLines = [];

// Mode Switcher Elements
const modeKisBtn = document.getElementById("mode-kis-btn");
const modeTrakeBtn = document.getElementById("mode-trake-btn");
const kisActionDock = document.getElementById("kis-action-dock");
const trakeActionDock = document.getElementById("trake-action-dock");

// ASR Context & Guidance Elements
const asrWindowPills = document.getElementById("asr-window-pills");
let selectedAsrWindowN = 1;

const geminiRerankGuidance = document.getElementById("gemini-rerank-guidance");
const headerRerankGuidance = document.getElementById("header-rerank-guidance");
const bodyRerankGuidance = document.getElementById("body-rerank-guidance");
const chevronRerankGuidance = document.getElementById("chevron-rerank-guidance");
const discriminatingTagsContainer = document.getElementById("discriminating-tags-container");
const addDiscFeatureInput = document.getElementById("add-disc-feature-input");
const btnAddDiscFeature = document.getElementById("btn-add-disc-feature");
const rerankHintInput = document.getElementById("rerank-hint-input");
let currentDiscriminatingFeatures = [];

// TRAKE Qwen Rerank Toolbar Elements
const trakeQwenToolbar = document.getElementById("trake-qwen-toolbar");
const trakeQwenRerankBtn = document.getElementById("trake-qwen-rerank-btn");
const qwenTopkPills = document.getElementById("qwen-topk-pills");
const qwenTopkChainsInput = document.getElementById("qwen-topk-chains-input");
const qwenProgressText = document.getElementById("qwen-progress-text");
const qwenStatusBadge = document.getElementById("qwen-status-badge");

// TRAKE Parameter Elements
const cardTrakeParams = document.getElementById("card-trake-params");
const trakeCandidatePoolSlider = document.getElementById("trake-candidate-pool-slider");
const trakeCandidatePoolInput = document.getElementById("trake-candidate-pool");
const trakeTopVInput = document.getElementById("trake-top-v-input");
const trakeFramesPerVidInput = document.getElementById("trake-frames-per-vid");
const trakeLambdaInput = document.getElementById("trake-lambda-input");
const trakeKpathsInput = document.getElementById("trake-kpaths-input");
const trakeTopKInput = document.getElementById("trake-top-k-input");
const trakeSingleSearchBtn = document.getElementById("trake-single-search-btn");
const trakeSearchBtn = document.getElementById("trake-search-btn");
const trakeExportBtn = document.getElementById("trake-export-btn");
const resultsMainTitle = document.getElementById("results-main-title");
const multiSceneBadgeTitle = document.getElementById("multi-scene-badge-title");

// Elements - Parser & Multi-Scene
const rawQueryInput = document.getElementById("raw-query-input");
const aiParseBtn = document.getElementById("ai-parse-btn");
const geminiApiKeyInput = document.getElementById("gemini-api-key");
const qaCallout = document.getElementById("qa-callout");
const qaQuestionText = document.getElementById("qa-question-text");
const multiSceneBar = document.getElementById("multi-scene-bar");
const multiSceneSummary = document.getElementById("multi-scene-summary");
const multiSceneTabs = document.getElementById("multi-scene-tabs");
const multiSceneEditorCards = document.getElementById("multi-scene-editor-cards");
const toggleSceneEditorBtn = document.getElementById("toggle-scene-editor-btn");

// Fusion Strategy (RRF)
const useRrfCheck = document.getElementById("use-rrf-check");
const rrfKInput = document.getElementById("rrf-k-input");
const rrfKGroup = document.getElementById("rrf-k-group");
const fusionModeDesc = document.getElementById("fusion-mode-desc");

// Module 1: Visual
const enableVisualCheck = document.getElementById("enable-visual-check");
const bodyVisual = document.getElementById("body-visual");
const searchInput = document.getElementById("search-input");
const weightVisualSlider = document.getElementById("weight-visual");
const valWeightVisual = document.getElementById("val-weight-visual");

// Module 2: ASR
const enableAsrCheck = document.getElementById("enable-asr-check");
const bodyAsr = document.getElementById("body-asr");
const speechInput = document.getElementById("speech-input");
const weightAsrSlider = document.getElementById("weight-asr");
const valWeightAsr = document.getElementById("val-weight-asr");

// Module 3: OCR
const enableOcrCheck = document.getElementById("enable-ocr-check");
const bodyOcr = document.getElementById("body-ocr");
const ocrInput = document.getElementById("ocr-input");
const weightOcrSlider = document.getElementById("weight-ocr");
const valWeightOcr = document.getElementById("val-weight-ocr");

// Module 4: RT-DETR
const enableObjectCheck = document.getElementById("enable-object-check");
const bodyObject = document.getElementById("body-object");
const objectClassSelect = document.getElementById("object-class-select");
const objectConfSlider = document.getElementById("object-conf-slider");
const valObjectConf = document.getElementById("val-object-conf");
const weightObjectSlider = document.getElementById("weight-object");
const valWeightObject = document.getElementById("val-weight-object");

// Module 5: Smart Dedup
const smartDedupCheck = document.getElementById("smart-dedup-check");
const bodyDedup = document.getElementById("body-dedup");
const visualSimSlider = document.getElementById("visual-sim-slider");
const visualSimNum = document.getElementById("visual-sim-num");
const uniqueVideoCheck = document.getElementById("unique-video-check");

// Module 6: Candidates & Rerank Filter
const topKSlider = document.getElementById("top-k-slider");
const topKNum = document.getElementById("top-k-num");
const rerankTopV = document.getElementById("rerank-top-v");
const rerankKPerVideo = document.getElementById("rerank-k-per-video");
const labelK2Total = document.getElementById("label-k2-total");
const btnTopVLabel = document.getElementById("btn-top-v-label");
const btnK2Label = document.getElementById("btn-k2-label");
const videoIdInput = document.getElementById("video-id");
const filterVideoResultsInput = document.getElementById("filter-video-results");

// Module 6A: Danh Mục & Tags Filter Elements (Database Pushdown)
const CATEGORIES_URL = "http://localhost:8000/api/categories";
const categorySelect = document.getElementById("category-select");
const groupSubcategory = document.getElementById("group-subcategory");
const subcategorySelect = document.getElementById("subcategory-select");
const includeTagsContainer = document.getElementById("include-tags-container");
const includeVideoInput = document.getElementById("include-video-input");
const excludeTagsContainer = document.getElementById("exclude-tags-container");
const excludeVideoInput = document.getElementById("exclude-video-input");

let currentIncludeTags = [];
let currentExcludeTags = [];
let categoriesTaxonomy = {};

// Cross-Video Dedup Elements (Chống QC/Intro lặp lại)
const enableCrossVideoDedupCheck = document.getElementById("enable-cross-video-dedup-check");
const crossVideoSimSlider = document.getElementById("cross-video-sim-slider");
const crossVideoSimNum = document.getElementById("cross-video-sim-num");

// Search & Rerank Action Buttons
const searchBtn = document.getElementById("search-btn");
const multiSearchBtn = document.getElementById("multi-search-btn");
const rerankBtn = document.getElementById("rerank-btn");
const statusText = document.getElementById("status-text");
const resultsGrid = document.getElementById("results-grid");
const resultCount = document.getElementById("result-count");

// Lightbox Elements
const lightboxModal = document.getElementById("lightbox-modal");
const lightboxCloseBtn = document.getElementById("lightbox-close-btn");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCanvas = document.getElementById("lightbox-canvas");
const lightboxToggleBoxes = document.getElementById("lightbox-toggle-boxes");
const lightboxTitle = document.getElementById("lightbox-title");
const lightboxScoreBadge = document.getElementById("lightbox-score-badge");
const lightboxTimeBadge = document.getElementById("lightbox-time-badge");
const lightboxCopyBtn = document.getElementById("lightbox-copy-btn");
const lightboxTimelineBtn = document.getElementById("lightbox-timeline-btn");
const lightboxVideoBtn = document.getElementById("lightbox-video-btn");
const copyStatus = document.getElementById("copy-status");
const lightboxTranscript = document.getElementById("lightbox-transcript");
const lightboxBoxesList = document.getElementById("lightbox-boxes-list");

// Timeline Elements
const timelineModal = document.getElementById("timeline-modal");
const timelineVideoTitle = document.getElementById("timeline-video-title");
const timelineFrameCount = document.getElementById("timeline-frame-count");
const timelineCloseBtn = document.getElementById("timeline-close-btn");
const timelineStrip = document.getElementById("timeline-strip");

// Video Player Elements
const VIDEO_STREAM_URL = "http://localhost:8000/videos";
const videoModal = document.getElementById("video-modal");
const videoPlayerTitle = document.getElementById("video-player-title");
const videoTimeBadge = document.getElementById("video-time-badge");
const videoCloseBtn = document.getElementById("video-close-btn");
const html5VideoPlayer = document.getElementById("html5-video-player");
const videoRewind5s = document.getElementById("video-rewind-5s");
const videoJumpKeyframe = document.getElementById("video-jump-keyframe");
const videoForward5s = document.getElementById("video-forward-5s");
const videoSpeedSelect = document.getElementById("video-speed-select");
const videoCopyBtn = document.getElementById("video-copy-btn");

// Global state
let currentResults = [];
let currentLightboxIndex = 0;
let activeResultsSource = [];
let cocoClasses = [];
let latestRerankerQuery = "";
let latestUnifiedVisual = "";
let latestUnifiedSpeech = "";
let latestUnifiedOcr = "";
let latestParsedScenes = [];
let isMultiSceneMode = false;
let multiSceneResultsData = null;
let activeSceneTabIdx = -1;
let currentPlayingTimestamp = 0;
let currentPlayingVideoId = "";
let currentPlayingFrameId = "";

// TRAKE Mode States
let currentAppMode = "KIS"; // "KIS" | "TRAKE"
let latestTrakeSequences = [];
let latestTrakeSubmissionLines = [];

// ==========================================================================
// CHUYỂN ĐỔI CHẾ ĐỘ: KIS / QA MODE <--> TRAKE MODE
// ==========================================================================
function switchMode(mode, saveState = true) {
    currentAppMode = mode;

    if (mode === "TRAKE") {
        if (modeTrakeBtn) modeTrakeBtn.classList.add("active");
        if (modeKisBtn) modeKisBtn.classList.remove("active");

        if (kisActionDock) kisActionDock.style.display = "none";
        if (trakeActionDock) trakeActionDock.style.display = "flex";

        if (cardTrakeParams) {
            cardTrakeParams.style.display = "block";
            cardTrakeParams.classList.remove("is-collapsed");
            const body = document.getElementById("body-trake-params");
            if (body) body.style.display = "block";
            const chevron = cardTrakeParams.querySelector(".chevron-icon");
            if (chevron) chevron.innerText = "▲";
        }

        if (resultsMainTitle) {
            resultsMainTitle.innerText = "⏱️ TRAKE STORYBOARD MATRIX (TARS MONOTONIC DP)";
        }

        // Nếu đã có kết quả TRAKE trước đó -> Render lại Storyboard Matrix
        if (latestTrakeSequences && latestTrakeSequences.length > 0) {
            renderTrakeStoryboard(latestTrakeSequences);
        }

        showToast("⏱️ Đã chuyển sang chế độ TRAKE (Temporal Event Extraction)!", "info", 2000);
    } else {
        if (modeKisBtn) modeKisBtn.classList.add("active");
        if (modeTrakeBtn) modeTrakeBtn.classList.remove("active");

        if (kisActionDock) kisActionDock.style.display = "flex";
        if (trakeActionDock) trakeActionDock.style.display = "none";

        if (cardTrakeParams) {
            cardTrakeParams.style.display = "none";
        }

        if (resultsMainTitle) {
            resultsMainTitle.innerText = "MULTIMODAL RESULTS";
        }

        // Render lại kết quả KIS nếu có
        if (currentResults && currentResults.length > 0) {
            renderResults(currentResults);
        }

        showToast("🔍 Đã chuyển về chế độ KIS / QA thông thường.", "info", 1800);
    }

    if (saveState) {
        saveControlPanelState();
    }
}

if (modeKisBtn) {
    modeKisBtn.addEventListener("click", () => switchMode("KIS"));
}
if (modeTrakeBtn) {
    modeTrakeBtn.addEventListener("click", () => switchMode("TRAKE"));
}

// ==========================================================================
// 0. FLOATING TOAST NOTIFICATION SYSTEM
// ==========================================================================
let activeLoadingToast = null;

function showToast(message, type = "info", duration = 3500) {
    const container = document.getElementById("toast-container");
    if (!container) return null;

    if (type === "loading" && activeLoadingToast) {
        activeLoadingToast.remove();
        activeLoadingToast = null;
    }

    const toast = document.createElement("div");
    toast.className = `toast-item toast-${type}`;

    let icon = "ℹ️";
    if (type === "success") icon = "✅";
    else if (type === "error") icon = "❌";
    else if (type === "warning") icon = "⚠️";
    else if (type === "loading") icon = "⏳";

    toast.innerHTML = `<span style="font-size: 1.1rem; flex-shrink: 0;">${icon}</span><span>${message}</span>`;
    container.appendChild(toast);

    requestAnimationFrame(() => {
        toast.classList.add("show");
    });

    if (type === "loading") {
        activeLoadingToast = toast;
        return toast;
    }

    if (duration > 0) {
        setTimeout(() => {
            toast.classList.remove("show");
            setTimeout(() => {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 300);
        }, duration);
    }
    return toast;
}

function dismissLoadingToast() {
    if (activeLoadingToast) {
        activeLoadingToast.classList.remove("show");
        setTimeout(() => {
            if (activeLoadingToast && activeLoadingToast.parentNode) {
                activeLoadingToast.parentNode.removeChild(activeLoadingToast);
            }
            activeLoadingToast = null;
        }, 250);
    }
}

// ==========================================================================
// 1. COLLAPSE / EXPAND TOGGLE & RRF STATE HANDLERS
// ==========================================================================
function setCardCollapsed(cardEl, collapse) {
    if (!cardEl) return;
    if (collapse) {
        cardEl.classList.add("is-collapsed");
    } else {
        cardEl.classList.remove("is-collapsed");
    }
    const chevron = cardEl.querySelector(".chevron-icon");
    if (chevron) {
        chevron.innerText = collapse ? "▼" : "▲";
    }
}

function toggleCardCollapse(cardEl) {
    if (!cardEl) return;
    const isNowCollapsed = !cardEl.classList.contains("is-collapsed");
    setCardCollapsed(cardEl, isNowCollapsed);
    saveControlPanelState();
}

// Bắt sự kiện click vào header card
document.querySelectorAll(".feature-card").forEach(card => {
    const header = card.querySelector(".card-header");
    if (header) {
        header.addEventListener("click", (e) => {
            if (e.target.closest(".switch-small") || e.target.closest("input[type='checkbox']")) {
                return;
            }
            toggleCardCollapse(card);
        });
    }
});

// Đóng / Xổ Khung Đặc Trưng Phân Biệt (Qwen Guide)
function setGuidanceCollapsed(collapse) {
    if (!geminiRerankGuidance || !bodyRerankGuidance) return;
    if (collapse) {
        geminiRerankGuidance.classList.add("is-collapsed");
        bodyRerankGuidance.style.display = "none";
        if (chevronRerankGuidance) chevronRerankGuidance.innerText = "▼";
    } else {
        geminiRerankGuidance.classList.remove("is-collapsed");
        bodyRerankGuidance.style.display = "block";
        if (chevronRerankGuidance) chevronRerankGuidance.innerText = "▲";
    }
}

function toggleGuidanceCollapse() {
    if (!geminiRerankGuidance) return;
    const isNowCollapsed = !geminiRerankGuidance.classList.contains("is-collapsed");
    setGuidanceCollapsed(isNowCollapsed);
    saveControlPanelState();
}

if (headerRerankGuidance) {
    headerRerankGuidance.addEventListener("click", toggleGuidanceCollapse);
}

// Logic RRF Mode UI Toggle
function updateRrfUIState() {
    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const kVal = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    if (rrfKGroup) {
        if (isRrf) {
            rrfKGroup.classList.remove("disabled-k");
            if (rrfKInput) rrfKInput.disabled = false;
        } else {
            rrfKGroup.classList.add("disabled-k");
            if (rrfKInput) rrfKInput.disabled = true;
        }
    }

    // Vô hiệu hóa triệt để tất cả slider trọng số khi bật RRF
    const sliders = [weightVisualSlider, weightOcrSlider, weightAsrSlider, weightObjectSlider];
    sliders.forEach(slider => {
        if (slider) slider.disabled = isRrf;
    });

    const weightRows = document.querySelectorAll(".weight-row");
    weightRows.forEach(row => {
        if (isRrf) {
            row.classList.add("disabled-weights");
        } else {
            row.classList.remove("disabled-weights");
        }
    });
}

if (useRrfCheck) {
    useRrfCheck.addEventListener("change", () => {
        updateRrfUIState();
        saveControlPanelState();
        showToast(useRrfCheck.checked ? `⚡ Đã bật RRF (k=${rrfKInput.value || 60}) - Trọng số đã khóa` : "⚖️ Đã chuyển sang Weighted Fusion - Cho phép kéo trọng số", "info", 2200);
    });
}

if (rrfKInput) {
    rrfKInput.addEventListener("input", () => {
        saveControlPanelState();
    });
}

// Bắt sự kiện Checkbox tính năng -> Tự động thu / xổ card tương ứng
function bindAutoCollapse(checkEl, cardId) {
    if (!checkEl) return;
    checkEl.addEventListener("change", () => {
        const card = document.getElementById(cardId);
        if (card) {
            setCardCollapsed(card, !checkEl.checked);
            saveControlPanelState();
        }
    });
}

bindAutoCollapse(enableVisualCheck, "card-visual");
bindAutoCollapse(enableAsrCheck, "card-asr");
bindAutoCollapse(enableOcrCheck, "card-ocr");
bindAutoCollapse(enableObjectCheck, "card-object");
bindAutoCollapse(smartDedupCheck, "card-dedup");

// ==========================================================================
// 2. STATE PERSISTENCE (LOCAL STORAGE - BAO GỒM TOÀN BỘ THAM SỐ TARS TRAKE)
// ==========================================================================
const STORAGE_KEY = "aic_2026_control_panel_state_v3";

function saveControlPanelState() {
    try {
        const state = {
            app_mode: currentAppMode,
            use_rrf: useRrfCheck ? useRrfCheck.checked : true,
            rrf_k: rrfKInput ? parseInt(rrfKInput.value) || 60 : 60,
            enable_visual: enableVisualCheck ? enableVisualCheck.checked : true,
            enable_ocr: enableOcrCheck ? enableOcrCheck.checked : false,
            enable_asr: enableAsrCheck ? enableAsrCheck.checked : true,
            enable_object: enableObjectCheck ? enableObjectCheck.checked : false,
            object_class: objectClassSelect ? objectClassSelect.value : "0",
            object_conf: objectConfSlider ? objectConfSlider.value : "0.30",
            weight_visual: weightVisualSlider ? weightVisualSlider.value : "1.0",
            weight_ocr: weightOcrSlider ? weightOcrSlider.value : "1.0",
            weight_asr: weightAsrSlider ? weightAsrSlider.value : "1.0",
            weight_object: weightObjectSlider ? weightObjectSlider.value : "0.5",
            smart_dedup: smartDedupCheck ? smartDedupCheck.checked : true,
            smart_submode: (document.querySelector('input[name="smart-submode"]:checked') || {}).value || "smart",
            visual_sim: visualSimSlider ? visualSimSlider.value : "0.90",
            unique_video: uniqueVideoCheck ? uniqueVideoCheck.checked : false,
            enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
            cross_video_sim: crossVideoSimSlider ? crossVideoSimSlider.value : "0.90",
            asr_window_n: selectedAsrWindowN,
            top_k: topKNum ? topKNum.value : "100",
            top_v: rerankTopV ? rerankTopV.value : "5",
            k_per_video: rerankKPerVideo ? rerankKPerVideo.value : "40",

            // Tham số TARS DP (TRAKE Mode)
            trake_candidate_pool: trakeCandidatePoolInput ? trakeCandidatePoolInput.value : "100",
            trake_top_v: trakeTopVInput ? trakeTopVInput.value : "10",
            trake_frames_per_vid: trakeFramesPerVidInput ? trakeFramesPerVidInput.value : "40",
            trake_lambda: trakeLambdaInput ? trakeLambdaInput.value : "0.001",
            trake_kpaths: trakeKpathsInput ? trakeKpathsInput.value : "3",
            trake_top_k: trakeTopKInput ? trakeTopKInput.value : "10",

            // Trạng thái thu gọn (Collapsed States)
            collapsed_gemini: document.getElementById("card-gemini")?.classList.contains("is-collapsed"),
            collapsed_image_query: document.getElementById("card-image-query")?.classList.contains("is-collapsed"),
            collapsed_rerank_guidance: geminiRerankGuidance?.classList.contains("is-collapsed"),
            collapsed_visual: document.getElementById("card-visual")?.classList.contains("is-collapsed"),
            collapsed_asr: document.getElementById("card-asr")?.classList.contains("is-collapsed"),
            collapsed_ocr: document.getElementById("card-ocr")?.classList.contains("is-collapsed"),
            collapsed_object: document.getElementById("card-object")?.classList.contains("is-collapsed"),
            collapsed_dedup: document.getElementById("card-dedup")?.classList.contains("is-collapsed"),
            collapsed_filter: document.getElementById("card-filter")?.classList.contains("is-collapsed"),
            collapsed_trake_params: document.getElementById("card-trake-params")?.classList.contains("is-collapsed"),
            collapsed_submission: document.getElementById("card-submission")?.classList.contains("is-collapsed"),
            sub_dir_path: subDirPathInput ? subDirPathInput.value : "query/query-p1/submission",
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
        console.warn("Could not save to localStorage", e);
    }
}

function restoreControlPanelState() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            // Mặc định ban đầu: Gemini, Visual và ASR MỞ; OCR, Object, Dedup, Filter, Submission THU GỌN
            setCardCollapsed(document.getElementById("card-gemini"), false);
            setCardCollapsed(document.getElementById("card-visual"), false);
            setCardCollapsed(document.getElementById("card-asr"), false);
            setCardCollapsed(document.getElementById("card-ocr"), true);
            setCardCollapsed(document.getElementById("card-object"), true);
            setCardCollapsed(document.getElementById("card-dedup"), true);
            setCardCollapsed(document.getElementById("card-filter"), true);
            setCardCollapsed(document.getElementById("card-submission"), true);
            updateRrfUIState();
            return;
        }
        const state = JSON.parse(raw);
        if (state.use_rrf !== undefined && useRrfCheck) useRrfCheck.checked = state.use_rrf;
        if (state.rrf_k !== undefined && rrfKInput) rrfKInput.value = state.rrf_k;

        if (state.enable_visual !== undefined && enableVisualCheck) enableVisualCheck.checked = state.enable_visual;
        if (state.enable_ocr !== undefined && enableOcrCheck) enableOcrCheck.checked = state.enable_ocr;
        if (state.enable_asr !== undefined && enableAsrCheck) enableAsrCheck.checked = state.enable_asr;
        if (state.enable_object !== undefined && enableObjectCheck) enableObjectCheck.checked = state.enable_object;

        if (state.asr_window_n !== undefined && asrWindowPills) {
            selectedAsrWindowN = parseInt(state.asr_window_n) || 1;
            asrWindowPills.querySelectorAll(".btn-pill").forEach(btn => {
                if (parseInt(btn.getAttribute("data-n")) === selectedAsrWindowN) {
                    btn.classList.add("active");
                } else {
                    btn.classList.remove("active");
                }
            });
        }

        if (state.object_class !== undefined && objectClassSelect) objectClassSelect.value = state.object_class;
        if (state.object_conf !== undefined && objectConfSlider) {
            objectConfSlider.value = state.object_conf;
            if (valObjectConf) valObjectConf.innerText = Number(state.object_conf).toFixed(2);
        }

        if (state.weight_visual !== undefined && weightVisualSlider) {
            weightVisualSlider.value = state.weight_visual;
            if (valWeightVisual) valWeightVisual.innerText = Number(state.weight_visual).toFixed(1);
        }
        if (state.weight_ocr !== undefined && weightOcrSlider) {
            weightOcrSlider.value = state.weight_ocr;
            if (valWeightOcr) valWeightOcr.innerText = Number(state.weight_ocr).toFixed(1);
        }
        if (state.weight_asr !== undefined && weightAsrSlider) {
            weightAsrSlider.value = state.weight_asr;
            if (valWeightAsr) valWeightAsr.innerText = Number(state.weight_asr).toFixed(1);
        }
        if (state.weight_object !== undefined && weightObjectSlider) {
            weightObjectSlider.value = state.weight_object;
            if (valWeightObject) valWeightObject.innerText = Number(state.weight_object).toFixed(1);
        }

        if (state.smart_dedup !== undefined && smartDedupCheck) smartDedupCheck.checked = state.smart_dedup;
        if (state.smart_submode) {
            const rad = document.querySelector(`input[name="smart-submode"][value="${state.smart_submode}"]`);
            if (rad) rad.checked = true;
        }
        if (state.visual_sim !== undefined && visualSimSlider) {
            visualSimSlider.value = state.visual_sim;
            if (visualSimNum) visualSimNum.value = state.visual_sim;
        }
        if (state.unique_video !== undefined && uniqueVideoCheck) uniqueVideoCheck.checked = state.unique_video;

        if (state.enable_cross_video_dedup !== undefined && enableCrossVideoDedupCheck) enableCrossVideoDedupCheck.checked = state.enable_cross_video_dedup;
        if (state.cross_video_sim !== undefined && crossVideoSimSlider) {
            crossVideoSimSlider.value = state.cross_video_sim;
            if (crossVideoSimNum) crossVideoSimNum.value = state.cross_video_sim;
        }

        if (state.top_k !== undefined && topKNum) {
            topKNum.value = state.top_k;
            if (topKSlider) topKSlider.value = state.top_k;
        }
        if (state.top_v !== undefined && rerankTopV) rerankTopV.value = state.top_v;
        if (state.k_per_video !== undefined && rerankKPerVideo) rerankKPerVideo.value = state.k_per_video;

        // Khôi phục Tham số TARS DP (TRAKE)
        if (state.trake_candidate_pool !== undefined && trakeCandidatePoolInput) {
            trakeCandidatePoolInput.value = state.trake_candidate_pool;
            if (trakeCandidatePoolSlider) trakeCandidatePoolSlider.value = state.trake_candidate_pool;
        }
        if (state.trake_top_v !== undefined && trakeTopVInput) trakeTopVInput.value = state.trake_top_v;
        if (state.trake_frames_per_vid !== undefined && trakeFramesPerVidInput) trakeFramesPerVidInput.value = state.trake_frames_per_vid;
        if (state.trake_lambda !== undefined && trakeLambdaInput) trakeLambdaInput.value = state.trake_lambda;
        if (state.trake_kpaths !== undefined && trakeKpathsInput) trakeKpathsInput.value = state.trake_kpaths;
        if (state.trake_top_k !== undefined && trakeTopKInput) trakeTopKInput.value = state.trake_top_k;

        if (state.sub_dir_path && subDirPathInput) {
            subDirPathInput.value = state.sub_dir_path;
        }

        // Khôi phục Chế độ KIS / TRAKE (không lưu đè state khi đang restore)
        if (state.app_mode && state.app_mode !== currentAppMode) {
            switchMode(state.app_mode, false);
        }

        // Khôi phục collapsed state (Gemini luôn mở trừ khi user cố tình thu)
        setCardCollapsed(document.getElementById("card-gemini"), state.collapsed_gemini !== undefined ? state.collapsed_gemini : false);
        setCardCollapsed(document.getElementById("card-visual"), state.collapsed_visual !== undefined ? state.collapsed_visual : !enableVisualCheck.checked);
        setCardCollapsed(document.getElementById("card-asr"), state.collapsed_asr !== undefined ? state.collapsed_asr : !enableAsrCheck.checked);
        setCardCollapsed(document.getElementById("card-ocr"), state.collapsed_ocr !== undefined ? state.collapsed_ocr : !enableOcrCheck.checked);
        setCardCollapsed(document.getElementById("card-object"), state.collapsed_object !== undefined ? state.collapsed_object : !enableObjectCheck.checked);
        setCardCollapsed(document.getElementById("card-dedup"), state.collapsed_dedup !== undefined ? state.collapsed_dedup : !smartDedupCheck.checked);
        setCardCollapsed(document.getElementById("card-filter"), state.collapsed_filter !== undefined ? state.collapsed_filter : true);
        if (cardTrakeParams && state.collapsed_trake_params !== undefined) {
            setCardCollapsed(cardTrakeParams, state.collapsed_trake_params);
        }
        setCardCollapsed(document.getElementById("card-submission"), state.collapsed_submission !== undefined ? state.collapsed_submission : true);

        if (state.collapsed_rerank_guidance !== undefined && geminiRerankGuidance) {
            setRerankGuidanceCollapsed(state.collapsed_rerank_guidance);
        }

        updateRrfUIState();
        updateK2Calculation();
    } catch (e) {
        console.warn("Could not restore localStorage", e);
    }
}

// Sliders and Values Sync
if (objectConfSlider && valObjectConf) {
    objectConfSlider.addEventListener("input", (e) => {
        valObjectConf.innerText = parseFloat(e.target.value).toFixed(2);
        saveControlPanelState();
    });
}

if (weightObjectSlider && valWeightObject) {
    weightObjectSlider.addEventListener("input", (e) => {
        valWeightObject.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

if (weightVisualSlider && valWeightVisual) {
    weightVisualSlider.addEventListener("input", (e) => {
        valWeightVisual.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

if (weightOcrSlider && valWeightOcr) {
    weightOcrSlider.addEventListener("input", (e) => {
        valWeightOcr.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

if (weightAsrSlider && valWeightAsr) {
    weightAsrSlider.addEventListener("input", (e) => {
        valWeightAsr.innerText = parseFloat(e.target.value).toFixed(1);
        saveControlPanelState();
    });
}

// Đồng bộ Visual Sim Slider & Number Input
if (visualSimSlider && visualSimNum) {
    visualSimSlider.addEventListener("input", (e) => {
        visualSimNum.value = parseFloat(e.target.value).toFixed(3);
        saveControlPanelState();
    });

    visualSimNum.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value) || 0.90;
        visualSimSlider.value = val;
        saveControlPanelState();
    });
}

// Đồng bộ Cross-Video Dedup Slider & Number Input
if (crossVideoSimSlider && crossVideoSimNum) {
    crossVideoSimSlider.addEventListener("input", (e) => {
        crossVideoSimNum.value = parseFloat(e.target.value).toFixed(3);
        saveControlPanelState();
    });

    crossVideoSimNum.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value) || 0.90;
        crossVideoSimSlider.value = val;
        saveControlPanelState();
    });
}

// Đồng bộ TRAKE Candidate Pool Slider & Number Input & Tự động lưu
if (trakeCandidatePoolSlider && trakeCandidatePoolInput) {
    trakeCandidatePoolSlider.addEventListener("input", (e) => {
        trakeCandidatePoolInput.value = parseInt(e.target.value, 10);
        saveControlPanelState();
    });

    trakeCandidatePoolInput.addEventListener("input", (e) => {
        const val = parseInt(e.target.value, 10) || 100;
        trakeCandidatePoolSlider.value = val;
        saveControlPanelState();
    });
}

// Gắn auto-save cho tất cả các input tham số TRAKE TARS DP
[trakeTopVInput, trakeFramesPerVidInput, trakeLambdaInput, trakeKpathsInput, trakeTopKInput].forEach(inp => {
    if (inp) {
        inp.addEventListener("input", saveControlPanelState);
        inp.addEventListener("change", saveControlPanelState);
    }
});

if (enableCrossVideoDedupCheck) {
    enableCrossVideoDedupCheck.addEventListener("change", () => {
        saveControlPanelState();
    });
}

// Đồng bộ Candidate Slider và Ô Nhập Số
if (topKSlider && topKNum) {
    topKSlider.addEventListener("input", (e) => {
        topKNum.value = e.target.value;
    });

    topKNum.addEventListener("input", (e) => {
        const val = parseInt(e.target.value) || 1;
        if (val <= 100) {
            topKSlider.value = val;
        } else {
            topKSlider.value = 100;
        }
    });
}

// Kẹp điều kiện giữa Smart Dedup và Unique Video
if (smartDedupCheck && uniqueVideoCheck) {
    uniqueVideoCheck.addEventListener("change", (e) => {
        if (e.target.checked) {
            smartDedupCheck.checked = false;
            bodyDedup.classList.add("disabled-body");
        }
    });

    smartDedupCheck.addEventListener("change", (e) => {
        if (e.target.checked) {
            uniqueVideoCheck.checked = false;
        }
    });
}

// Load & Save Gemini API Key vào localStorage
const DEFAULT_GEMINI_KEY = "REMOVED_REVOKED_CREDENTIAL";
if (geminiApiKeyInput) {
    const savedKey = localStorage.getItem("gemini_api_key");
    geminiApiKeyInput.value = savedKey || DEFAULT_GEMINI_KEY;

    geminiApiKeyInput.addEventListener("input", (e) => {
        localStorage.setItem("gemini_api_key", e.target.value.trim());
    });
}

// ==========================================================================
// ASR EXPANDED CONTEXT PILLS (±N SEGMENTS)
// ==========================================================================
if (asrWindowPills) {
    asrWindowPills.querySelectorAll(".btn-pill").forEach((btn) => {
        btn.addEventListener("click", () => {
            asrWindowPills.querySelectorAll(".btn-pill").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            selectedAsrWindowN = parseInt(btn.getAttribute("data-n")) || 0;
            saveControlPanelState();
            showToast(`🎙️ Đã chọn ASR Ngữ Cảnh mở rộng: ±${selectedAsrWindowN} cụm thoại liền kề`, "info", 1500);
        });
    });
}

// ==========================================================================
// DISCRIMINATING FEATURES TAGS HANDLER (QWEN GUIDE)
// ==========================================================================
function renderDiscriminatingTags() {
    if (!discriminatingTagsContainer) return;
    discriminatingTagsContainer.innerHTML = "";
    currentDiscriminatingFeatures.forEach((feat, idx) => {
        const tag = document.createElement("span");
        tag.className = "discriminating-tag";
        tag.innerHTML = `<span>${feat}</span><button type="button" class="btn-remove-tag" data-idx="${idx}" title="Xóa tag">&times;</button>`;
        tag.querySelector(".btn-remove-tag").addEventListener("click", () => {
            currentDiscriminatingFeatures.splice(idx, 1);
            renderDiscriminatingTags();
        });
        discriminatingTagsContainer.appendChild(tag);
    });
}

if (btnAddDiscFeature && addDiscFeatureInput) {
    const handleAddTag = () => {
        const val = addDiscFeatureInput.value.trim();
        if (val && !currentDiscriminatingFeatures.includes(val)) {
            currentDiscriminatingFeatures.push(val);
            renderDiscriminatingTags();
            addDiscFeatureInput.value = "";
        }
    };
    btnAddDiscFeature.addEventListener("click", handleAddTag);
    addDiscFeatureInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            handleAddTag();
        }
    });
}

// ==========================================================================
// QWEN TOP K CHAINS PILLS HANDLER
// ==========================================================================
if (qwenTopkPills && qwenTopkChainsInput) {
    qwenTopkPills.querySelectorAll(".btn-pill").forEach((btn) => {
        btn.addEventListener("click", () => {
            qwenTopkPills.querySelectorAll(".btn-pill").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            const kVal = parseInt(btn.getAttribute("data-k")) || 5;
            qwenTopkChainsInput.value = kVal;
        });
    });

    qwenTopkChainsInput.addEventListener("input", (e) => {
        const val = parseInt(e.target.value) || 5;
        qwenTopkPills.querySelectorAll(".btn-pill").forEach((btn) => {
            if (parseInt(btn.getAttribute("data-k")) === val) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    });
}

// ==========================================================================
// GEMINI PROMPT PARSER (TÔN TRỌNG TRẠNG THÁI CHECKBOX CỦA NGƯỜI DÙNG)
// ==========================================================================
async function parseWithGemini() {
    const rawText = rawQueryInput ? rawQueryInput.value.trim() : "";
    if (!rawText) {
        statusText.innerText = "⚠️ Vui lòng dán đề bài thô vào ô trên trước.";
        showToast("⚠️ Vui lòng dán đề bài thô vào ô trên trước.", "warning");
        return;
    }

    const apiKey = geminiApiKeyInput ? geminiApiKeyInput.value.trim() : "";

    aiParseBtn.disabled = true;
    aiParseBtn.innerText = "⏳ ĐANG PHÂN TÁCH...";
    statusText.innerText = "Đang gửi đề bài tới Gemini AI để bóc tách...";
    showToast("⏳ Đang gửi đề bài tới Gemini AI để bóc tách...", "loading", 0);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 8000);

    try {
        const response = await fetch(PARSE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ raw_text: rawText, api_key: apiKey }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (!response.ok) {
            throw new Error(`Parse Error: ${response.statusText}`);
        }

        const resData = await response.json();
        dismissLoadingToast();
        const data = resData.data;

        // Lưu lại câu reranker_query chi tiết và các câu thống nhất
        latestRerankerQuery = data.reranker_query || rawText;
        latestUnifiedVisual = data.visual_query || "";
        latestUnifiedSpeech = data.speech_query || "";
        latestUnifiedOcr = data.ocr_query || "";

        // Điền tự động nội dung chữ vào các ô tìm kiếm
        if (searchInput && data.visual_query) searchInput.value = data.visual_query;
        if (ocrInput) ocrInput.value = data.ocr_query || "";
        if (speechInput) speechInput.value = data.speech_query || "";

        // Nạp và hiển thị các đặc trưng phân biệt & gợi ý AI Rerank
        currentDiscriminatingFeatures = Array.isArray(data.discriminating_features) ? [...data.discriminating_features] : [];
        renderDiscriminatingTags();
        if (rerankHintInput) {
            rerankHintInput.value = data.rerank_hint || "";
        }
        if (geminiRerankGuidance) {
            geminiRerankGuidance.style.display = (data.task_type === "TRAKE" || data.is_multi_scene || currentDiscriminatingFeatures.length > 0) ? "block" : "none";
        }

        // Điền gợi ý lớp vật thể RT-DETR (KHÔNG tự động bật công tắc)
        if (data.object_filter && data.object_filter.class_id !== undefined && data.object_filter.class_id >= 0) {
            if (objectClassSelect) {
                objectClassSelect.value = data.object_filter.class_id;
            }
            if (data.object_filter.min_confidence && objectConfSlider && valObjectConf) {
                objectConfSlider.value = data.object_filter.min_confidence;
                valObjectConf.innerText = parseFloat(data.object_filter.min_confidence).toFixed(2);
            }
        }

        // Hiển thị banner nhắc nhở câu hỏi QA (nếu có)
        if (qaCallout && qaQuestionText) {
            if (data.task_type === "QA" && data.qa_question) {
                qaQuestionText.innerText = data.qa_question;
                qaCallout.style.display = "flex";
            } else {
                qaCallout.style.display = "none";
            }
        }

        const parserSrc = data.source || "Gemini AI";

        // Xử lý bài toán TRAKE (Temporal Reasoning & Keyframe Extraction)
        if (data.task_type === "TRAKE" && data.scenes && data.scenes.length >= 2) {
            isMultiSceneMode = true;
            latestParsedScenes = data.scenes;
            renderMultiSceneTabs(data.scenes);
            if (multiSceneBar) multiSceneBar.style.display = "flex";
            if (multiSceneBadgeTitle) multiSceneBadgeTitle.innerText = `⏱️ TRAKE (${data.scenes.length} EVENTS)`;
            if (multiSceneSummary) {
                multiSceneSummary.innerText = `Chuỗi sự kiện: ${data.scenes.map(s => s.name || `E${s.scene_idx}`).join(" ➔ ")}`;
            }

            // Tự động chuyển sang TRAKE Mode
            switchMode("TRAKE");
            statusText.innerText = `⏱️ [TRAKE MODE - ${data.scenes.length} SỰ KIỆN - ${parserSrc}] Đã nạp bối cảnh và chuỗi E1..E${data.scenes.length}. Bấm '⏱️ SEARCH TRAKE' để chạy TARS Monotonic DP.`;
            showToast(`⏱️ Đã phát hiện bài toán TRAKE (${data.scenes.length} sự kiện tuần tự) -> Tự động chuyển sang TRAKE Mode!`, "success", 3000);

        } else if (data.is_multi_scene && data.scenes && data.scenes.length >= 2) {
            // Xử lý chế độ Đa Phân Cảnh KIS thông thường
            if (currentAppMode === "TRAKE") switchMode("KIS");
            isMultiSceneMode = true;
            latestParsedScenes = data.scenes;
            renderMultiSceneTabs(data.scenes);
            if (multiSceneBar) multiSceneBar.style.display = "flex";
            if (multiSceneBadgeTitle) multiSceneBadgeTitle.innerText = "🎬 MULTI-SCENE";
            if (multiSceneSummary) {
                multiSceneSummary.innerText = `Phát hiện ${data.scenes.length} phân cảnh nối tiếp: ${data.scenes.map(s => s.name || `Cảnh ${s.scene_idx}`).join(" ➔ ")}`;
            }
            if (multiSearchBtn) {
                multiSearchBtn.disabled = false;
                multiSearchBtn.classList.add("active-multi-ready");
            }
            statusText.innerText = `✨ [ĐA PHÂN CẢNH ${data.scenes.length} ĐOẠN - ${parserSrc}] Nút 🎬 MULTI-SCENE đã mở khóa. Bấm '🚀 SEARCH NOW' (Đơn Cảnh) hoặc '🎬 MULTI-SCENE' (Giao Thoa).`;
            showToast(`✨ Đã mở khóa nút 🎬 MULTI-SCENE (${data.scenes.length} cảnh)!`, "success");
        } else {
            // Đơn cảnh thông thường
            if (currentAppMode === "TRAKE") switchMode("KIS");
            isMultiSceneMode = false;
            latestParsedScenes = [];
            if (multiSceneBar) multiSceneBar.style.display = "none";
            if (multiSearchBtn) {
                multiSearchBtn.disabled = true;
                multiSearchBtn.classList.remove("active-multi-ready");
            }
            statusText.innerText = `✨ [${data.task_type} - ${parserSrc}] Đã nạp nội dung tự nhiên. Bấm '🚀 SEARCH NOW'.`;
            showToast(`✨ Đã nạp nội dung phân tách (${parserSrc})!`, "success");
        }

    } catch (err) {
        clearTimeout(timeoutId);
        dismissLoadingToast();
        const errMsg = err.name === "AbortError" ? "Quá thời gian kết nối (Timeout)" : err.message;
        statusText.innerText = `[ERROR AI] ${errMsg}`;
        showToast(`Lỗi AI Parser: ${errMsg}`, "error");
        console.error(err);
    } finally {
        aiParseBtn.disabled = false;
        aiParseBtn.innerText = "✨ PHÂN TÁCH ĐỀ BÀI";
    }
}

if (aiParseBtn) {
    aiParseBtn.addEventListener("click", parseWithGemini);
}

// Toggle ẩn hiện Bảng Soạn Thảo Cảnh
if (toggleSceneEditorBtn && multiSceneEditorCards) {
    toggleSceneEditorBtn.addEventListener("click", () => {
        if (multiSceneEditorCards.style.display === "none") {
            multiSceneEditorCards.style.display = "grid";
            toggleSceneEditorBtn.innerText = "👁️ Ẩn Bảng Soạn Thảo";
        } else {
            multiSceneEditorCards.style.display = "none";
            toggleSceneEditorBtn.innerText = "✏️ Hiện Bảng Soạn Thảo";
        }
    });
}

// ==========================================================================
// RENDER THANH SEGMENTED PILLS ĐA PHÂN CẢNH (MULTI-SCENE SWITCHER)
// ==========================================================================
function renderMultiSceneTabs(scenes) {
    if (!multiSceneTabs) return;
    multiSceneTabs.innerHTML = "";

    // 1. Pill "Đầy Đủ (Đơn Cảnh)"
    const allTab = document.createElement("button");
    allTab.className = "scene-tab-btn active-scene-tab";
    allTab.innerText = "🎬 Đầy Đủ (Đơn Cảnh)";
    allTab.title = "Truy vấn toàn bộ đề bài thành 1 cảnh đơn";
    allTab.addEventListener("click", () => {
        switchSceneTab(-1);
    });
    multiSceneTabs.appendChild(allTab);

    // 2. Pills từng phân cảnh con
    scenes.forEach((sc, idx) => {
        const tabBtn = document.createElement("button");
        tabBtn.className = "scene-tab-btn";
        tabBtn.id = `scene-tab-btn-${idx}`;
        const titleText = sc.name || `Cảnh ${idx + 1}`;
        tabBtn.innerText = `${idx + 1}️⃣ ${titleText}`;
        tabBtn.title = sc.visual_query || "";
        tabBtn.addEventListener("click", () => {
            switchSceneTab(idx);
        });
        multiSceneTabs.appendChild(tabBtn);
    });
}

// ==========================================================================
// RENDER MULTI-SCENE STORYBOARD (GOM CỤM THEO HÀNG VIDEO TUẦN TỰ THỜI GIAN)
// ==========================================================================
function renderMultiSceneStoryboard(storyboardSequences) {
    if (!resultsGrid) return;
    resultsGrid.classList.add("trake-mode-grid");
    resultsGrid.innerHTML = "";

    if (!storyboardSequences || storyboardSequences.length === 0) {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 40px; text-align: center; color: var(--text-dim);">Không tìm thấy chuỗi phân cảnh nào phù hợp.</div>`;
        return;
    }

    const container = document.createElement("div");
    container.className = "storyboard-container";

    storyboardSequences.forEach((seq) => {
        const groupCard = document.createElement("div");
        groupCard.className = "storyboard-group-card";

        const chronoBadge = seq.is_chronological
            ? `<span class="storyboard-chrono-badge">⏱️ Chuẩn Tuần Tự (t₁ < t₂ < ...)</span>`
            : `<span class="storyboard-chrono-badge" style="background: #854d0e; color: #fef08a;">⚠️ Lệch Thứ Tự</span>`;

        let scenesHtml = "";
        seq.scenes.forEach((sc, scIdx) => {
            const timeSec = sc.timestamp_sec !== undefined ? Number(sc.timestamp_sec) : 0;
            const mins = Math.floor(timeSec / 60);
            const secs = Math.floor(timeSec % 60);
            const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            const scoreVal = sc.score !== undefined ? Number(sc.score).toFixed(4) : "0.0000";

            let arrowHtml = "";
            if (sc.delta_t_to_next !== undefined && sc.delta_t_to_next !== null) {
                const deltaSec = sc.delta_t_to_next;
                const deltaM = Math.floor(deltaSec / 60);
                const deltaS = Math.floor(deltaSec % 60);
                const deltaStr = deltaM > 0 ? `+${deltaM}m ${deltaS}s` : `+${deltaS}s`;

                arrowHtml = `
                    <div class="storyboard-arrow-wrapper">
                        <span class="storyboard-arrow-icon">➔</span>
                        <span class="storyboard-delta-time">${deltaStr}</span>
                    </div>
                `;
            }

            scenesHtml += `
                <div class="storyboard-scene-card" data-vid="${sc.video_id}" data-fid="${sc.frame_id}">
                    <div class="storyboard-scene-tag">🎬 ${sc.scene_name || `Cảnh ${sc.scene_idx}`}</div>
                    <div class="storyboard-thumb-wrapper" title="🔍 Bấm để soi Bounding Box">
                        <img src="${sc.image_url}" loading="lazy" alt="Frame ${sc.frame_id}">
                        <span class="card-score-pill">${scoreVal}</span>
                    </div>
                    <div class="storyboard-meta-bar">
                        <div class="storyboard-info-row">
                            <span style="font-weight: 700; color: #38bdf8;">⏱️ ${timeStr} (${timeSec}s)</span>
                            <span style="color: var(--text-dim);">F:${sc.frame_id}</span>
                        </div>
                        <div class="storyboard-actions-row">
                            <button class="btn-card-action btn-sb-video" title="Xem video tại giây này">▶️ Video</button>
                            <button class="btn-card-action btn-sb-copy" title="Copy Video ID & Frame ID">📋 Copy ID</button>
                        </div>
                    </div>
                </div>
                ${arrowHtml}
            `;
        });

        groupCard.innerHTML = `
            <div class="storyboard-group-header">
                <div class="storyboard-header-left">
                    <span class="storyboard-vid-badge">🏆 #${seq.rank} Video: ${seq.video_id}</span>
                    <span class="storyboard-joint-score">Score Chuỗi: ${seq.joint_score}</span>
                    ${chronoBadge}
                </div>
                <div class="storyboard-header-actions">
                    <button class="btn-card-action btn-sb-timeline-all" data-vid="${seq.video_id}">🎞️ Xem Timeline Toàn Bộ Video</button>
                </div>
            </div>
            <div class="storyboard-sequence-row">
                ${scenesHtml}
            </div>
        `;

        // Gắn sự kiện tương tác cho từng scene card
        groupCard.querySelectorAll(".storyboard-scene-card").forEach((scCard, idx) => {
            const scData = seq.scenes[idx];
            const imgBox = scCard.querySelector(".storyboard-thumb-wrapper");
            if (imgBox) {
                imgBox.addEventListener("click", () => {
                    openLightbox(0, [scData]);
                });
            }

            const vidBtn = scCard.querySelector(".btn-sb-video");
            if (vidBtn) {
                vidBtn.addEventListener("click", () => {
                    openVideoPlayer(scData.video_id, scData.frame_id, scData.timestamp_sec);
                });
            }

            const copyBtn = scCard.querySelector(".btn-sb-copy");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    const text = `${scData.video_id} ${scData.frame_id}`;
                    navigator.clipboard.writeText(text).then(() => {
                        copyBtn.innerText = "✅ ĐÃ COPY!";
                        showToast(`📋 Đã sao chép: ${text}`, "success", 1800);
                        setTimeout(() => { copyBtn.innerText = "📋 Copy ID"; }, 1500);
                    });
                });
            }
        });

        // Nút xem toàn bộ timeline của video
        const tlAllBtn = groupCard.querySelector(".btn-sb-timeline-all");
        if (tlAllBtn) {
            tlAllBtn.addEventListener("click", () => {
                const firstFrame = seq.scenes[0] ? seq.scenes[0].frame_id : null;
                openTimeline(seq.video_id, firstFrame);
            });
        }

        container.appendChild(groupCard);
    });

    resultsGrid.appendChild(container);
}

function switchSceneTab(tabIdx) {
    activeSceneTabIdx = tabIdx;
    if (!multiSceneTabs) return;

    // Cập nhật trạng thái active của các nút Tab
    const buttons = multiSceneTabs.querySelectorAll(".scene-tab-btn");
    buttons.forEach((btn, idx) => {
        if (idx === tabIdx + 1) {
            btn.classList.add("active-scene-tab");
        } else {
            btn.classList.remove("active-scene-tab");
        }
    });

    // Đồng bộ nội dung sang Sidebar để người dùng dễ dàng xem và chỉnh sửa
    if (tabIdx >= 0 && latestParsedScenes && latestParsedScenes[tabIdx]) {
        const sc = latestParsedScenes[tabIdx];
        if (searchInput) searchInput.value = sc.visual_query || "";
        if (ocrInput) ocrInput.value = sc.ocr_query || "";
        if (speechInput) speechInput.value = sc.speech_query || "";
        if (statusText) statusText.innerText = `🎬 Đang xem [Cảnh ${tabIdx + 1}: ${sc.name || ""}]. Bấm "🚀 SEARCH NOW" (cho riêng cảnh này) hoặc "🎬 MULTI-SCENE" (quét tất cả).`;
    } else if (tabIdx === -1) {
        if (searchInput) searchInput.value = latestUnifiedVisual || latestRerankerQuery || "";
        if (ocrInput) ocrInput.value = latestUnifiedOcr || "";
        if (speechInput) speechInput.value = latestUnifiedSpeech || "";
        if (statusText) statusText.innerText = "🎬 Đang ở chế độ Đầy Đủ (Giao Thoa Storyboard). Bấm '🚀 SEARCH NOW' để tìm đơn cảnh, hoặc '🎬 MULTI-SCENE' để quét giao thoa.";
    }

    // Nếu đã có kết quả tìm kiếm trước đó thì chuyển đổi chế độ xem kết quả
    if (!multiSceneResultsData) return;

    if (tabIdx === -1) {
        currentResults = multiSceneResultsData.combined_results || [];
        activeResultsSource = currentResults;
        resultCount.innerText = `${currentResults.length} items (Giao Thoa Đa Cảnh)`;
        renderResults(currentResults);
    } else {
        const scResults = multiSceneResultsData.scenes_results[tabIdx]?.results || [];
        currentResults = scResults;
        activeResultsSource = currentResults;
        const scName = multiSceneResultsData.scenes_results[tabIdx]?.name || `Cảnh ${tabIdx + 1}`;
        resultCount.innerText = `${currentResults.length} items (${scName})`;
        renderResults(currentResults);
    }
}

// Đồng bộ 2 chiều: Khi người dùng gõ trực tiếp vào sidebar inputs trong lúc chọn 1 Cảnh con
if (searchInput) {
    searchInput.addEventListener("input", (e) => {
        if (activeSceneTabIdx >= 0 && latestParsedScenes && latestParsedScenes[activeSceneTabIdx]) {
            latestParsedScenes[activeSceneTabIdx].visual_query = e.target.value;
        }
    });
}
if (ocrInput) {
    ocrInput.addEventListener("input", (e) => {
        if (activeSceneTabIdx >= 0 && latestParsedScenes && latestParsedScenes[activeSceneTabIdx]) {
            latestParsedScenes[activeSceneTabIdx].ocr_query = e.target.value;
        }
    });
}
if (speechInput) {
    speechInput.addEventListener("input", (e) => {
        if (activeSceneTabIdx >= 0 && latestParsedScenes && latestParsedScenes[activeSceneTabIdx]) {
            latestParsedScenes[activeSceneTabIdx].speech_query = e.target.value;
        }
    });
}

// ==========================================================================
// HÀM TRUY VẤN ĐA PHÂN CẢNH (MULTI-SCENE SEARCH & VIDEO INTERSECTION)
// ==========================================================================
async function performMultiSceneSearch() {
    if (!latestParsedScenes || latestParsedScenes.length < 2) return;

    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = enableObjectCheck ? enableObjectCheck.checked : false;

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    const topK1Val = parseInt(topKNum ? topKNum.value : 100) || 100;
    const topNVal = parseInt(rerankTopV ? rerankTopV.value : 5) || 5;
    const mFramesVal = parseInt(rerankKPerVideo ? rerankKPerVideo.value : 40) || 40;

    const objClass = isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null;
    const objConf = isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0;
    const simThresh = parseFloat(visualSimNum ? visualSimNum.value : 0.90) || 0.90;

    const wVisual = parseFloat(weightVisualSlider ? weightVisualSlider.value : 0.5);
    const wOcr = parseFloat(weightOcrSlider ? weightOcrSlider.value : 0.2);
    const wAsr = parseFloat(weightAsrSlider ? weightAsrSlider.value : 0.2);
    const wObj = parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.1);

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    searchBtn.disabled = true;
    rerankBtn.disabled = true;
    if (multiSearchBtn) {
        multiSearchBtn.disabled = true;
        multiSearchBtn.innerText = "⏳ ĐANG QUÉT...";
    }
    statusText.innerText = `🎬 Đang quét đồng thời ${latestParsedScenes.length} phân cảnh & Giao thoa Top ${topNVal} video...`;
    showToast(`🎬 Đang quét ${latestParsedScenes.length} phân cảnh & Giao thoa Top ${topNVal} Video...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(MULTI_SCENE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenes: latestParsedScenes,
                enable_visual: isVis,
                enable_ocr: isOcr,
                enable_asr: isAsr,
                enable_object: isObj,
                object_class_id: objClass,
                object_conf_thresh: objConf,
                weight_visual: wVisual,
                weight_ocr: wOcr,
                weight_asr: wAsr,
                weight_object: wObj,
                dedup_mode: dedupMode,
                visual_sim_thresh: simThresh,
                enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
                cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
                top_k: topK1Val,
                top_v: topNVal,
                k_per_video: mFramesVal,
                use_rrf: isRrf,
                rrf_k: rrfK,
                asr_window_n: selectedAsrWindowN
            })
        });

        if (!response.ok) {
            throw new Error(`Multi-Scene Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        multiSceneResultsData = data;
        currentResults = data.combined_results || [];
        const elapsedSec = data.elapsed_sec || ((performance.now() - t0) / 1000).toFixed(2);
        const topVList = (data.top_videos || []).join(", ");
        resultCount.innerText = `${currentResults.length} items (Giao Thoa ${latestParsedScenes.length} Cảnh)`;
        statusText.innerText = `🎯 ĐA PHÂN CẢNH: Giao thoa Top ${topNVal} Video [${topVList}] -> ${currentResults.length} frames (${elapsedSec}s)!`;
        showToast(`✅ Đã tìm thấy ${currentResults.length} frames từ Top ${topNVal} video (${elapsedSec}s)!`, "success");

        switchSceneTab(-1);

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR MULTI-SCENE] ${err.message}`;
        showToast(`Lỗi Multi-Scene: ${err.message}`, "error");
        console.error(err);
    } finally {
        searchBtn.disabled = false;
        rerankBtn.disabled = false;
        if (multiSearchBtn) {
            multiSearchBtn.disabled = false;
            multiSearchBtn.innerText = "🎬 MULTI-SCENE";
        }
        updateK2Calculation();
    }
}

// ==========================================================================
// RENDER TRAKE STORYBOARD MATRIX (TARS MONOTONIC DP RESULTS)
// ==========================================================================
function renderTrakeStoryboard(storyboardSequences) {
    if (!resultsGrid) return;
    resultsGrid.classList.add("trake-mode-grid");
    resultsGrid.innerHTML = "";

    // Hiển thị thanh công cụ Qwen AI Reranker khi ở chế độ TRAKE
    if (trakeQwenToolbar) {
        trakeQwenToolbar.style.display = (currentAppMode === "TRAKE" && storyboardSequences && storyboardSequences.length > 0) ? "flex" : "none";
    }

    if (!storyboardSequences || storyboardSequences.length === 0) {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 50px; text-align: center; color: var(--text-dim); font-size: 0.95rem;">
            ⏱️ Không tìm thấy chuỗi sự kiện TRAKE nào phù hợp. Vui lòng kiểm tra lại bối cảnh và các sự kiện con.
        </div>`;
        return;
    }

    const wrapper = document.createElement("div");
    wrapper.className = "trake-results-wrapper";

    storyboardSequences.forEach((seq) => {
        const groupCard = document.createElement("div");
        groupCard.className = "trake-group-card";

        const chronoBadge = seq.is_chronological
            ? `<span class="storyboard-chrono-badge">⏱️ Chuẩn Tuần Tự (t₁ < t₂ < ...)</span>`
            : `<span class="storyboard-chrono-badge" style="background: #854d0e; color: #fef08a;">⚠️ Lệch Thứ Tự</span>`;

        let qwenBadgeHtml = "";
        if (seq.qwen_score !== undefined && seq.qwen_score !== null) {
            let qwenClass = "qwen-high";
            if (seq.qwen_score < 0.5) qwenClass = "qwen-low";
            else if (seq.qwen_score < 0.8) qwenClass = "qwen-med";
            const deltaStr = seq.original_rank ? `<span class="storyboard-rank-delta">(TARS #${seq.original_rank})</span>` : "";
            qwenBadgeHtml = `<span class="storyboard-qwen-score ${qwenClass}">🤖 Qwen: ${Number(seq.qwen_score).toFixed(4)}</span> ${deltaStr}`;
        }

        let scenesHtml = "";
        seq.scenes.forEach((sc, scIdx) => {
            const timeSec = sc.timestamp_sec !== undefined ? Number(sc.timestamp_sec) : 0;
            const mins = Math.floor(timeSec / 60);
            const secs = Math.floor(timeSec % 60);
            const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            const scoreVal = sc.score !== undefined ? Number(sc.score).toFixed(4) : "0.0000";

            let arrowHtml = "";
            if (sc.delta_t_to_next !== undefined && sc.delta_t_to_next !== null) {
                const deltaSec = sc.delta_t_to_next;
                const deltaM = Math.floor(deltaSec / 60);
                const deltaS = Math.floor(deltaSec % 60);
                const deltaStr = deltaM > 0 ? `+${deltaM}m ${deltaS}s` : `+${deltaS}s`;

                arrowHtml = `
                    <div class="storyboard-arrow-wrapper">
                        <span class="storyboard-arrow-icon">➔</span>
                        <span class="storyboard-delta-time">${deltaStr}</span>
                    </div>
                `;
            }

            const eventLabel = sc.scene_name ? (sc.scene_name.includes(":") ? sc.scene_name.split(":")[0] : `E${scIdx + 1}`) : `E${scIdx + 1}`;

            scenesHtml += `
                <div class="storyboard-scene-card" data-vid="${sc.video_id}" data-fid="${sc.frame_id}">
                    <span class="trake-event-tag">${eventLabel}</span>
                    <div class="storyboard-thumb-wrapper" title="🔍 Bấm để xem chi tiết / Bounding Box">
                        <img src="${sc.image_url}" loading="lazy" alt="Frame ${sc.frame_id}">
                        <span class="card-score-pill">${scoreVal}</span>
                    </div>
                    <div class="storyboard-meta-bar">
                        <div class="storyboard-info-row">
                            <span style="font-weight: 700; color: #38bdf8;">⏱️ ${timeStr} (${timeSec}s)</span>
                            <span style="color: var(--text-dim); font-weight: 700;">F:${sc.frame_id}</span>
                        </div>
                        <div class="storyboard-actions-row">
                            <button class="btn-card-action btn-sb-video" title="Xem video tại giây này">▶️ Video</button>
                            <button class="btn-card-action btn-sb-copy" title="Copy Frame ID">📋 Copy ID</button>
                        </div>
                    </div>
                </div>
                ${arrowHtml}
            `;
        });

        groupCard.innerHTML = `
            <div class="trake-group-header">
                <div class="storyboard-header-left" style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <span class="trake-vid-badge">🏆 #${seq.rank} Video: ${seq.video_id}</span>
                    <span style="font-family: var(--font-mono); font-size: 0.76rem; color: #cbd5e1;">Path #${seq.path_idx || 1}</span>
                    <span class="storyboard-joint-score">Score: ${seq.joint_score}</span>
                    ${chronoBadge}
                    ${qwenBadgeHtml}
                </div>
                <div class="storyboard-header-actions" style="display: flex; align-items: center; gap: 8px;">
                    <code class="trake-sub-code">${seq.submission_line}</code>
                    <button class="btn-card-action btn-copy-trake-line" data-line="${seq.submission_line}">📋 Copy Line</button>
                    <button class="btn-card-action btn-sb-timeline-all" data-vid="${seq.video_id}">🎞️ Timeline</button>
                </div>
            </div>
            <div class="storyboard-sequence-row">
                ${scenesHtml}
            </div>
        `;

        // Gắn sự kiện tương tác cho từng scene card
        groupCard.querySelectorAll(".storyboard-scene-card").forEach((scCard, idx) => {
            const scData = seq.scenes[idx];
            const imgBox = scCard.querySelector(".storyboard-thumb-wrapper");
            if (imgBox) {
                imgBox.addEventListener("click", () => {
                    openLightbox(0, [scData]);
                });
            }

            const vidBtn = scCard.querySelector(".btn-sb-video");
            if (vidBtn) {
                vidBtn.addEventListener("click", () => {
                    openVideoPlayer(scData.video_id, scData.frame_id, scData.timestamp_sec);
                });
            }

            const copyBtn = scCard.querySelector(".btn-sb-copy");
            if (copyBtn) {
                copyBtn.addEventListener("click", () => {
                    const text = `${scData.video_id} ${scData.frame_id}`;
                    navigator.clipboard.writeText(text).then(() => {
                        copyBtn.innerText = "✅ ĐÃ COPY!";
                        showToast(`📋 Đã sao chép: ${text}`, "success", 1800);
                        setTimeout(() => { copyBtn.innerText = "📋 Copy ID"; }, 1500);
                    });
                });
            }
        });

        // Nút Copy toàn bộ dòng TRAKE Line
        const copyLineBtn = groupCard.querySelector(".btn-copy-trake-line");
        if (copyLineBtn) {
            copyLineBtn.addEventListener("click", () => {
                const subLine = copyLineBtn.getAttribute("data-line");
                navigator.clipboard.writeText(subLine).then(() => {
                    copyLineBtn.innerText = "✅ ĐÃ COPY LINE!";
                    showToast(`📋 Đã copy TRAKE submission line:\n${subLine}`, "success", 2500);
                    setTimeout(() => { copyLineBtn.innerText = "📋 Copy TRAKE Line"; }, 2000);
                });
            });
        }

        // Nút xem toàn bộ timeline của video
        const tlAllBtn = groupCard.querySelector(".btn-sb-timeline-all");
        if (tlAllBtn) {
            tlAllBtn.addEventListener("click", () => {
                const firstFrame = seq.scenes[0] ? seq.scenes[0].frame_id : null;
                openTimeline(seq.video_id, firstFrame);
            });
        }

        wrapper.appendChild(groupCard);
    });

    resultsGrid.appendChild(wrapper);
}

// ==========================================================================
// HÀM QWEN-VL RE-RANK CHO CÁC CHUỖI TRAKE STORYBOARD (GIAI ĐOẠN 2)
// ==========================================================================
async function performQwenRerankChains() {
    if (!latestTrakeSequences || latestTrakeSequences.length === 0) {
        statusText.innerText = "⚠️ Chưa có dữ liệu chuỗi TRAKE. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.";
        showToast("⚠️ Chưa có kết quả TRAKE. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.", "warning", 3000);
        return;
    }

    const topKVal = parseInt(qwenTopkChainsInput ? qwenTopkChainsInput.value : 5) || 5;
    const globalQ = searchInput ? searchInput.value.trim() : (latestUnifiedVisual || "");
    const hintVal = rerankHintInput ? rerankHintInput.value.trim() : "";

    if (trakeQwenRerankBtn) {
        trakeQwenRerankBtn.disabled = true;
        trakeQwenRerankBtn.innerText = "⏳ ĐANG AI RERANK...";
    }
    if (qwenProgressText) qwenProgressText.innerText = `Đang gửi ${topKVal} chuỗi lên Qwen-VL GPU...`;
    showToast(`🤖 Đang chấm điểm Top ${topKVal} chuỗi TRAKE qua Qwen-VL GPU...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(QWEN_RERANK_CHAINS_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                chains: latestTrakeSequences,
                global_query: globalQ,
                discriminating_features: currentDiscriminatingFeatures,
                rerank_hint: hintVal,
                top_k_chains: topKVal,
                asr_window_n: selectedAsrWindowN
            })
        });

        if (!response.ok) {
            throw new Error(`Qwen Rerank Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        const elapsed = ((performance.now() - t0) / 1000).toFixed(2);

        if (data.storyboard_sequences && data.storyboard_sequences.length > 0) {
            latestTrakeSequences = data.storyboard_sequences;
            renderTrakeStoryboard(latestTrakeSequences);
            if (qwenProgressText) qwenProgressText.innerText = `✅ Đã chấm ${data.total_evaluated} chuỗi (${elapsed}s)`;
            statusText.innerText = `🤖 QWEN RERANK HOÀN TẤT: Đã tái xếp hạng ${data.total_evaluated} chuỗi TRAKE trong ${elapsed}s!`;
            showToast(`✅ Qwen-VL đã tái xếp hạng thành công ${data.total_evaluated} chuỗi (${elapsed}s)!`, "success");
        }
    } catch (err) {
        dismissLoadingToast();
        if (qwenProgressText) qwenProgressText.innerText = `❌ Lỗi: ${err.message}`;
        statusText.innerText = `[ERROR QWEN RERANK] ${err.message}`;
        showToast(`Lỗi Qwen Rerank: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (trakeQwenRerankBtn) {
            trakeQwenRerankBtn.disabled = false;
            trakeQwenRerankBtn.innerText = "🤖 QWEN RERANK CHAINS";
        }
    }
}

// Gắn sự kiện kích hoạt Qwen Reranker
if (trakeQwenRerankBtn) {
    trakeQwenRerankBtn.addEventListener("click", performQwenRerankChains);
}

// ==========================================================================
// HÀM TRUY VẤN TRAKE (TARS MONOTONIC DP & EVENT EXTRACTION)
// ==========================================================================
async function performTrakeSearch() {
    let scenesToSearch = latestParsedScenes;
    if (!scenesToSearch || scenesToSearch.length < 2) {
        statusText.innerText = "⚠️ Chế độ TRAKE yêu cầu ít nhất 2 sự kiện (E1, E2...). Vui lòng dán đề bài và bấm '✨ PHÂN TÁCH ĐỀ BÀI' trước.";
        showToast("⚠️ Vui lòng dán đề bài và bấm '✨ PHÂN TÁCH ĐỀ BÀI' để bóc tách các sự kiện E1..EN.", "warning", 3500);
        return;
    }

    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = enableObjectCheck ? enableObjectCheck.checked : false;

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    const lambdaVal = parseFloat(trakeLambdaInput ? trakeLambdaInput.value : 0.001) || 0.001;
    const kPathsVal = parseInt(trakeKpathsInput ? trakeKpathsInput.value : 3) || 3;
    const topKSeqVal = parseInt(trakeTopKInput ? trakeTopKInput.value : 10) || 10;
    const topK1Val = parseInt(trakeCandidatePoolInput ? trakeCandidatePoolInput.value : (topKNum ? topKNum.value : 100)) || 100;
    const topVVal = parseInt(trakeTopVInput ? trakeTopVInput.value : 10) || 10;
    const mFramesVal = parseInt(trakeFramesPerVidInput ? trakeFramesPerVidInput.value : 40) || 40;

    const objClass = isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null;
    const objConf = isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0;
    const simThresh = parseFloat(visualSimNum ? visualSimNum.value : 0.90) || 0.90;

    const wVisual = isVis ? parseFloat(weightVisualSlider ? weightVisualSlider.value : 1.0) : 0.0;
    const wOcr = isOcr ? parseFloat(weightOcrSlider ? weightOcrSlider.value : 1.0) : 0.0;
    const wAsr = isAsr ? parseFloat(weightAsrSlider ? weightAsrSlider.value : 1.0) : 0.0;
    const wObj = isObj ? parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.5) : 0.0;

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    if (trakeSearchBtn) {
        trakeSearchBtn.disabled = true;
        trakeSearchBtn.innerText = "⏳ ĐANG CHẠY TARS DP...";
    }
    statusText.innerText = `⏱️ Đang thực thi TARS Monotonic DP (Pool K1=${topK1Val}, Top V=${topVVal}, λ=${lambdaVal}, K-paths=${kPathsVal})...`;
    showToast(`⏱️ Đang chạy TARS Monotonic DP (Pool K1=${topK1Val}, Top V=${topVVal})...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(MULTI_SCENE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scenes: scenesToSearch,
                enable_visual: isVis,
                enable_ocr: isOcr,
                enable_asr: isAsr,
                enable_object: isObj,
                object_class_id: objClass,
                object_conf_thresh: objConf,
                weight_visual: wVisual,
                weight_ocr: wOcr,
                weight_asr: wAsr,
                weight_object: wObj,
                dedup_mode: dedupMode,
                visual_sim_thresh: simThresh,
                enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
                cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
                top_k: topK1Val,
                top_v: topVVal,
                k_per_video: mFramesVal,
                use_rrf: isRrf,
                rrf_k: rrfK,
                trake_mode: true,
                lambda_penalty: lambdaVal,
                alpha_fusion: 0.35,
                k_paths: kPathsVal,
                trake_top_k: topKSeqVal,
                asr_window_n: selectedAsrWindowN
            })
        });

        if (!response.ok) {
            throw new Error(`TRAKE Search Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();

        latestTrakeSequences = data.storyboard_sequences || [];
        latestTrakeSubmissionLines = data.trake_submission_lines || [];
        const elapsedSec = data.elapsed_sec || ((performance.now() - t0) / 1000).toFixed(2);

        resultCount.innerText = `${latestTrakeSequences.length} sequences (${data.top_videos ? data.top_videos.length : 0} videos)`;
        statusText.innerText = `⏱️ TRAKE: TARS DP hoàn tất trong ${elapsedSec}s! Tìm thấy ${latestTrakeSequences.length} chuỗi tuần tự từ Top ${data.top_videos ? data.top_videos.length : 0} Video.`;
        showToast(`✅ Đã tìm thấy ${latestTrakeSequences.length} chuỗi TRAKE tối ưu (${elapsedSec}s)!`, "success");

        renderTrakeStoryboard(latestTrakeSequences);

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR TRAKE] ${err.message}`;
        showToast(`Lỗi TRAKE Search: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (trakeSearchBtn) {
            trakeSearchBtn.disabled = false;
            trakeSearchBtn.innerText = "⏱️ SEARCH TRAKE";
        }
    }
}

// ==========================================================================
// HÀM XUẤT CSV NỘP BÀI TRAKE (EXPORT SUBMISSION CSV)
// ==========================================================================
function exportTrakeCsv() {
    if (!latestTrakeSubmissionLines || latestTrakeSubmissionLines.length === 0) {
        statusText.innerText = "⚠️ Chưa có dữ liệu chuỗi TRAKE để xuất CSV. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.";
        showToast("⚠️ Chưa có kết quả TRAKE. Vui lòng bấm '⏱️ SEARCH TRAKE' trước.", "warning", 3000);
        return;
    }

    const csvContent = latestTrakeSubmissionLines.join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = url;
    a.download = `aic2026_trake_submission_${new Date().toISOString().replace(/[:.]/g, "-")}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast(`📥 Đã xuất thành công ${latestTrakeSubmissionLines.length} dòng submission vào file CSV!`, "success", 3000);
}

// ==========================================================================
// HÀM TÌM KIẾM ĐƠN CẢNH / THEO INPUT HIỆN TẠI (SEARCH NOW)
// ==========================================================================
async function performSearch() {
    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = enableObjectCheck ? enableObjectCheck.checked : false;

    const qVisual = isVis && searchInput ? searchInput.value.trim() : "";
    const qOcr = isOcr && ocrInput ? ocrInput.value.trim() : "";
    const qSpeech = isAsr && speechInput ? speechInput.value.trim() : "";
    const vId = videoIdInput ? videoIdInput.value.trim() : "";

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    if (!qVisual && !qOcr && !qSpeech && !vId && !isObj) {
        if (currentImageQueryBase64) {
            executeImageSearch();
            return;
        }
        statusText.innerText = "⚠️ Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.";
        showToast("Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.", "warning");
        return;
    }

    const imageCount = currentImageQueries.length;
    const hasImageQuery = imageCount > 0;
    const isComposed = hasImageQuery && (!!qVisual || !!qOcr);

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    searchBtn.disabled = true;
    searchBtn.innerText = "SEARCHING...";
    if (btnSearchImageQuery) {
        btnSearchImageQuery.disabled = true;
        btnSearchImageQuery.innerText = "⏳ ĐANG TÌM...";
    }

    if (isComposed) {
        statusText.innerText = `🖼️ (${imageCount} góc ảnh) + 📝 Đang tìm kiếm KẾT HỢP (Ảnh + Văn bản: '${qVisual}') (${isRrf ? `RRF k=${rrfK}` : "Weighted"})...`;
        showToast(`🖼️ (${imageCount} góc ảnh) + 📝 Đang tìm kiếm kết hợp Đa Phương Thức...`, "loading", 0);
    } else if (hasImageQuery) {
        statusText.innerText = `🖼️ Đang tìm kiếm bằng ${imageCount} góc ảnh (Qwen3-VL 2048D Multi-View)...`;
        showToast(`🖼️ Đang tìm kiếm bằng ${imageCount} góc ảnh (Qwen3-VL 2048D Multi-View)...`, "loading", 0);
    } else {
        statusText.innerText = `Đang tìm kiếm đa kênh (${isRrf ? `RRF k=${rrfK}` : "Weighted"} | Vis: ${isVis ? "ON" : "OFF"}, ASR: ${isAsr ? "ON" : "OFF"}, OCR: ${isOcr ? "ON" : "OFF"}, RT-DETR: ${isObj ? "ON" : "OFF"})...`;
        showToast(`⏳ Đang tìm kiếm (${isRrf ? `RRF k=${rrfK}` : "Weighted"})...`, "loading", 0);
    }
    resultsGrid.innerHTML = "";

    commitPendingTags();

    const selCat = categorySelect ? categorySelect.value : "";
    const selSubCat = (groupSubcategory && groupSubcategory.style.display !== "none" && subcategorySelect) ? subcategorySelect.value : "";

    const payload = {
        query: qVisual,
        image_base64: currentImageQueries.length > 0 ? currentImageQueries[0] : null,
        images_base64: currentImageQueries.length > 0 ? currentImageQueries : null,
        image_weight: 0.5,
        ocr_query: qOcr,
        speech_query: qSpeech,
        enable_asr: isAsr,
        enable_object: isObj,
        object_class_id: isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null,
        object_conf_thresh: isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0,
        dedup_mode: dedupMode,
        visual_sim_thresh: parseFloat(visualSimNum ? visualSimNum.value : 0.90),
        enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
        cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
        video_id: "",
        include_videos: currentIncludeTags.length > 0 ? currentIncludeTags : null,
        exclude_videos: currentExcludeTags.length > 0 ? currentExcludeTags : null,
        category: (selCat && selCat !== "all") ? selCat : "",
        sub_category: (selSubCat && selSubCat !== "all") ? selSubCat : "",
        weight_visual: isVis ? parseFloat(weightVisualSlider ? weightVisualSlider.value : 1.0) : 0.0,
        weight_ocr: isOcr ? parseFloat(weightOcrSlider ? weightOcrSlider.value : 1.0) : 0.0,
        weight_asr: isAsr ? parseFloat(weightAsrSlider ? weightAsrSlider.value : 1.0) : 0.0,
        weight_object: isObj ? parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.5) : 0.0,
        top_k: parseInt(topKNum ? topKNum.value : 60) || 60,
        use_rrf: isRrf,
        rrf_k: rrfK,
        asr_window_n: selectedAsrWindowN
    };

    const t0 = performance.now();

    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        currentResults = data.results || [];
        currentResults.forEach((r, i) => {
            r.original_rank = i + 1;
        });
        activeResultsSource = currentResults;
        const elapsedMs = ((performance.now() - t0) / 1000).toFixed(2);

        resultCount.innerText = `${currentResults.length} items found`;
        statusText.innerText = `✅ Xong: ${currentResults.length} frames (${elapsedMs}s)`;
        showToast(`✅ Đã tìm thấy ${currentResults.length} frames (${elapsedMs}s)!`, "success");

        renderResults(currentResults);

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR] ${err.message}`;
        showToast(`Lỗi tìm kiếm: ${err.message}`, "error");
        console.error(err);
    } finally {
        searchBtn.disabled = false;
        searchBtn.innerText = "🚀 SEARCH NOW";
    }
}

// ==========================================================================
// TỰ ĐỘNG TÍNH TOÁN K2 = N x M VÀ ĐỒNG BỘ NHÃN GIAO DIỆN
// ==========================================================================
function updateK2Calculation() {
    const n = parseInt(rerankTopV ? rerankTopV.value : 5) || 1;
    const m = parseInt(rerankKPerVideo ? rerankKPerVideo.value : 40) || 1;
    const k2 = n * m;

    if (labelK2Total) labelK2Total.innerText = `${k2} frames`;
    if (btnTopVLabel) btnTopVLabel.innerText = n;
    if (btnK2Label) btnK2Label.innerText = k2;
}

if (rerankTopV) rerankTopV.addEventListener("input", updateK2Calculation);
if (rerankKPerVideo) rerankKPerVideo.addEventListener("input", updateK2Calculation);
updateK2Calculation();

// ==========================================================================
// HÀM TÁI XẾP HẠNG (TWO-STAGE RERANK TOP N VIDEOS TỪ TOP K1 BAN ĐẦU)
// ==========================================================================
async function performTwoStageRerank() {
    if (isMultiSceneMode && latestParsedScenes && latestParsedScenes.length >= 2) {
        return performMultiSceneSearch();
    }

    const isVis = enableVisualCheck ? enableVisualCheck.checked : true;
    const isOcr = enableOcrCheck ? enableOcrCheck.checked : true;
    const isAsr = enableAsrCheck ? enableAsrCheck.checked : true;
    const isObj = enableObjectCheck ? enableObjectCheck.checked : false;

    const qVisual = isVis && searchInput ? searchInput.value.trim() : "";
    const qOcr = isOcr && ocrInput ? ocrInput.value.trim() : "";
    const qSpeech = isAsr && speechInput ? speechInput.value.trim() : "";

    if (!qVisual && !qOcr && !qSpeech && !isObj) {
        statusText.innerText = "⚠️ Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.";
        showToast("⚠️ Vui lòng nhập Visual Query, OCR, Lời thoại hoặc bật Lọc Vật Thể.", "warning");
        return;
    }

    let dedupMode = "none";
    if (uniqueVideoCheck && uniqueVideoCheck.checked) {
        dedupMode = "unique";
    } else if (smartDedupCheck && smartDedupCheck.checked) {
        const selectedRadio = document.querySelector('input[name="smart-submode"]:checked');
        dedupMode = selectedRadio ? selectedRadio.value : "smart";
    }

    const topK1Val = parseInt(topKNum ? topKNum.value : 100) || 100;
    const topNVal = parseInt(rerankTopV ? rerankTopV.value : 5) || 5;
    const mFramesVal = parseInt(rerankKPerVideo ? rerankKPerVideo.value : 40) || 40;
    const expectedK2 = topNVal * mFramesVal;

    const objClass = isObj ? parseInt(objectClassSelect ? objectClassSelect.value : 0) : null;
    const objConf = isObj ? parseFloat(objectConfSlider ? objectConfSlider.value : 0.3) : 0.0;
    const simThresh = parseFloat(visualSimNum ? visualSimNum.value : 0.90) || 0.90;

    const wVisual = parseFloat(weightVisualSlider ? weightVisualSlider.value : 0.5);
    const wOcr = parseFloat(weightOcrSlider ? weightOcrSlider.value : 0.2);
    const wAsr = parseFloat(weightAsrSlider ? weightAsrSlider.value : 0.2);
    const wObj = parseFloat(weightObjectSlider ? weightObjectSlider.value : 0.1);

    commitPendingTags();

    const selCat = categorySelect ? categorySelect.value : "";
    const selSubCat = (groupSubcategory && groupSubcategory.style.display !== "none" && subcategorySelect) ? subcategorySelect.value : "";

    const isRrf = useRrfCheck ? useRrfCheck.checked : true;
    const rrfK = rrfKInput ? (parseInt(rrfKInput.value) || 60) : 60;

    rerankBtn.disabled = true;
    rerankBtn.innerText = "⏳ ĐANG RERANK...";
    statusText.innerText = `🎯 Đang lấy Top ${topNVal} Video từ Top ${topK1Val} ban đầu -> Quét ${expectedK2} frames...`;
    showToast(`⏳ Đang Rerank Top ${topNVal} Video (${expectedK2} frames)...`, "loading", 0);

    const t0 = performance.now();

    try {
        const response = await fetch(RERANK_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: qVisual,
                image_base64: currentImageQueryBase64 || null,
                reranker_query: qVisual,
                top_v: topNVal,
                k_per_video: mFramesVal,
                ocr_query: qOcr,
                speech_query: qSpeech,
                enable_asr: isAsr,
                enable_object: isObj,
                object_class_id: objClass,
                object_conf_thresh: objConf,
                weight_visual: wVisual,
                weight_ocr: wOcr,
                weight_asr: wAsr,
                weight_object: wObj,
                dedup_mode: dedupMode,
                visual_sim_thresh: simThresh,
                enable_cross_video_dedup: enableCrossVideoDedupCheck ? enableCrossVideoDedupCheck.checked : true,
                cross_video_sim_thresh: parseFloat(crossVideoSimNum ? crossVideoSimNum.value : 0.90) || 0.90,
                video_id: "",
                include_videos: currentIncludeTags.length > 0 ? currentIncludeTags : null,
                exclude_videos: currentExcludeTags.length > 0 ? currentExcludeTags : null,
                category: (selCat && selCat !== "all") ? selCat : "",
                sub_category: (selSubCat && selSubCat !== "all") ? selSubCat : "",
                top_k: topK1Val,
                use_rrf: isRrf,
                rrf_k: rrfK
            })
        });

        if (!response.ok) {
            throw new Error(`Rerank Error: ${response.statusText}`);
        }

        const data = await response.json();
        dismissLoadingToast();
        currentResults = data.results || [];
        currentResults.forEach((r, i) => {
            r.original_rank = i + 1;
        });
        activeResultsSource = currentResults;
        const elapsedSec = data.elapsed_sec || ((performance.now() - t0) / 1000).toFixed(2);
        const topVList = (data.top_videos || []).join(", ");

        resultCount.innerText = `${currentResults.length} reranked items`;
        statusText.innerText = `🎯 ĐÃ RERANK: Lọc ${data.total_candidates} frames từ Top ${topNVal} Video [${topVList}] (${elapsedSec}s)!`;
        showToast(`🎯 Đã Rerank: ${currentResults.length} frames từ Top ${topNVal} Video (${elapsedSec}s)!`, "success");

        renderResults(currentResults);

        // Cuộn mượt lên đầu danh sách để xem ngay kết quả Rerank Top 1
        const mainContent = document.querySelector(".main-content");
        if (mainContent) mainContent.scrollIntoView({ behavior: "smooth", block: "start" });

    } catch (err) {
        dismissLoadingToast();
        statusText.innerText = `[ERROR RERANK] ${err.message}`;
        showToast(`Lỗi Rerank: ${err.message}`, "error");
        console.error(err);
    } finally {
        rerankBtn.disabled = false;
        updateK2Calculation();
        rerankBtn.innerText = "🎯 RERANK VIDEO";
    }
}

if (searchBtn) {
    searchBtn.addEventListener("click", performSearch);
}

if (trakeSingleSearchBtn) {
    trakeSingleSearchBtn.addEventListener("click", performSearch);
}

if (rerankBtn) {
    rerankBtn.addEventListener("click", performTwoStageRerank);
}

// ==========================================================================
// RENDER KẾT QUẢ TÌM KIẾM THEO CHUẨN LINEAR PRO MINIMALIST
// ==========================================================================
function renderResults(results) {
    if (!resultsGrid) return;
    resultsGrid.classList.remove("trake-mode-grid");
    resultsGrid.innerHTML = ""; // Xóa sạch kết quả cũ trước khi vẽ mới

    if (results.length === 0) {
        resultsGrid.innerHTML = `<div style="grid-column: 1/-1; padding: 40px; text-align: center; color: var(--text-dim); font-size: 0.9rem;">Không tìm thấy keyframe nào thỏa mãn bộ lọc.</div>`;
        return;
    }

    results.forEach((res, idx) => {
        const card = document.createElement("div");
        const isExcluded = currentExcludeTags.includes((res.video_id || "").toUpperCase());
        card.className = isExcluded ? "result-card card-excluded" : "result-card";
        card.dataset.videoId = res.video_id || "";

        const scoreFinal = (res.score !== undefined && res.score !== null) ? Number(res.score).toFixed(4) : "0.0000";
        const scoreVis = (res.visual_score !== undefined && res.visual_score !== null) ? Number(res.visual_score).toFixed(4) : "0.0000";
        const scoreOcr = (res.ocr_score !== undefined && res.ocr_score !== null) ? Number(res.ocr_score).toFixed(4) : "0.0000";
        const scoreAsr = (res.asr_score !== undefined && res.asr_score !== null) ? Number(res.asr_score).toFixed(4) : "0.0000";
        const scoreObj = (res.obj_score !== undefined && res.obj_score !== null) ? Number(res.obj_score).toFixed(4) : "0.0000";

        // Mốc thời gian
        let timeStr = "";
        if (res.timestamp_sec !== undefined && res.timestamp_sec !== null) {
            const totalSec = Math.floor(res.timestamp_sec);
            const m = Math.floor(totalSec / 60).toString().padStart(2, "0");
            const s = (totalSec % 60).toString().padStart(2, "0");
            timeStr = `${m}:${s}`;
        }

        // Speech Box Callout
        let speechHtml = "";
        if (res.asr_transcript && res.asr_transcript.trim()) {
            const startStr = res.asr_start !== undefined ? Number(res.asr_start).toFixed(1) : "0.0";
            const endStr = res.asr_end !== undefined ? Number(res.asr_end).toFixed(1) : "0.0";
            speechHtml = `
                <div class="speech-box">
                    🎙️ "${res.asr_transcript}"
                    <span class="speech-time">⏱️ [${startStr}s ➔ ${endStr}s]</span>
                </div>
            `;
        }

        const boxCount = (res.boxes && res.boxes.length > 0) ? res.boxes.length : 0;
        const boxBadge = boxCount > 0 ? `<span class="badge-obj">📦 ${boxCount} objs</span>` : "";
        const rerankBadge = res.is_reranked ? `<span class="badge-reranked">🎯 Reranked</span>` : "";
        const sceneBadge = res.scene_name ? `<span class="scene-badge-tag">🎬 ${res.scene_name}</span>` : "";

        card.innerHTML = `
            <div class="card-thumb-wrapper" title="🔍 Bấm vào ảnh để phóng to và soi Bounding Box">
                <img src="${res.image_url}" loading="lazy" alt="Keyframe ${res.frame_id}">
                <span class="card-score-pill">Score: ${scoreFinal}</span>
                ${timeStr ? `<span class="card-time-pill">⏱️ ${timeStr}</span>` : ""}
            </div>
            <div class="card-meta-bar">
                <div class="card-video-row">
                    <span class="card-vid-name">#${res.original_rank !== undefined ? res.original_rank : (idx + 1)} ${res.video_id}</span>
                    <span class="card-frame-id">F:${res.frame_id}</span>
                </div>
                <div class="score-breakdown">
                    ${rerankBadge}
                    ${sceneBadge}
                    <span class="badge-vis">👁️ ${scoreVis}</span>
                    <span class="badge-asr">🎙️ ${scoreAsr}</span>
                    ${scoreOcr > 0 ? `<span class="badge-ocr">🔤 ${scoreOcr}</span>` : ""}
                    ${scoreObj > 0 ? `<span class="badge-obj">📦 ${scoreObj}</span>` : ""}
                    ${boxBadge}
                </div>
                ${speechHtml}
                <div class="card-actions-row">
                    <button class="btn-card-action btn-card-video" title="Xem video tại giây này">▶️ Video</button>
                    <button class="btn-card-action btn-card-copy" title="Copy Video ID & Frame ID nộp bài">📋 Copy ID</button>
                    <button class="btn-card-action btn-card-timeline" style="grid-column: 1 / -1;" title="Xem toàn bộ Timeline">🎞️ Timeline</button>
                    <button class="btn-card-exclude ${isExcluded ? 'is-excluded' : ''}" title="Loại trừ / hoàn tác loại trừ video này khỏi kết quả">
                        ${isExcluded ? '🔴 Đã loại • Bỏ loại trừ' : '🚫 Loại video này'}
                    </button>
                </div>
            </div>
        `;

        // CHỈ GẮN SỰ KIỆN PHÓNG TO KHI CLICK ĐÚNG VÀO HÌNH ẢNH
        const imgBox = card.querySelector(".card-thumb-wrapper");
        if (imgBox) {
            imgBox.addEventListener("click", (e) => {
                e.stopPropagation();
                openLightbox(idx, results);
            });
        }

        // Nút xem Video Player trực tiếp tại mốc thời gian của keyframe
        const vidBtn = card.querySelector(".btn-card-video");
        if (vidBtn) {
            vidBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                openVideoPlayer(res.video_id, res.frame_id, res.timestamp_sec);
            });
        }

        // Nút Copy ID trực tiếp trên card
        const copyBtn = card.querySelector(".btn-card-copy");
        if (copyBtn) {
            copyBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const submissionText = `${res.video_id} ${res.frame_id}`;
                navigator.clipboard.writeText(submissionText).then(() => {
                    copyBtn.innerText = "✅ ĐÃ COPY!";
                    showToast(`📋 Đã sao chép: ${submissionText}`, "success", 1800);
                    setTimeout(() => {
                        copyBtn.innerText = "📋 Copy ID";
                    }, 1500);
                });
            });
        }

        // Nút xem Timeline với tính năng Auto-scroll đến đúng frame hiện tại
        const tlBtn = card.querySelector(".btn-card-timeline");
        if (tlBtn) {
            tlBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                openTimeline(res.video_id, res.frame_id);
            });
        }

        // Nút Quick-Exclude: Toggle loại trừ Video này khỏi currentExcludeTags
        const excludeBtn = card.querySelector(".btn-card-exclude");
        if (excludeBtn) {
            excludeBtn.addEventListener("click", (e) => {
                e.stopPropagation();
                const vidUpper = (res.video_id || "").toUpperCase();
                const idx = currentExcludeTags.indexOf(vidUpper);
                if (idx === -1) {
                    // Chưa có → Thêm vào Exclude
                    currentExcludeTags.push(vidUpper);
                    showToast(`🚫 Đã loại trừ video ${res.video_id} khỏi danh sách`, "warning", 2000);
                } else {
                    // Đã có → Xóa khỏi Exclude (Toggle bỏ loại trừ)
                    currentExcludeTags.splice(idx, 1);
                    showToast(`✅ Đã bỏ loại trừ video ${res.video_id}`, "success", 2000);
                }
                // Cập nhật dashboard Exclude Tags
                renderExcludeTags();
                // Cập nhật trạng thái mờ/sáng của TẤT CẢ card cùng video_id trong lưới
                applyExcludeStateToGrid();
            });
        }

        resultsGrid.appendChild(card);
    });
}

// ==========================================================================
// APPLY EXCLUDE STATE TO GRID (không re-render, chỉ cập nhật CSS trực tiếp)
// ==========================================================================
function applyExcludeStateToGrid() {
    if (!resultsGrid) return;
    const cards = resultsGrid.querySelectorAll(".result-card");
    cards.forEach(card => {
        const videoId = (card.dataset.videoId || "").toUpperCase();
        const isExcluded = currentExcludeTags.includes(videoId);
        if (isExcluded) {
            card.classList.add("card-excluded");
        } else {
            card.classList.remove("card-excluded");
        }
        // Cập nhật nút exclude trên card
        const excludeBtn = card.querySelector(".btn-card-exclude");
        if (excludeBtn) {
            if (isExcluded) {
                excludeBtn.classList.add("is-excluded");
                excludeBtn.textContent = "🔴 Đã loại • Bỏ loại trừ";
            } else {
                excludeBtn.classList.remove("is-excluded");
                excludeBtn.textContent = "🚫 Loại video này";
            }
        }
    });
}
window.applyExcludeStateToGrid = applyExcludeStateToGrid;



async function openLightbox(index, sourceList = currentResults) {
    activeResultsSource = sourceList;
    if (!activeResultsSource || index < 0 || index >= activeResultsSource.length) return;

    resetZoom();

    currentLightboxIndex = index;
    const item = activeResultsSource[currentLightboxIndex];

    lightboxTitle.innerText = `${item.video_id} / ${item.frame_id}.jpg`;
    const scoreVal = item.score !== undefined ? Number(item.score).toFixed(4) : "1.0000";
    lightboxScoreBadge.innerText = `Score: ${scoreVal}`;

    const sec = item.timestamp_sec || 0;
    const mins = Math.floor(sec / 60);
    const secs = Math.floor(sec % 60);
    lightboxTimeBadge.innerText = `⏱️ ${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')} (${sec}s)`;

    if (copyStatus) copyStatus.innerText = "";
    
    let asrDisplay = "Không có lời thoại phân đoạn này.";
    if (item.asr_transcript) {
        asrDisplay = `"${item.asr_transcript}"`;
        if (item.asr_prev || item.asr_next) {
            asrDisplay += `\n\n🎙️ [Ngữ cảnh mở rộng ±${selectedAsrWindowN} segments]:\n`;
            if (item.asr_prev) asrDisplay += `⏪ Trước: "${item.asr_prev}"\n`;
            if (item.asr_next) asrDisplay += `⏩ Sau: "${item.asr_next}"`;
        }
    }
    lightboxTranscript.innerText = asrDisplay;

    // Helper render Box Pills
    function renderBoxPills(boxes) {
        if (!lightboxBoxesList) return;
        lightboxBoxesList.innerHTML = "";
        if (boxes && boxes.length > 0) {
            boxes.forEach(b => {
                const pill = document.createElement("span");
                pill.className = "box-pill";
                pill.innerText = `${b.class_name}: ${(b.confidence * 100).toFixed(0)}%`;
                lightboxBoxesList.appendChild(pill);
            });
        } else {
            lightboxBoxesList.innerHTML = `<span style="font-size: 0.8rem; color: #888;">Không phát hiện vật thể nổi bật.</span>`;
        }
    }

    renderBoxPills(item.boxes);

    // Mở modal trước để layout tính đúng clientWidth / clientHeight
    lightboxModal.style.display = "flex";

    // Set image source và kích hoạt vẽ canvas sau khi layout sẵn sàng
    lightboxImg.src = item.image_url;
    if (lightboxImg.complete) {
        setTimeout(drawLightboxCanvas, 50);
    } else {
        lightboxImg.onload = () => {
            setTimeout(drawLightboxCanvas, 50);
        };
    }

    // Nếu item.boxes chưa có, gọi API lấy on-demand
    if (!item.boxes || item.boxes.length === 0) {
        try {
            const bRes = await fetch(`http://localhost:8000/api/keyframe-boxes/${item.video_id}/${item.frame_id}`);
            if (bRes.ok) {
                const bData = await bRes.json();
                if (bData.status === "success" && bData.boxes && bData.boxes.length > 0) {
                    item.boxes = bData.boxes;
                    renderBoxPills(item.boxes);
                    setTimeout(drawLightboxCanvas, 30);
                }
            }
        } catch (e) {
            console.warn("Could not fetch boxes on demand", e);
        }
    }
}

function closeLightbox() {
    resetZoom();
    if (lightboxModal) lightboxModal.style.display = "none";
}

function drawLightboxCanvas() {
    if (!lightboxImg || !lightboxCanvas) return;

    const item = activeResultsSource[currentLightboxIndex];
    if (!item) return;

    const displayW = lightboxImg.clientWidth;
    const displayH = lightboxImg.clientHeight;
    const naturalW = lightboxImg.naturalWidth || 1280;
    const naturalH = lightboxImg.naturalHeight || 720;

    lightboxCanvas.width = displayW;
    lightboxCanvas.height = displayH;

    const ctx = lightboxCanvas.getContext("2d");
    ctx.clearRect(0, 0, displayW, displayH);

    if (!lightboxToggleBoxes.checked || !item.boxes || item.boxes.length === 0) return;

    const scaleX = displayW / naturalW;
    const scaleY = displayH / naturalH;

    item.boxes.forEach((b) => {
        const [x1, y1, x2, y2] = b.box;
        const drawX = x1 * scaleX;
        const drawY = y1 * scaleY;
        const drawW = (x2 - x1) * scaleX;
        const drawH = (y2 - y1) * scaleY;

        ctx.strokeStyle = "#ff6d00";
        ctx.lineWidth = 2.5;
        ctx.strokeRect(drawX, drawY, drawW, drawH);

        const label = `${b.class_name} ${(b.confidence * 100).toFixed(0)}%`;
        ctx.font = "bold 11px 'Space Mono', monospace";
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = "#ff6d00";
        ctx.fillRect(drawX, Math.max(0, drawY - 18), textWidth + 8, 18);

        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, drawX + 4, Math.max(12, drawY - 4));
    });
}

// ==========================================================================
// INTERACTIVE PAN & ZOOM (MOUSE WHEEL ZOOM AT CURSOR & DRAG TO PAN)
// ==========================================================================
let zoomScale = 1.0;
let zoomTranslateX = 0;
let zoomTranslateY = 0;
let isZoomDragging = false;
let zoomStartX = 0;
let zoomStartY = 0;

const lightboxMediaContainer = document.getElementById("lightbox-media-container");
const lightboxZoomBox = document.getElementById("lightbox-zoom-box");
const lightboxZoomBadge = document.getElementById("lightbox-zoom-badge");
const lightboxZoomResetBtn = document.getElementById("lightbox-zoom-reset-btn");

function updateZoomTransform() {
    if (!lightboxZoomBox) return;
    lightboxZoomBox.style.transform = `translate(${zoomTranslateX}px, ${zoomTranslateY}px) scale(${zoomScale})`;
    if (lightboxZoomBadge) {
        lightboxZoomBadge.innerText = `🔍 ${(zoomScale * 100).toFixed(0)}% (Lăn chuột để Zoom / Kéo để Pan)`;
    }
}

function resetZoom() {
    zoomScale = 1.0;
    zoomTranslateX = 0;
    zoomTranslateY = 0;
    isZoomDragging = false;
    if (lightboxMediaContainer) lightboxMediaContainer.classList.remove("is-dragging");
    updateZoomTransform();
}

if (lightboxMediaContainer) {
    // Wheel Zoom toward Cursor
    lightboxMediaContainer.addEventListener("wheel", (e) => {
        e.preventDefault();
        const rect = lightboxMediaContainer.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomDelta = e.deltaY < 0 ? 1.18 : 0.85;
        const nextScale = Math.min(Math.max(1.0, zoomScale * zoomDelta), 10.0);

        if (nextScale === 1.0) {
            resetZoom();
            return;
        }

        // Adjust translate so the point under the cursor remains fixed
        zoomTranslateX = mouseX - (mouseX - zoomTranslateX) * (nextScale / zoomScale);
        zoomTranslateY = mouseY - (mouseY - zoomTranslateY) * (nextScale / zoomScale);
        zoomScale = nextScale;

        updateZoomTransform();
    }, { passive: false });

    // Drag to Pan
    lightboxMediaContainer.addEventListener("mousedown", (e) => {
        if (zoomScale <= 1.0) return;
        isZoomDragging = true;
        zoomStartX = e.clientX - zoomTranslateX;
        zoomStartY = e.clientY - zoomTranslateY;
        lightboxMediaContainer.classList.add("is-dragging");
    });

    window.addEventListener("mousemove", (e) => {
        if (!isZoomDragging) return;
        zoomTranslateX = e.clientX - zoomStartX;
        zoomTranslateY = e.clientY - zoomStartY;
        updateZoomTransform();
    });

    window.addEventListener("mouseup", () => {
        if (isZoomDragging) {
            isZoomDragging = false;
            if (lightboxMediaContainer) lightboxMediaContainer.classList.remove("is-dragging");
        }
    });

    // Double click to toggle Zoom / Reset
    lightboxMediaContainer.addEventListener("dblclick", (e) => {
        if (zoomScale > 1.0) {
            resetZoom();
        } else {
            const rect = lightboxMediaContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            zoomScale = 2.5;
            zoomTranslateX = mouseX - (mouseX * 2.5);
            zoomTranslateY = mouseY - (mouseY * 2.5);
            updateZoomTransform();
        }
    });
}

if (lightboxZoomResetBtn) {
    lightboxZoomResetBtn.addEventListener("click", resetZoom);
}

// Video Button in Lightbox
if (lightboxVideoBtn) {
    lightboxVideoBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (item && item.video_id) {
            openVideoPlayer(item.video_id, item.frame_id, item.timestamp_sec);
        }
    });
}

// Copy ID Button Handler
if (lightboxCopyBtn) {
    lightboxCopyBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (item) {
            const textToCopy = `${item.video_id} ${item.frame_id}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                if (copyStatus) {
                    copyStatus.innerText = `✅ Đã chép: ${textToCopy}`;
                    setTimeout(() => { if (copyStatus) copyStatus.innerText = ""; }, 2500);
                }
            }).catch(err => {
                if (copyStatus) copyStatus.innerText = `Lỗi copy: ${err}`;
            });
        }
    });
}

// Timeline Button in Lightbox
if (lightboxTimelineBtn) {
    lightboxTimelineBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (item && item.video_id) {
            closeLightbox(); // Tự động đóng/ẩn Lightbox để Timeline hiển thị ngay lập tức lên phía trước!
            openTimeline(item.video_id, item.frame_id);
        }
    });
}

// ==========================================================================
// VIDEO PLAYER (TRÌNH PHÁT VIDEO HTML5 TỰ ĐỘNG SEEK ĐẾN KEYFRAME)
// ==========================================================================
function openVideoPlayer(videoId, frameId, timestampSec) {
    if (!videoId) return;
    currentPlayingVideoId = videoId;
    currentPlayingFrameId = frameId;
    currentPlayingTimestamp = (timestampSec !== undefined && timestampSec !== null) ? Number(timestampSec) : 0;

    const mins = Math.floor(currentPlayingTimestamp / 60);
    const secs = Math.floor(currentPlayingTimestamp % 60);
    const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

    if (videoPlayerTitle) videoPlayerTitle.innerText = `▶️ VIDEO: ${videoId}.mp4`;
    if (videoTimeBadge) videoTimeBadge.innerText = `⏱️ Keyframe: ${timeStr} (${currentPlayingTimestamp}s)`;

    if (html5VideoPlayer) {
        html5VideoPlayer.src = `${VIDEO_STREAM_URL}/${videoId}.mp4`;
        html5VideoPlayer.onloadedmetadata = () => {
            // Tua lùi lại 1.5 giây để người dùng quan sát được hành động trước keyframe
            html5VideoPlayer.currentTime = Math.max(0, currentPlayingTimestamp - 1.5);
            html5VideoPlayer.playbackRate = parseFloat(videoSpeedSelect ? videoSpeedSelect.value : 1.0);
            html5VideoPlayer.play().catch(err => console.log("Autoplay blocked:", err));
        };
    }

    if (videoModal) videoModal.style.display = "flex";
}

function closeVideoPlayer() {
    if (html5VideoPlayer) {
        html5VideoPlayer.pause();
        html5VideoPlayer.src = "";
    }
    if (videoModal) videoModal.style.display = "none";
}

if (videoCloseBtn) videoCloseBtn.addEventListener("click", closeVideoPlayer);

if (videoModal) {
    videoModal.addEventListener("click", (e) => {
        if (e.target === videoModal) closeVideoPlayer();
    });
}

const videoStartBtn = document.getElementById("video-start-btn");
if (videoStartBtn) {
    videoStartBtn.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = 0;
    });
}

if (videoRewind5s) {
    videoRewind5s.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = Math.max(0, html5VideoPlayer.currentTime - 5);
    });
}

if (videoForward5s) {
    videoForward5s.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = html5VideoPlayer.currentTime + 5;
    });
}

if (videoJumpKeyframe) {
    videoJumpKeyframe.addEventListener("click", () => {
        if (html5VideoPlayer) html5VideoPlayer.currentTime = currentPlayingTimestamp;
    });
}

if (videoSpeedSelect) {
    videoSpeedSelect.addEventListener("change", () => {
        if (html5VideoPlayer) html5VideoPlayer.playbackRate = parseFloat(videoSpeedSelect.value);
    });
}

if (videoCopyBtn) {
    videoCopyBtn.addEventListener("click", () => {
        if (currentPlayingVideoId && currentPlayingFrameId) {
            const textToCopy = `${currentPlayingVideoId} ${currentPlayingFrameId}`;
            navigator.clipboard.writeText(textToCopy).then(() => {
                videoTimeBadge.innerText = `✅ Đã chép: ${textToCopy}`;
                setTimeout(() => {
                    const mins = Math.floor(currentPlayingTimestamp / 60);
                    const secs = Math.floor(currentPlayingTimestamp % 60);
                    videoTimeBadge.innerText = `⏱️ Keyframe: ${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')} (${currentPlayingTimestamp}s)`;
                }, 2000);
            });
        }
    });
}

// ==========================================================================
// TIMELINE BROWSER (CHIẾN THUẬT 5 + AUTO-SCROLL & HIGHLIGHT ĐẾN ĐÚNG FRAME)
// ==========================================================================
async function openTimeline(videoId, targetFrameId = null) {
    if (!videoId) return;

    if (timelineVideoTitle) timelineVideoTitle.innerText = `🎞️ TIMELINE: ${videoId}`;
    if (timelineFrameCount) timelineFrameCount.innerText = "Đang tải...";
    if (timelineStrip) timelineStrip.innerHTML = `<div style="padding: 30px; font-weight: bold; width: 100%; text-align: center;">⏳ Đang tải toàn bộ keyframes của video ${videoId}...</div>`;
    if (timelineModal) timelineModal.style.display = "flex";

    try {
        const resp = await fetch(`${TIMELINE_URL}/${videoId}`);
        if (!resp.ok) throw new Error(`Timeline Error: ${resp.statusText}`);

        const data = await resp.json();
        const frames = data.frames || [];

        if (timelineFrameCount) timelineFrameCount.innerText = `${frames.length} keyframes`;

        if (frames.length === 0) {
            timelineStrip.innerHTML = `<div style="padding: 30px; font-weight: bold; width: 100%; text-align: center;">Không tìm thấy keyframe nào của ${videoId}.</div>`;
            return;
        }

        timelineStrip.innerHTML = "";
        let targetElementToScroll = null;

        frames.forEach((f, fIdx) => {
            const itemDiv = document.createElement("div");
            itemDiv.className = "timeline-item";

            // Nếu đây là frame được bấm mở từ thẻ kết quả -> Đánh dấu active
            if (targetFrameId && f.frame_id === targetFrameId) {
                itemDiv.classList.add("timeline-item-active");
                targetElementToScroll = itemDiv;
            }

            // Click vào thẻ keyframe để mở Lightbox soi chi tiết (nổi lên trên tab timeline)
            itemDiv.addEventListener("click", () => {
                openLightbox(fIdx, frames);
            });

            const mins = Math.floor(f.timestamp_sec / 60);
            const secs = Math.floor(f.timestamp_sec % 60);
            const timeStr = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            const asrIcon = f.asr_transcript ? `🎙️ ` : `⏱️ `;

            itemDiv.innerHTML = `
                <div class="timeline-thumb-box" title="🔍 Bấm để soi chi tiết">
                    <img src="${f.image_url}" loading="lazy" alt="Frame ${f.frame_id}">
                    <span class="timeline-time-badge">${asrIcon}${timeStr}</span>
                </div>
                <div class="timeline-item-meta">
                    <span class="timeline-meta-frame">F:${f.frame_id}</span>
                    <span class="timeline-meta-sec">${f.timestamp_sec}s</span>
                </div>
            `;
            timelineStrip.appendChild(itemDiv);
        });

        // TỰ ĐỘNG CUỘN ĐẾN ĐÚNG VỊ TRÍ FRAME ĐÃ CHỌN
        if (targetElementToScroll) {
            setTimeout(() => {
                targetElementToScroll.scrollIntoView({ behavior: "smooth", block: "center" });
            }, 80);
        }

    } catch (err) {
        if (timelineStrip) timelineStrip.innerHTML = `<div style="padding: 30px; color: red; font-weight: bold;">Lỗi tải timeline: ${err.message}</div>`;
        console.error(err);
    }
}

function closeTimeline() {
    if (timelineModal) timelineModal.style.display = "none";
}

if (timelineCloseBtn) timelineCloseBtn.addEventListener("click", closeTimeline);

if (timelineModal) {
    timelineModal.addEventListener("click", (e) => {
        if (e.target === timelineModal) closeTimeline();
    });
}

// Event Listeners cho Lightbox
if (lightboxCloseBtn) lightboxCloseBtn.addEventListener("click", closeLightbox);

if (lightboxToggleBoxes) {
    lightboxToggleBoxes.addEventListener("change", () => {
        drawLightboxCanvas();
    });
}
window.addEventListener("resize", () => {
    if (lightboxModal && lightboxModal.style.display !== "none") {
        drawLightboxCanvas();
    }
});

// Đóng lightbox khi click ra ngoài vùng modal
if (lightboxModal) {
    lightboxModal.addEventListener("click", (e) => {
        if (e.target === lightboxModal) closeLightbox();
    });
}

// Phím tắt bàn phím
window.addEventListener("keydown", (e) => {
    if (videoModal && videoModal.style.display !== "none") {
        if (e.key === "Escape") {
            closeVideoPlayer();
        }
    } else if (lightboxModal && lightboxModal.style.display !== "none") {
        if (e.key === "Escape") {
            closeLightbox();
        }
    } else if (timelineModal && timelineModal.style.display !== "none") {
        if (e.key === "Escape") {
            closeTimeline();
        }
    }
});

// Bắt sự kiện phím Enter cho các ô tìm kiếm
[searchInput, ocrInput, speechInput, videoIdInput, topKNum, visualSimNum].forEach(el => {
    if (el) {
        el.addEventListener("keypress", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                performSearch();
            }
        });
    }
});

if (searchBtn) searchBtn.addEventListener("click", performSearch);
if (multiSearchBtn) multiSearchBtn.addEventListener("click", performMultiSceneSearch);
if (trakeSearchBtn) trakeSearchBtn.addEventListener("click", performTrakeSearch);
if (trakeExportBtn) trakeExportBtn.addEventListener("click", exportTrakeCsv);

// Tự động lưu trạng thái khi người dùng thay đổi bất kỳ tham số nào
[
    objectClassSelect, objectConfSlider,
    weightVisualSlider, weightOcrSlider, weightAsrSlider, weightObjectSlider,
    visualSimSlider, visualSimNum, uniqueVideoCheck,
    topKSlider, topKNum, rerankTopV, rerankKPerVideo,
    trakeLambdaInput, trakeKpathsInput, trakeTopKInput
].forEach(el => {
    if (el) {
        el.addEventListener("input", saveControlPanelState);
        el.addEventListener("change", saveControlPanelState);
    }
});

document.querySelectorAll('input[name="smart-submode"]').forEach(r => {
    r.addEventListener("change", saveControlPanelState);
});

// Khôi phục trạng thái bảng điều khiển từ localStorage khi khởi động
restoreControlPanelState();

// ==========================================================================
// SUBMISSION MANAGER LOGIC & HANDLERS (AIC 2026 CONTEST SUITE)
// ==========================================================================

function switchSubMode(mode) {
    currentSubMode = mode;
    if (subModePills) {
        subModePills.querySelectorAll(".btn-pill").forEach(btn => {
            if (btn.getAttribute("data-mode") === mode) {
                btn.classList.add("active");
            } else {
                btn.classList.remove("active");
            }
        });
    }

    if (mode === "kis") {
        if (subKisQaFields) subKisQaFields.style.display = "block";
        if (subQaFields) subQaFields.style.display = "none";
        if (subTrakeFields) subTrakeFields.style.display = "none";
        if (subHedgingRow) subHedgingRow.style.display = "block";
    } else if (mode === "qa") {
        if (subKisQaFields) subKisQaFields.style.display = "block";
        if (subQaFields) subQaFields.style.display = "block";
        if (subTrakeFields) subTrakeFields.style.display = "none";
        if (subHedgingRow) subHedgingRow.style.display = "block";
    } else if (mode === "trake") {
        if (subKisQaFields) subKisQaFields.style.display = "none";
        if (subQaFields) subQaFields.style.display = "none";
        if (subTrakeFields) subTrakeFields.style.display = "block";
        if (subHedgingRow) subHedgingRow.style.display = "none";
    }
}

if (subModePills) {
    subModePills.querySelectorAll(".btn-pill").forEach(btn => {
        btn.addEventListener("click", () => {
            const mode = btn.getAttribute("data-mode") || "kis";
            switchSubMode(mode);
        });
    });
}

const subHedgingRow = document.getElementById("sub-hedging-row");

async function loadSubmissionDirectory() {
    const dir = subDirPathInput ? subDirPathInput.value.trim() : "query/query-p1/submission";
    if (!dir) return;

    try {
        const response = await fetch(SUBMISSION_LIST_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dir_path: dir })
        });

        if (!response.ok) return;

        const data = await response.json();
        if (data.status === "success" && data.exists && subFileListContainer && subFileItemsContainer) {
            subFileListContainer.style.display = "block";
            subFileCountSpan.innerText = data.file_count;
            subFileItemsContainer.innerHTML = "";

            data.files.forEach(f => {
                const row = document.createElement("div");
                row.className = "sub-file-item-row";
                const typeClass = f.type === "QA" ? "sub-type-qa" : (f.type === "TRAKE" ? "sub-type-trake" : "sub-type-kis");
                row.innerHTML = `
                    <span style="font-weight: 600;">${f.filename}</span>
                    <div style="display: flex; align-items: center; gap: 4px;">
                        <span class="sub-type-badge ${typeClass}">${f.type}</span>
                        <span style="color: #94a3b8;">${f.line_count}d</span>
                    </div>
                `;
                row.addEventListener("click", () => {
                    if (subFileNameInput) subFileNameInput.value = f.filename;
                    showToast(`📄 Đã chọn file: ${f.filename}`, "info", 1500);
                });
                subFileItemsContainer.appendChild(row);
            });
        }
    } catch (e) {
        console.warn("Could not load submission directory:", e);
    }
}

if (btnSubRefreshDir) {
    btnSubRefreshDir.addEventListener("click", () => {
        loadSubmissionDirectory();
        showToast("🔄 Đã quét lại thư mục submission", "info", 1500);
    });
}

if (subDirPathInput) {
    subDirPathInput.addEventListener("change", () => {
        saveControlPanelState();
        loadSubmissionDirectory();
    });
}

async function performSubmissionFill() {
    const vId = subVideoIdInput ? subVideoIdInput.value.trim() : "";
    if (!vId && currentSubMode !== "trake") {
        showToast("⚠️ Vui lòng nhập Video ID (Ví dụ: L21_V015)", "warning");
        return;
    }

    const fA = subFrameAInput ? parseInt(subFrameAInput.value) : null;
    const fB = subFrameBInput ? parseInt(subFrameBInput.value) : null;
    const fA2 = subFrameA2Input ? parseInt(subFrameA2Input.value) : null;
    const fB2 = subFrameB2Input ? parseInt(subFrameB2Input.value) : null;
    const answer = subQaAnswerInput ? subQaAnswerInput.value.trim() : "";

    let secondInterval = null;
    if (!isNaN(fA2) && !isNaN(fB2) && fA2 !== null && fB2 !== null) {
        secondInterval = [fA2, fB2];
    }

    let trakeFrames = null;
    if (currentSubMode === "trake") {
        const rawT = subTrakeFramesInput ? subTrakeFramesInput.value.trim() : "";
        if (!rawT) {
            showToast("⚠️ Vui lòng nhập chuỗi Frame IDs cho các sự kiện TRAKE", "warning");
            return;
        }
        trakeFrames = rawT.split(/[\s,]+/).filter(x => x).map(x => parseInt(x)).filter(x => !isNaN(x));
        if (trakeFrames.length < 2) {
            showToast("⚠️ Chuỗi TRAKE phải chứa ít nhất 2 Frame IDs", "warning");
            return;
        }
    }

    // Thu thập System Backup Items từ kết quả tìm kiếm hiện tại nếu bật Risk Hedging
    let backupItems = null;
    if (subEnableHedgingCheck && subEnableHedgingCheck.checked && currentResults && currentResults.length > 0) {
        backupItems = currentResults
            .filter(r => r.video_id !== vId)
            .map(r => ({ video_id: r.video_id, frame_id: r.frame_id }));
    }

    if (btnSubFillPreview) {
        btnSubFillPreview.disabled = true;
        btnSubFillPreview.innerText = "⏳ ĐANG TÍNH...";
    }

    try {
        const response = await fetch(SUBMISSION_FILL_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                mode: currentSubMode,
                video_id: vId,
                frame_a: fA,
                frame_b: fB,
                second_interval: secondInterval,
                answer: answer,
                trake_event_frames: trakeFrames,
                system_backup_items: backupItems,
                target_count: 100
            })
        });

        if (!response.ok) {
            throw new Error(`Fill Error: ${response.statusText}`);
        }

        const data = await response.json();
        if (data.status === "success") {
            latestGeneratedSubmissionLines = data.lines || [];
            if (subPreviewContainer && subPreviewCode && subPreviewTotalSpan) {
                subPreviewContainer.style.display = "block";
                subPreviewTotalSpan.innerText = data.total_lines;
                subPreviewCode.innerText = (data.preview || []).join("\n") + (data.total_lines > 5 ? `\n... (+${data.total_lines - 5} dòng khác)` : "");
                if (subPreviewStatusSpan) {
                    subPreviewStatusSpan.innerText = `✅ Đã sinh ${data.total_lines} dòng (${data.mode.toUpperCase()})`;
                }
            }
            showToast(`🎯 Đã sinh thành công ${data.total_lines} dòng submission theo chiến thuật!`, "success");
        } else {
            showToast(`Lỗi: ${data.message}`, "error");
        }
    } catch (err) {
        showToast(`Lỗi fill submission: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (btnSubFillPreview) {
            btnSubFillPreview.disabled = false;
            btnSubFillPreview.innerText = "🎯 FILL & XEM";
        }
    }
}

if (btnSubFillPreview) {
    btnSubFillPreview.addEventListener("click", performSubmissionFill);
}

async function saveSubmissionFile() {
    if (!latestGeneratedSubmissionLines || latestGeneratedSubmissionLines.length === 0) {
        showToast("⚠️ Chưa có nội dung. Vui lòng bấm '🎯 FILL & XEM' trước khi lưu.", "warning");
        return;
    }

    let fname = subFileNameInput ? subFileNameInput.value.trim() : "";
    if (!fname) {
        showToast("⚠️ Vui lòng nhập tên file CSV (Ví dụ: query-p1-1-kis.csv)", "warning");
        return;
    }
    if (!fname.toLowerCase().endsWith(".csv")) {
        fname += ".csv";
        if (subFileNameInput) subFileNameInput.value = fname;
    }

    const dir = subDirPathInput ? subDirPathInput.value.trim() : "query/query-p1/submission";
    const fullPath = `${dir}/${fname}`.replace(/\/+/g, "/");

    if (btnSubSaveFile) {
        btnSubSaveFile.disabled = true;
        btnSubSaveFile.innerText = "⏳ ĐANG LƯU...";
    }

    try {
        const response = await fetch(SUBMISSION_SAVE_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                file_path: fullPath,
                lines: latestGeneratedSubmissionLines
            })
        });

        if (!response.ok) throw new Error(response.statusText);

        const data = await response.json();
        if (data.status === "success") {
            showToast(`💾 Đã lưu thành công ${data.line_count} dòng vào ${fname}!`, "success", 3000);
            loadSubmissionDirectory();
        } else {
            showToast(`Lỗi: ${data.message}`, "error");
        }
    } catch (err) {
        showToast(`Lỗi lưu file: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (btnSubSaveFile) {
            btnSubSaveFile.disabled = false;
            btnSubSaveFile.innerText = "💾 LƯU FILE CSV";
        }
    }
}

if (btnSubSaveFile) {
    btnSubSaveFile.addEventListener("click", saveSubmissionFile);
}

async function exportSubmissionZip() {
    const dir = subDirPathInput ? subDirPathInput.value.trim() : "query/query-p1/submission";
    let zipOut = subZipOutputPathInput ? subZipOutputPathInput.value.trim() : "query/submission_p1.zip";
    if (zipOut && !zipOut.toLowerCase().endsWith(".zip")) {
        zipOut += ".zip";
        if (subZipOutputPathInput) subZipOutputPathInput.value = zipOut;
    }

    if (btnSubExportZip) {
        btnSubExportZip.disabled = true;
        btnSubExportZip.innerText = "⏳ ĐANG ĐÓNG GÓI ZIP...";
    }

    try {
        const response = await fetch(SUBMISSION_ZIP_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                submission_dir: dir,
                output_zip_path: zipOut
            })
        });

        if (!response.ok) throw new Error(response.statusText);

        const data = await response.json();
        if (data.status === "success") {
            showToast(`📦 ĐÃ XUẤT THÀNH CÔNG ${data.file_count} FILES VÀO ${zipOut} (${data.file_size_kb} KB)!`, "success", 4000);
        } else {
            showToast(`Lỗi: ${data.message}`, "error");
        }
    } catch (err) {
        showToast(`Lỗi xuất zip: ${err.message}`, "error");
        console.error(err);
    } finally {
        if (btnSubExportZip) {
            btnSubExportZip.disabled = false;
            btnSubExportZip.innerText = "📦 XUẤT SUBMISSION.ZIP (CHUẨN BTC)";
        }
    }
}

if (btnSubExportZip) {
    btnSubExportZip.addEventListener("click", exportSubmissionZip);
}

// Nạp nhanh Keyframe từ Lightbox vào Submission Manager
if (lightboxSubBtn) {
    lightboxSubBtn.addEventListener("click", () => {
        const item = activeResultsSource[currentLightboxIndex];
        if (!item) return;

        if (subVideoIdInput) subVideoIdInput.value = item.video_id;
        if (subFrameAInput) subFrameAInput.value = item.frame_id;
        if (subFrameBInput && !subFrameBInput.value) {
            subFrameBInput.value = item.frame_id + 300;
        }

        // Mở card Submission nếu đang bị thu gọn
        if (cardSubmission && cardSubmission.classList.contains("is-collapsed")) {
            setCardCollapsed(cardSubmission, false);
        }

        // Cuộn tới Card Submission
        if (cardSubmission) {
            cardSubmission.scrollIntoView({ behavior: "smooth", block: "center" });
        }

        showToast(`📥 Đã nạp ${item.video_id} frame ${item.frame_id} vào Submission Manager!`, "success", 2000);
    });
}

// ==========================================================================
// MULTI-VIEW IMAGE GALLERY & CLIPBOARD PASTE LISTENER
// ==========================================================================
let currentImageQueries = []; // Danh sách các ảnh base64 đa góc nhìn

function addImageQuery(base64Data) {
    if (!base64Data) return;
    currentImageQueries.push(base64Data);
    currentImageQueryBase64 = currentImageQueries[0];
    renderImageGallery();
    if (cardImageQuery && cardImageQuery.classList.contains("is-collapsed")) {
        setCardCollapsed(cardImageQuery, false);
    }
    showToast(`📸 Đã nạp góc ảnh #${currentImageQueries.length} thành công!`, "info");
}

function removeImageQuery(idx) {
    if (idx >= 0 && idx < currentImageQueries.length) {
        currentImageQueries.splice(idx, 1);
        currentImageQueryBase64 = currentImageQueries.length > 0 ? currentImageQueries[0] : null;
        renderImageGallery();
        showToast("🗑️ Đã xóa góc ảnh.", "info");
    }
}

function clearAllImageQueries() {
    currentImageQueries = [];
    currentImageQueryBase64 = null;
    renderImageGallery();
    if (imageFileInput) imageFileInput.value = "";
    showToast("🗑️ Đã xóa toàn bộ ảnh.", "info");
}

function renderImageGallery() {
    const gallery = document.getElementById("image-gallery-container");
    const placeholder = document.getElementById("image-dropzone-placeholder");
    const actionsBar = document.getElementById("image-actions-bar");
    const badgeCount = document.getElementById("badge-image-count");

    if (!gallery || !placeholder) return;

    gallery.innerHTML = "";
    if (currentImageQueries.length === 0) {
        gallery.style.display = "none";
        placeholder.style.display = "block";
        if (actionsBar) actionsBar.style.display = "none";
        if (badgeCount) badgeCount.style.display = "none";
    } else {
        gallery.style.display = "grid";
        placeholder.style.display = "none";
        if (actionsBar) actionsBar.style.display = "flex";
        if (badgeCount) {
            badgeCount.style.display = "inline-block";
            badgeCount.innerText = `${currentImageQueries.length} ảnh`;
        }

        currentImageQueries.forEach((b64, idx) => {
            const thumbWrap = document.createElement("div");
            thumbWrap.style.cssText = "position: relative; border-radius: 6px; border: 1px solid #6366f1; background: #0f172a; overflow: hidden; height: 80px; display: flex; align-items: center; justify-content: center;";
            
            thumbWrap.innerHTML = `
                <img src="${b64}" style="max-width: 100%; max-height: 100%; object-fit: cover;" alt="Góc ảnh #${idx + 1}">
                <span style="position: absolute; bottom: 2px; left: 2px; background: rgba(0,0,0,0.75); color: #fff; font-size: 0.65rem; padding: 1px 4px; border-radius: 3px; font-weight: 700;">#${idx + 1}</span>
                <button type="button" onclick="event.stopPropagation(); removeImageQuery(${idx})" style="position: absolute; top: 2px; right: 2px; background: rgba(239, 68, 68, 0.9); color: white; border: none; border-radius: 50%; width: 18px; height: 18px; cursor: pointer; font-size: 0.65rem; line-height: 1; display: flex; align-items: center; justify-content: center;" title="Xóa góc ảnh này">✕</button>
            `;
            gallery.appendChild(thumbWrap);
        });
    }
}
window.removeImageQuery = removeImageQuery;
window.clearAllImageQueries = clearAllImageQueries;

// Global Clipboard Paste Listener (Ctrl + V)
window.addEventListener("paste", (e) => {
    const items = (e.clipboardData || (e.originalEvent && e.originalEvent.clipboardData))?.items;
    if (!items) return;
    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        if (item.kind === 'file' && item.type.startsWith('image/')) {
            const blob = item.getAsFile();
            const reader = new FileReader();
            reader.onload = (event) => {
                addImageQuery(event.target.result);
            };
            reader.readAsDataURL(blob);
        }
    }
});

// Dropzone file selection & drag-and-drop
if (imageDropzone) {
    imageDropzone.addEventListener("click", (e) => {
        if (e.target.tagName === "BUTTON") return;
        if (imageFileInput) imageFileInput.click();
    });

    imageDropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        imageDropzone.style.borderColor = "#6366f1";
        imageDropzone.style.background = "rgba(99, 102, 241, 0.15)";
    });

    imageDropzone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        imageDropzone.style.borderColor = "#475569";
        imageDropzone.style.background = "rgba(15, 23, 42, 0.5)";
    });

    imageDropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        imageDropzone.style.borderColor = "#475569";
        imageDropzone.style.background = "rgba(15, 23, 42, 0.5)";
        if (e.dataTransfer && e.dataTransfer.files) {
            Array.from(e.dataTransfer.files).forEach(file => {
                if (file.type.startsWith("image/")) {
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        addImageQuery(event.target.result);
                    };
                    reader.readAsDataURL(file);
                }
            });
        }
    });
}

if (imageFileInput) {
    imageFileInput.addEventListener("change", (e) => {
        if (e.target.files) {
            Array.from(e.target.files).forEach(file => {
                const reader = new FileReader();
                reader.onload = (event) => {
                    addImageQuery(event.target.result);
                };
                reader.readAsDataURL(file);
            });
        }
    });
}

const btnAddMoreImage = document.getElementById("btn-add-more-image");
if (btnAddMoreImage) {
    btnAddMoreImage.addEventListener("click", (e) => {
        e.stopPropagation();
        if (imageFileInput) imageFileInput.click();
    });
}

const btnClearAllImages = document.getElementById("btn-clear-all-images");
if (btnClearAllImages) {
    btnClearAllImages.addEventListener("click", (e) => {
        e.stopPropagation();
        clearAllImageQueries();
    });
}

// Polymorphic Image Search Action (Kế thừa và đa hình 100% từ performSearch)
async function executeImageSearch() {
    return performSearch();
}

if (btnSearchImageQuery) {
    btnSearchImageQuery.addEventListener("click", executeImageSearch);
}
window.executeImageSearch = executeImageSearch;

// Tag UI Helper Functions (Hỗ trợ thêm nhiều video ID / Prefix đồng thời)
function addIncludeTokens(rawText) {
    if (!rawText) return;
    const tokens = rawText.split(/[,;\s\n\t]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    let changed = false;
    tokens.forEach(tok => {
        if (!currentIncludeTags.includes(tok)) {
            currentIncludeTags.push(tok);
            changed = true;
        }
    });
    if (changed) renderIncludeTags();
}

function addExcludeTokens(rawText) {
    if (!rawText) return;
    const tokens = rawText.split(/[,;\s\n\t]+/).map(t => t.trim().toUpperCase()).filter(Boolean);
    let changed = false;
    tokens.forEach(tok => {
        if (!currentExcludeTags.includes(tok)) {
            currentExcludeTags.push(tok);
            changed = true;
        }
    });
    if (changed) renderExcludeTags();
}

function commitPendingTags() {
    if (includeVideoInput && includeVideoInput.value.trim()) {
        addIncludeTokens(includeVideoInput.value);
        includeVideoInput.value = "";
    }
    if (excludeVideoInput && excludeVideoInput.value.trim()) {
        addExcludeTokens(excludeVideoInput.value);
        excludeVideoInput.value = "";
    }
}
window.commitPendingTags = commitPendingTags;

function renderIncludeTags() {
    if (!includeTagsContainer) return;
    includeTagsContainer.innerHTML = "";
    currentIncludeTags.forEach((tag, idx) => {
        const span = document.createElement("span");
        span.className = "tag-item tag-include";
        span.innerHTML = `<span>${tag}</span><span class="tag-remove-btn" title="Xóa" onclick="removeIncludeTag(${idx})">&times;</span>`;
        includeTagsContainer.appendChild(span);
    });
}

function removeIncludeTag(idx) {
    currentIncludeTags.splice(idx, 1);
    renderIncludeTags();
}
window.removeIncludeTag = removeIncludeTag;

function renderExcludeTags() {
    if (!excludeTagsContainer) return;
    excludeTagsContainer.innerHTML = "";
    currentExcludeTags.forEach((tag, idx) => {
        const span = document.createElement("span");
        span.className = "tag-item tag-exclude";
        span.innerHTML = `<span>${tag}</span><span class="tag-remove-btn" title="Xóa" onclick="removeExcludeTag(${idx})">&times;</span>`;
        excludeTagsContainer.appendChild(span);
    });
}

function removeExcludeTag(idx) {
    currentExcludeTags.splice(idx, 1);
    renderExcludeTags();
    // Đồng bộ trạng thái mờ/sáng trên lưới kết quả
    applyExcludeStateToGrid();
}
window.removeExcludeTag = removeExcludeTag;

if (includeVideoInput) {
    includeVideoInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            addIncludeTokens(includeVideoInput.value);
            includeVideoInput.value = "";
        }
    });
    includeVideoInput.addEventListener("paste", () => {
        setTimeout(() => {
            addIncludeTokens(includeVideoInput.value);
            includeVideoInput.value = "";
        }, 30);
    });
}

if (excludeVideoInput) {
    excludeVideoInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            addExcludeTokens(excludeVideoInput.value);
            excludeVideoInput.value = "";
        }
    });
    excludeVideoInput.addEventListener("paste", () => {
        setTimeout(() => {
            addExcludeTokens(excludeVideoInput.value);
            excludeVideoInput.value = "";
        }, 30);
    });
}

// Client-side quick filter on Top K (Lọc an toàn từ currentResults gốc, hoàn tác 100% khi xóa bộ lọc)
if (filterVideoResultsInput) {
    filterVideoResultsInput.addEventListener("input", () => {
        const rawFilter = filterVideoResultsInput.value.trim().toLowerCase();
        if (!currentResults || currentResults.length === 0) return;
        
        if (!rawFilter) {
            resultCount.innerText = `${currentResults.length} items found`;
            renderResults(currentResults);
            return;
        }

        const tokens = rawFilter.split(/[,;\s]+/).map(t => t.trim()).filter(Boolean);
        if (tokens.length === 0) {
            resultCount.innerText = `${currentResults.length} items found`;
            renderResults(currentResults);
            return;
        }

        const posTokens = tokens.filter(t => !t.startsWith("-"));
        const negTokens = tokens.filter(t => t.startsWith("-")).map(t => t.slice(1)).filter(Boolean);

        const filtered = currentResults.filter(item => {
            const vid = (item.video_id || "").toLowerCase();
            const fid = String(item.frame_id || "").toLowerCase();
            const fullId = `${vid}_${fid}`;

            // Kiểm tra loại trừ
            for (const nt of negTokens) {
                if (vid.includes(nt) || fid.includes(nt) || fullId.includes(nt)) {
                    return false;
                }
            }

            if (posTokens.length === 0) return true;

            // Kiểm tra bao gồm (khớp ít nhất 1 trong các posTokens)
            return posTokens.some(pt => vid.includes(pt) || fid.includes(pt) || fullId.includes(pt));
        });

        resultCount.innerText = `${filtered.length} / ${currentResults.length} items (đang lọc)`;
        renderResults(filtered);
    });
}

// Nạp danh sách Danh Mục Chủ Đề từ Backend
async function loadCategories() {
    try {
        const res = await fetch(CATEGORIES_URL);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "success" && data.taxonomy) {
            categoriesTaxonomy = data.taxonomy;
            if (categorySelect) {
                categorySelect.innerHTML = "";
                for (const [catKey, catInfo] of Object.entries(categoriesTaxonomy)) {
                    const opt = document.createElement("option");
                    opt.value = catKey;
                    opt.innerText = catInfo.label || catKey;
                    categorySelect.appendChild(opt);
                }
            }
        }
    } catch (e) {
        console.warn("Could not load categories", e);
    }
}

if (categorySelect) {
    categorySelect.addEventListener("change", () => {
        const catKey = categorySelect.value;
        const catInfo = categoriesTaxonomy[catKey];
        if (catInfo && catInfo.sub && Object.keys(catInfo.sub).length > 0) {
            if (groupSubcategory) groupSubcategory.style.display = "block";
            if (subcategorySelect) {
                subcategorySelect.innerHTML = "";
                for (const [subKey, subLabel] of Object.entries(catInfo.sub)) {
                    const opt = document.createElement("option");
                    opt.value = subKey;
                    opt.innerText = subLabel;
                    subcategorySelect.appendChild(opt);
                }
            }
        } else {
            if (groupSubcategory) groupSubcategory.style.display = "none";
            if (subcategorySelect) subcategorySelect.innerHTML = '<option value="all">--- Tất cả ---</option>';
        }
    });
}

// Nạp danh sách 80 lớp COCO cho Object Filter Dropdown
async function loadCocoClasses() {
    try {
        const res = await fetch(COCO_URL);
        if (!res.ok) return;
        const data = await res.json();
        if (data.status === "success" && data.classes && objectClassSelect) {
            const currentVal = objectClassSelect.value;
            objectClassSelect.innerHTML = "";
            data.classes.forEach((clsName, idx) => {
                const opt = document.createElement("option");
                opt.value = idx;
                opt.innerText = `${idx}: ${clsName}`;
                objectClassSelect.appendChild(opt);
            });
            if (currentVal !== undefined && currentVal !== "") {
                objectClassSelect.value = currentVal;
            }
        }
    } catch (e) {
        console.warn("Could not load COCO classes", e);
    }
}

// Tự động nạp ban đầu khi khởi động web
setTimeout(loadSubmissionDirectory, 800);
setTimeout(loadCocoClasses, 300);
setTimeout(loadCategories, 400);





