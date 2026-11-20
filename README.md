# BazarSetu V3.0 - AI/ML System 01: Product Image Intelligence

> High-throughput, low-latency marketplace image processing optimized for **Dual NVIDIA Tesla T4 GPUs (Turing Compute 7.5, FP16)** on Kaggle.

---

## 1. System 01 Pipeline Architecture (Stages A–H)

```
[ Raw Smartphone Upload ] (JPEG / PNG / WebP)
          │
          ▼
   Stage A: Ingestion ──> Stored in Immutable Product Master (products/{id}/original/)
          │
          ▼
   Stage B: Quality Gate ──> Checks Laplacian Sharpness, Exposure, Framing
          │                  (Emits: BLURRY, TOO_DARK, TOO_BRIGHT, TOO_SMALL, CUT_OFF_FRAME)
          ▼
   Stage C & D: BiRefNet-HR ──> Aspect-Ratio Safe Letterboxed Matting (FP16 on cuda:0 / cuda:1)
          │                      (Fallback: SAM 2 Single-Tap Interactive Refinement)
          ▼
   Stage E: Product Master ──> Immutable Write-Protected Asset Bundle
          │
          ▼
   Stage F: Deterministic Polish ──> Auto White Balance + Bilateral Denoise + Unsharp Mask
          │                          (Governor: Enforces Delta E00 < 3.0 Color Fidelity)
          ▼
   Stage G: Rendering & Grounding ──> 1200x1200 Canvas (72% Dominance) + Anchored Contact Shadow
          │
          ▼
   Stage H: Fidelity Gate ──> Validates CIEDE2000 Delta E00, Boundary F1, and Mask IoU
          │
          ▼
[ Final Marketplace Hero & Thumbnail WebP ] ──> Ready for Seller Approval ("Sahi Hai")
```

---

## 2. Quantitative Acceptance Targets (Definition of Done)

| Metric | Target Gate | Purpose |
| :--- | :--- | :--- |
| **Segmentation Mask IoU** | $\ge 0.94$ | Accurate foreground extraction on cluttered backgrounds |
| **Boundary F1-Score** | $\ge 0.92$ | Preservation of thin jewellery, handloom threads, and filigree |
| **Color Fidelity ($\Delta E_{00}$)** | $< 3.0$ | Strict color preservation of natural dyes, terracotta, and labels |
| **GPU Inference Latency** | $< 400\text{ ms}$ / image | High-throughput FP16 execution on Tesla T4 |
| **Total Turnaround Time** | $< 3.5\text{ s}$ end-to-end | Finished before seller completes voice note recording |

---

## 3. Kaggle Setup Instructions (Dual Tesla T4)

### Step 1: Enable Dual GPUs in Kaggle
In your Kaggle Notebook settings (right sidebar):
* **Accelerator:** `GPU T4 x 2`
* **Internet:** `On`

### Step 2: Run Setup
In a notebook cell:
```bash
!bash kaggle/setup_kaggle.sh
```

### Step 3: Run Inference / Batch Processing
```bash
!python kaggle/run_kaggle.py --input_images /path/to/photo1.jpg /path/to/photo2.jpg --product_id my_artisan_pot
```

### Step 4: Run Held-Out Benchmark Evaluation
```bash
!python eval/benchmark_runner.py --data_dir /kaggle/input/bazarsetu-benchmark/
```

### Step 5: Start Live API Server & Tunnel for Amar's Flutter App
```bash
!python kaggle/start_api_tunnel.py
```
This automatically starts the FastAPI server and launches a free Cloudflare tunnel (`trycloudflare.com`). It will print the public HTTPS URL to copy into Amar's Flutter `.env` (`BPP_BACKEND_URL`).

---

## 4. Directory Structure

```
V3.0/
├── config.py                 # Configuration parameters, thresholds, and paths
├── core/
│   ├── device.py             # Dual Tesla T4 stream manager and FP16 enforcement
│   └── product_master.py     # Product Master directory layout, immutability, audit JSON
├── pipeline/
│   ├── stage_a_ingestion.py  # Stage A: File validation, hashing, raw storage
│   ├── stage_b_quality.py    # Stage B: Laplacian blur, exposure, framing QA gate
│   ├── stage_cd_segmentation.py # Stage C & D: BiRefNet-HR aspect-ratio-safe segmentation
│   ├── stage_d_sam2.py       # Stage D: SAM 2 single-tap interactive refinement fallback
│   ├── stage_f_polish.py     # Stage F: Deterministic polish & Delta E00 governor
│   ├── stage_g_rendering.py  # Stage G: 1200x1200 white canvas + mathematical contact shadow
│   ├── stage_h_fidelity.py   # Stage H: Automated Delta E00 & boundary verification gate
│   └── orchestrator.py       # Full Stage A-H coordinator with dual-GPU batch execution
├── eval/
│   ├── metrics.py            # IoU, Boundary F1, and CIEDE2000 implementations
│   └── benchmark_runner.py   # Runs held-out benchmark suite and produces DOD pass/fail table
├── kaggle/
│   ├── setup_kaggle.sh       # Environment initialization for Kaggle Dual T4
│   └── run_kaggle.py         # Kaggle entrypoint for batch image processing
├── requirements.txt
├── System1_DOD.txt
└── V3_plan.txt
```

