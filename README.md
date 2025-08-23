# **Visual Damage Assessment Pipeline (Secure Future Insurance)**

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com)
[![Version](https://img.shields.io/badge/version-v0.2.0--alpha-blue)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](./LICENSE)

**Visual Damage Assessment Pipeline** is a full-stack ML system built to automate car damage assessment for the insurance company "Secure Future." It turns photos of damage into ready-to-use reports in minutes instead of days.

## Business Context: from days to 5 minutes

Traditional car damage assessment in insurance is slow and expensive: it requires an adjuster to visit, manually inspect, and fill out paperwork, taking anywhere from 24 hours to several days. This causes unhappy customers, churn, and high operational costs.

**Goal of this project** — reduce the time from claim submission to a first damage report **from several days to under 5 minutes**.

### 📈 Key Business Metrics (KPIs)

-   **⭐ North Star Metric**: **Retention Rate**– happy customers who get quick resolutions stay with the company.
-   **⏱️ Operational Metrics**:
    -   **Time-to-First-Decision (TTFD)**: Goal < 5 minutes.
    -   **First-Pass Automation Rate (FPAR)**: % of cases handled without human intervention.
    -   **Adjuster Productivity**: Increase cases per adjuster via routine automation.
-   **💰 Revenue & Quality Metrics**:
    -   **Error/Discrepancy Rate**: Minimize differences between automated and manual estimates.
    -   **Churn Cost Avoidance**: Save money by keeping customers.

---

## System Architecture

The system is implemented as a FastAPI service wrapping a complex ML pipeline. Key feature — **LLM-powered report synthesis**, where a large language model (DeepSeek) doesn’t just summarize, but corrects and enriches CV model outputs.

```mermaid
graph TD
    subgraph "API Endpoint: /assess_damage"
        A[POST: Images + Vehicle Metadata] --> B{Parallel Inference}
    end

    subgraph "Computer Vision Core"
        B -- Full Image --> C[Part Segmentation (YOLOv8-Seg)]
        B -- Full Image --> D[Damage Detection (YOLOv8-Det)]
    end

    subgraph "Logic & Enrichment Layer"
        C --> E{Damage-Part Association}
        D --> E
        E --> F[Tiered Cost Estimation:
        1. CSV Match (Make/Model/Year)
        2. CSV Fallback (Global Mean)
        3. LLM Estimate Needed]
    end

    subgraph "Report Generation Layer"
        F --> G{LLM Synthesis & Correction}
        A --> G
        G --> H[Final PDF/JSON Report]
    end

    style G fill:#f9f,stroke:#333,stroke-width:2px
```

### How It Works: Step-by-Step explanation

1.  **Data Intake**: User uploads 1–N photos + metadata (make, model, year, VIN) to `/assess_damage`.
2.  **Parallel Inference**:
    -   `parts_model` (YOLOv11-Seg)finds car part masks.
    -   `damage_model` (YOLOv11-Det)finds damage bounding boxes.
3.  **Matching**:Algorithm links each damage instance to a part (e.g., “dent on front door”).
4.  **Tiered Cost Estimation**:
    -   **Level 1**: Look up exact match in `damage_costs.csv` by make, model, year, and damage type.
    -   **Level 2 (Fallback)**: If no exact match, use average cost for that damage type/part.
    -   **Level 3 (LLM)**: If no data, pass assessment task to the LLM.
5.  **LLM Report Synthesis**: Real critical step. Raw CV outputs go to DeepSeek with a prompt to:
    -   **Act as an expert adjuster.**.
    -   **Correct physically impossible detections** (like “dent on front light → “crack on front light”).
    -   **Format the report** in supported languages (English, Italian, Russian, etc.).
6.  **PDF Generation**: Professional PDF is created from LLM output + annotated images.

---
## 🚀 API: How to use me

The system runs as a uvicorn server. Example call with `curl`:

```bash
# start
uvicorn app:app --reload

# example
curl -X POST "http://127.0.0.1:8000/assess_damage" \
-H "Content-Type: multipart/form-data" \
-F "images=@/path/to/your/image1.jpg" \
-F "images=@/path/to/your/image2.jpg" \
-F "make=Kia" \
-F "model=Sportage" \
-F "year=2022" \
-F "vin=IT12345" \
-F "first_name=John" \
-F "last_name=Doe" \
-F "generate_llm_report_flag=true" \
-F "language=en"
```

---

## Training Strategy & Data

Models were trained on a hybrid dataset (CarDD, Roboflow, proprietary images). To maximize quality with minimal labeling, a **self-training loop** was used::
1.  Train on 40% labeled data.
2.  Infer on 60% unlabeled data.
3.  Add confident (>0.7) predictions (pseudo-labels) to training set.
4.  **Result**: **IoU > 90%** on dev set.

---

## Tech Stack

-   **Backend**: FastAPI, Pydantic
-   **ML Frameworks**: PyTorch, Ultralytics (YOLOv8)
-   **LLM**: ChatGPT/DeepSeek (OpenAI-compatible client)
-   **Data**: Pandas
-   **Deployment**: Uvicorn, Docker (scheduled)

---
## Roadmap & Current Limitations

**Status**: **Alpha**. It is functional,still under internal testing.

### Work In Progress:
-   3D visualization integration via COLMAP.
-   Expand `damage_costs.csv` for more model coverage.

### Future Plans (per design doc):
-   **Add more classes**: new parts (e.g, fender, pillar, chassis).
-   **Increase recall**: improve small damage detection (crack, scratch).
-   **Handle rare cases**: strategies for rare cars and custom tuning.
-   **Human-in-the-loop**: interface for adjusters to correct reports; corrections used for retraining.
-   **Video support**: slice video into keyframes for assessment.