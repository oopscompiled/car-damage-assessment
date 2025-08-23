# start: uvicorn app:app --reload
import os
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, Form, File
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import asyncio
from typing import Literal, List

app = FastAPI()

from src.detection.detector import Detector
from src.segmentation.segmenter import Segmenter
from src.utils import build_damage_summary, text_prepare, Inference
from src.pdf_utils import generate_pdf_report

LANGUAGE_MAP = {
    "en": "English",
    "ru": "Russian",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ja": "Japanese",
    "ar": "Arabic"
}
LanguageCode = Literal["en", "ru", "es", "fr", "de", "it", "pt", "zh", "ja", "ar"]

damage_model = Detector('/Users/pacuk/Documents/clean-repo/models/yolo-detect-0.54-960px.pt')
parts_model = Segmenter('/Users/pacuk/Documents/clean-repo/models/yolo-seg-0.8-1280px.pt')

damage_costs = pd.read_csv('/Users/pacuk/Documents/clean-repo/data/damage_costs.csv')
fallback_means = (
    damage_costs
    .groupby(['damage_type', 'part'])['average_cost_usd']
    .mean()
    .reset_index()
)

os.makedirs("report", exist_ok=True)
os.makedirs("annotated", exist_ok=True)


class UserData(BaseModel):
    make: str
    model: str
    year: int = Field(..., description="Year of manufacture")
    vin: str = "UNKNOWN"
    first_name: str = "John"
    last_name: str = "Doe"  

    @field_validator("year")
    def validate_year(cls, v):
        current_year = datetime.now().year
        if v < 1900 or v > current_year + 1:
            raise ValueError(f"Year must be between 1900 and {current_year+1}, got {v}")
        return v


# === UTILS ===
def get_estimated_cost(damage_type, part, user_data: UserData):
    filtered = damage_costs[
        (damage_costs['make'] == user_data.make.lower()) &
        (damage_costs['model'] == user_data.model.lower()) &
        (damage_costs['model_year'] == user_data.year)
    ]
    match = filtered[
        (filtered['damage_type'] == damage_type) &
        (filtered['part'] == part)
    ]
    if not match.empty:
        avg = match['average_cost_usd'].mean()
        return round(avg, 2), round(avg * 0.85, 2), round(avg * 1.15, 2)

    fallback = fallback_means[
        (fallback_means['damage_type'] == damage_type) &
        (fallback_means['part'] == part)
    ]
    if not fallback.empty:
        avg = fallback['average_cost_usd'].values[0]
        return round(avg, 2), round(avg * 0.85, 2), round(avg * 1.15, 2)

    return None, None, None


def generate_llm_report(
    user_data: UserData,
    language: str = "en",
    language_full: str = "English",
    case_id: str = None
) -> str:
    
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url="https://api.deepseek.com"
    )

    if case_id is None:
        case_id = f"{user_data.first_name}_{user_data.last_name}_{user_data.vin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_path = f"report/report_{language}_{case_id}.txt"

    if not os.path.exists(report_path):
        raise FileNotFoundError(f"{report_path} not found. Make sure detection step ran before LLM.")

    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""
        You are a professional automotive damage report writer with expert knowledge of vehicle parts and materials.
        Your task is to create a clear, accurate, and physically plausible damage report.

        Write the report in {language_full} language.

        Format rules:
        - Title: "Vehicle Damage Assessment Report"
        - Then "Report Date: <date>"
        - Then "Prepared for: {user_data.first_name} {user_data.last_name}"
        - Then "Vehicle Information" as bullet points
        - Then "Detected Damages" (one bullet per damage, with confidence, IoU, estimated cost, and "See attached image").
        - At the end add "Summary of Estimated Repair Costs":
            * bullet list of part: cost
            * final line "Total Estimated Repair Cost: $<sum>"

        Processing and Correction Rules:
        - You will receive a list of damages detected by a computer vision model. This data may contain errors.
        - **Critically analyze each detected damage. If a damage type is physically impossible for a specific part (e.g., a "dent" on a glass or plastic part like a headlight or window), you MUST correct it to a plausible damage type.**
        - For example, if the input is "Dent in front light", you should report it as "Cracked front light", "Broken front light", or "Scratched front light". Use your expertise to choose the most likely term.
        - Do NOT invent damages if none are detected. Only correct the description of existing, but poorly described, damages.

        If any price is missing, estimate a realistic repair cost.

        Report Date: {current_time}
        Vehicle Information:
        - Make: {user_data.make}
        - Model: {user_data.model}
        - Year: {user_data.year}
        - VIN: {user_data.vin}

        Detected Damages (from CV models, requires your expert correction):
        {report_content}
        """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"You are a report generator for vehicle damage assessments. Output in {language_full}."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()


# === MAIN ENDPOINT ===
@app.post("/assess_damage")
async def assess_damage(
    images: List[UploadFile] = File(...),
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    vin: str = Form("UNKNOWN"),
    first_name: str = Form("UNKNOWN"),
    last_name: str = Form("UNKNOWN"),
    generate_llm_report_flag: bool = Form(False),
    language: LanguageCode = Form("en")
):
    case_id = f"{first_name}_{last_name}_{vin}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_dir = "report"
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"report_{language}_{case_id}.txt")
    annot_dir = "annotated"
    os.makedirs(annot_dir, exist_ok=True)

    user = UserData(make=make, model=model, year=year, vin=vin, first_name=first_name, last_name=last_name)

    all_final_damage_list = []
    all_annot_paths = []
    seen_keys = set() # keys to prevent duplicate damages

    for image in images:
        img_path = f"temp/{image.filename}"
        with open(img_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        summary = build_damage_summary(img_path, damage_model, parts_model, conf=0.3)

        for item in summary:
            damage_type = item['damage']
            part = item['part']

            # unique key
            key = f"{damage_type}_{part}"

            if key in seen_keys:
                continue
            seen_keys.add(key)

            line = f"{damage_type} in {part} (conf={item['confidence']:.2f}, IoU={item['iou']:.2f})"

            est, low, high = get_estimated_cost(damage_type, part, user)
            if est:
                final_line = f"{line} — Estimated Cost: ${est} (range: ${low}–${high})"
            else:
                final_line = f"{line} — Estimated Cost: [LLM Estimate Needed]"

            all_final_damage_list.append(final_line)


        try:
            infer_damage = Inference(damage_model, img_path)
            damage_annot_path = os.path.join(annot_dir, f"damage_{image.filename}.jpg")
            infer_damage.run(conf=0.3, iou=0.1, save_path=damage_annot_path)
            all_annot_paths.append(damage_annot_path)

            infer_parts = Inference(parts_model, img_path)
            parts_annot_path = os.path.join(annot_dir, f"parts_{image.filename}.jpg")
            infer_parts.run(conf=0.25, iou=0.1, save_path=parts_annot_path)
            all_annot_paths.append(parts_annot_path)
        except Exception as e:
            print(f"[WARNING] Inference visualization failed for {image.filename}: {e}")

    text_prepare(all_final_damage_list, language=language, case_id=case_id)

    # LLM REPORT
    llm_report = None
    if generate_llm_report_flag:
        try:
            llm_report = await asyncio.to_thread(
                generate_llm_report,
                user,
                language=language,
                language_full=LANGUAGE_MAP[language],
                case_id=case_id
            )
        except Exception as e:
            print("LLM generation failed:", e)

    #  PDF REPORT 
    pdf_path = generate_pdf_report(
        report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_data=user,
        damages=all_final_damage_list if llm_report is None else None,
        llm_report=llm_report,
        images=[os.path.abspath(p) for p in all_annot_paths if p]
    )

    return {
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vehicle": user,
        "damages": all_final_damage_list,
        "llm_report": llm_report,
        "pdf_path": pdf_path,
        "annotated_images": all_annot_paths,
        "report_path": report_path
    }