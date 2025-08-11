# app.py
# uvicorn app:app --reload
import os
import shutil
import pandas as pd
from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import asyncio

app = FastAPI()

from src.detection.detector import Detector
from src.segmentation.segmenter import Segmenter
from src.utils import build_damage_summary, text_prepare, Inference
from src.pdf_utils import generate_pdf_report
from typing import Literal

# LANGUAGE SUPPORT
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

damage_model = Detector('path/to/model')
parts_model = Segmenter('path/to/model')

damage_costs = pd.read_csv('path/to/data')

fallback_means = (
    damage_costs
    .groupby(['damage_type', 'part'])['average_cost_usd']
    .mean()
    .reset_index()
)


class UserData(BaseModel):
    make: str
    model: str
    year: int
    vin: str = "UNKNOWN"


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


def generate_llm_report(user_data: UserData, language_full: str = "English") -> str:
    """
    Generates a professional damage report using LLM based on report.txt and user vehicle data.
    """
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),
        base_url="https://api.deepseek.com"
    )

    report_path = "report/report.txt"
    if not os.path.exists(report_path):
        raise FileNotFoundError("report.txt not found. Make sure detection step ran before LLM.")

    with open(report_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prompt = f"""
    You are a professional automotive damage report writer.
    Write the report in {language_full} language.
    Your task is to write a **clear, human-readable, structured** vehicle damage assessment based on the provided data.
    Do NOT use markdown tables or complex formatting — only headings, bullet points, and paragraphs.
    Never output extra characters, code blocks, or decorative symbols.

    Rules for formatting:
    - Start with "Vehicle Damage Assessment Report"
    - Then "Report Date: <date>"
    - Then "Vehicle Information:" as bullet points
    - Then "Detected Damages:" — one bullet per damage, each with:
        • Damage type and vehicle part
        • Detection confidence (%)
        • Localization accuracy (IoU %)
        • Estimated repair cost in USD (with a range if available)
        • "See attached image" for photo reference

    Also include a final section "Summary of Estimated Repair Costs" — just a bullet list of part: cost.

    If any price is missing, estimate a realistic market value for that type of repair.

    Report Date: {current_time}
    Vehicle Information:
    - Make: {user_data.make}
    - Model: {user_data.model}
    - Year: {user_data.year}
    - VIN: {user_data.vin}

    Detected Damages (from CV models):
    {report_content}

    Write the report exactly in this structure, without any extra commentary.
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": f"You are a report generator for vehicle damage assessments. Output in {language_full}."},
            {"role": "user", "content": prompt}
        ],
        stream=False
    )

    return response.choices[0].message.content.strip()


@app.post("/assess_damage")
async def assess_damage(
    image: UploadFile,
    make: str = Form(...),
    model: str = Form(...),
    year: int = Form(...),
    vin: str = Form("UNKNOWN"),
    generate_llm_report_flag: bool = Form(False),
    language: LanguageCode = Form("en")
):
    
    img_path = f"temp/{image.filename}"
    os.makedirs("temp", exist_ok=True)

    with open(img_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    summary = build_damage_summary(img_path, damage_model, parts_model, conf=0.6)

    report_lines = []
    for item in summary:
        line = f"{item['damage']} in {item['part']} (conf={item['confidence']:.2f}, IoU={item['iou']:.2f})"
        report_lines.append(line)

    user = UserData(make=make, model=model, year=year, vin=vin)
    final_damage_list = []
    for line in report_lines:
        if " in " not in line:
            continue
        damage_type = line.split(" in ")[0].strip()
        part = line.split(" in ")[1].split(" ")[0].strip()
        est, low, high = get_estimated_cost(damage_type, part, user)
        if est:
            final_line = f"{line} — Estimated Cost: ${est} (range: ${low}–${high})"
        else:
            final_line = f"{line} — Estimated Cost: [LLM Estimate Needed]"
        final_damage_list.append(final_line)

    # report
    text_prepare(final_damage_list)

    # IMAGE SAVE
    annot_dir = "temp/annotated"
    os.makedirs(annot_dir, exist_ok=True)
    try:
        infer_damage = Inference(damage_model, img_path)
        damage_annot_path = os.path.join(annot_dir, f"damage_{image.filename}.jpg")
        infer_damage.run(conf=0.6, iou=0.1, save_path=damage_annot_path)

        infer_parts = Inference(parts_model, img_path)
        parts_annot_path = os.path.join(annot_dir, f"parts_{image.filename}.jpg")
        infer_parts.run(conf=0.45, iou=0.1, save_path=parts_annot_path)
    except Exception as e:
        print(f"[WARNING] Inference visualization failed: {e}")
        damage_annot_path = None
        parts_annot_path = None

    # LLM
    llm_report = None
    if generate_llm_report_flag:
        try:
            llm_report = await asyncio.to_thread(generate_llm_report, user, LANGUAGE_MAP[language])
        except Exception as e:
            print("LLM generation failed:", e)
            llm_report = None
    # PDF
    pdf_path = generate_pdf_report(
        report_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_data=user,
        damages=final_damage_list if llm_report is None else None,
        llm_report=llm_report,
        images=[os.path.abspath(p) for p in [damage_annot_path, parts_annot_path] if p]
# images=[p for p in [damage_annot_path, parts_annot_path] if p]
    )

    return {
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vehicle": user,
        "damages": final_damage_list,
        "llm_report": llm_report,
        "pdf_path": pdf_path
    }