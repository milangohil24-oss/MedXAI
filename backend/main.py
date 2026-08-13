import os
import gc
import uuid
import io
from datetime import datetime
from typing import Optional

# ============================================================
# TENSORFLOW MEMORY & CPU OPTIMIZATIONS
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"

import shutil
import tensorflow as tf
import cv2

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .database import (
    users_collection,
    analyses_collection,
    reports_collection,
)
from . import auth

from .services.prediction import (
    predict_mri,
    get_model,
)

from .services.gradcam import (
    generate_gradcam,
)

from .services.lime_explainer import (
    generate_lime_explanation,
)


# ============================================================
# USER & ANALYSIS FORMATTERS
# ============================================================

def public_user(user_doc: dict) -> dict:
    return {
        "id": user_doc["_id"],
        "name": user_doc.get("name", ""),
        "email": user_doc.get("email", ""),
        "role": user_doc.get("role", "patient"),
        "created_at": user_doc.get("created_at"),
    }


def format_analysis(doc: dict) -> dict:
    if not doc:
        return {}
    doc["id"] = doc["_id"]
    return doc


# ============================================================
# FASTAPI APP & CORS SETUP
# ============================================================

app = FastAPI(
    title="MEDXAI - Explainable MRI Intelligence API",
    description="Backend API for Alzheimer's MRI Disease Detection backed by MongoDB Atlas.",
    version="1.0.0",
)

def _normalize_origin(value: str) -> str:
    value = str(value or "").strip().strip('\"\'')
    if value.startswith("[") and "](" in value:
        value = value[1:value.find("](")]
    return value.strip().rstrip("/")

frontend_url = _normalize_origin(
    os.getenv("FRONTEND_URL", "https://medxai-frontend.onrender.com")
)

configured_origins = os.getenv("CORS_ORIGINS", "").strip()

origins = [
    frontend_url,
    "https://medxai-frontend.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if configured_origins:
    for raw_origin in configured_origins.replace(";", ",").replace("\n", ",").split(","):
        normalized = _normalize_origin(raw_origin)
        if normalized:
            origins.append(normalized)

origins = list(dict.fromkeys(origin for origin in origins if origin))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=(
        r"^https://([a-zA-Z0-9-]+\.)?onrender\.com$"
        r"|^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)


# ============================================================
# DIRECTORIES & STATIC FILES
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "service": "MEDXAI FastAPI Engine + MongoDB Atlas"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "MEDXAI FastAPI Engine + MongoDB Atlas"}


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

def _register_user(data: dict, forced_role: Optional[str] = None):
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).lower().strip()
    password = str(data.get("password", ""))
    role = str(forced_role or data.get("role", "patient")).lower().strip()

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required")

    existing_user = users_collection.find_one({"email": email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    hashed_password = auth.get_password_hash(password)
    created_at = datetime.utcnow().isoformat()

    user_doc = {
        "_id": user_id,
        "name": name,
        "email": email,
        "password_hash": hashed_password,
        "role": role,
        "created_at": created_at,
    }

    users_collection.insert_one(user_doc)
    access_token = auth.create_access_token(data={"sub": user_id, "email": email, "role": role})

    return {"access_token": access_token, "token_type": "bearer", "user": public_user(user_doc)}


@app.post("/auth/register")
def register(data: dict):
    return _register_user(data)

@app.post("/auth/patient/register")
def patient_register(data: dict):
    return _register_user(data, "patient")

@app.post("/auth/doctor/register")
def doctor_register(data: dict):
    return _register_user(data, "doctor")


async def _perform_login(request: Request, required_role: Optional[str] = None):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        data = await request.json()
        email = str(data.get("email", "")).lower().strip()
        password = str(data.get("password", ""))
    else:
        form = await request.form()
        email = str(form.get("username", form.get("email", ""))).lower().strip()
        password = str(form.get("password", ""))

    user = users_collection.find_one({"email": email})
    if not user or not auth.verify_password(password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    user_id = user["_id"]
    role = user.get("role", "patient")
    access_token = auth.create_access_token(data={"sub": user_id, "email": email, "role": role})

    return {"access_token": access_token, "token_type": "bearer", "user": public_user(user)}


@app.post("/auth/login")
async def login(request: Request):
    return await _perform_login(request)

@app.post("/auth/doctor/login")
async def doctor_login(request: Request):
    return await _perform_login(request, "doctor")

@app.post("/auth/patient/login")
async def patient_login(request: Request):
    return await _perform_login(request, "patient")

@app.get("/auth/me")
def get_me(current_user: dict = Depends(auth.get_current_user)):
    return public_user(current_user)

@app.post("/auth/logout")
def logout(current_user: dict = Depends(auth.get_current_user)):
    return {"message": "Logged out successfully"}

@app.get("/profile")
def get_profile(current_user: dict = Depends(auth.get_current_user)):
    return public_user(current_user)


# ============================================================
# DASHBOARD STATS
# ============================================================

@app.get("/dashboard/stats")
def get_dashboard_stats(
    current_user: dict = Depends(auth.get_current_user),
):
    user_id = current_user["_id"]
    cursor = analyses_collection.find({"user_id": user_id}).sort("created_at", -1)
    all_analyses = [format_analysis(doc) for doc in cursor]

    total = len(all_analyses)
    avg_confidence = (
        sum(float(a.get("confidence_percentage", 0) or 0) for a in all_analyses) / total
        if total > 0
        else 0.0
    )

    recent = all_analyses[:5]
    latest_prediction = recent[0].get("prediction") if recent else None

    return {
        "total_analyses": total,
        "average_confidence": round(avg_confidence, 2),
        "latest_prediction": latest_prediction,
        "recent": recent,
    }


# ============================================================
# MRI PREDICTION ROUTE
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: dict = Depends(auth.get_current_user),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")

    analysis_id = str(uuid.uuid4())
    safe_filename = os.path.basename(file.filename)
    filename = f"{analysis_id}_{safe_filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction_result = predict_mri(file_path)

    gradcam_filename = f"gradcam_{filename}"
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)

    model = get_model()
    gradcam_error = None

    try:
        generate_gradcam(image_path=file_path, model=model, output_path=gradcam_path)
    except Exception as error:
        gradcam_error = str(error)

    gradcam_url = f"/uploads/{gradcam_filename}" if gradcam_error is None and os.path.exists(gradcam_path) else None

    analysis_doc = {
        "_id": analysis_id,
        "user_id": current_user["_id"],
        "filename": safe_filename,
        "prediction": prediction_result["prediction"],
        "confidence": prediction_result["confidence"],
        "confidence_percentage": prediction_result["confidence_percentage"],
        "probabilities": prediction_result.get("probabilities", {}),
        "gradcam_url": gradcam_url,
        "lime_url": None,
        "image_url": f"/uploads/{filename}",
        "created_at": datetime.utcnow().isoformat(),
    }

    analyses_collection.insert_one(analysis_doc)
    gc.collect()

    return {
        "analysis_id": analysis_id,
        "filename": safe_filename,
        "prediction": prediction_result["prediction"],
        "confidence": prediction_result["confidence"],
        "confidence_percentage": prediction_result["confidence_percentage"],
        "probabilities": prediction_result["probabilities"],
        "image_url": f"/uploads/{filename}",
        "gradcam_url": gradcam_url,
        "lime_url": None,
        "explanation_status": {
            "gradcam": "success" if gradcam_url else "failed",
            "lime": "deferred",
        },
    }


# ============================================================
# HISTORY: GET ALL ANALYSES
# ============================================================

@app.get("/analyses")
def get_analyses(
    current_user: dict = Depends(auth.get_current_user),
):
    user_id = current_user["_id"]
    cursor = analyses_collection.find({"user_id": user_id}).sort("created_at", -1)
    return [format_analysis(doc) for doc in cursor]


@app.get("/analyses/{id}")
def get_analysis_by_id(
    id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    record = analyses_collection.find_one({"_id": id, "user_id": current_user["_id"]})
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return format_analysis(record)


# ============================================================
# LIME EXPLANATION ROUTE
# ============================================================

@app.post("/explain/lime")
def get_lime_explain(
    data: Optional[dict] = None,
    current_user: dict = Depends(auth.get_current_user),
):
    if data is None:
        data = {}

    analysis_id = data.get("analysis_id")
    user_id = current_user["_id"]

    if analysis_id:
        record = analyses_collection.find_one({"_id": analysis_id, "user_id": user_id})
    else:
        cursor = analyses_collection.find({"user_id": user_id}).sort("created_at", -1).limit(1)
        records = list(cursor)
        record = records[0] if records else None

    if not record or not record.get("image_url"):
        raise HTTPException(status_code=404, detail="Analysis record or image not found")

    image_filename = record["image_url"].replace("/uploads/", "")
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    lime_filename = f"lime_{image_filename}"
    lime_path = os.path.join(UPLOAD_DIR, lime_filename)

    model = get_model()

    try:
        generate_lime_explanation(model, image_path, lime_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"LIME generation failed: {str(error)}")

    lime_url = f"/uploads/{lime_filename}"
    analyses_collection.update_one({"_id": record["_id"]}, {"$set": {"lime_url": lime_url}})

    gc.collect()

    return {"analysis_id": record["_id"], "url": lime_url, "features": []}


# ============================================================
# HISTORY: GET ALL REPORTS
# ============================================================

@app.get("/reports")
def get_reports(
    current_user: dict = Depends(auth.get_current_user),
):
    user_id = current_user["_id"]
    cursor = reports_collection.find({"user_id": user_id}).sort("created_at", -1)
    reports = []
    for doc in cursor:
        doc["id"] = doc["_id"]
        reports.append(doc)
    return reports


# ============================================================
# FAST & ENHANCED PDF REPORT GENERATOR (< 0.5s BUILD TIME)
# ============================================================

def generate_pdf_report(report_id: str, analysis: dict, pdf_path: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image as RLImage,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )
    styles = getSampleStyleSheet()
    story = []

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=12,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=8,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Italic"],
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#64748b"),
    )

    story.append(Paragraph("MEDXAI CLINICAL MRI RESEARCH REPORT", title_style))
    story.append(Spacer(1, 6))

    prediction = str(analysis.get("prediction", "Unknown"))
    confidence = float(analysis.get("confidence_percentage", 0) or 0)

    # 1. Overview Table
    info_data = [
        [Paragraph("<b>Report ID:</b>", styles["Normal"]), Paragraph(str(report_id), styles["Normal"])],
        [Paragraph("<b>Scan Filename:</b>", styles["Normal"]), Paragraph(str(analysis.get("filename", "")), styles["Normal"])],
        [Paragraph("<b>Classification:</b>", styles["Normal"]), Paragraph(f"<b>{prediction}</b>", styles["Normal"])],
        [Paragraph("<b>Confidence:</b>", styles["Normal"]), Paragraph(f"<b>{confidence:.2f}%</b>", styles["Normal"])],
    ]

    info_table = Table(info_data, colWidths=[140, 360])
    info_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ])
    )

    story.append(info_table)
    story.append(Spacer(1, 8))

    # 2. Probability Breakdown Table
    probs = analysis.get("probabilities", {}) or {}
    if probs:
        story.append(Paragraph("Stage Probability Distribution", section_style))
        prob_rows = [["Alzheimer's Stage", "Probability Score"]]
        for stage, score in probs.items():
            try:
                score_pct = f"{float(score) * 100:.2f}%" if float(score) <= 1.0 else f"{float(score):.2f}%"
            except Exception:
                score_pct = str(score)
            prob_rows.append([stage, score_pct])

        prob_table = Table(prob_rows, colWidths=[250, 250])
        prob_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(prob_table)
        story.append(Spacer(1, 8))

    # 3. Fast In-Memory Thumbnail Helper for ReportLab
    def make_fast_thumbnail(rel_url, label_text):
        if not rel_url or not rel_url.startswith("/uploads/"):
            return None, None
        full_path = os.path.join(UPLOAD_DIR, rel_url.replace("/uploads/", ""))
        if not os.path.exists(full_path) or os.path.getsize(full_path) < 100:
            return None, None

        try:
            img_bgr = cv2.imread(full_path)
            if img_bgr is None:
                return None, None
            
            img_resized = cv2.resize(img_bgr, (240, 240), interpolation=cv2.INTER_AREA)
            is_success, buffer = cv2.imencode(".jpg", img_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not is_success:
                return None, None

            bytes_io = io.BytesIO(buffer.tobytes())
            rl_img = RLImage(bytes_io, width=145, height=145)
            title_p = Paragraph(f"<b>{label_text}</b>", styles["Normal"])
            return title_p, rl_img
        except Exception:
            return None, None

    # Process all 3 visual assets
    t1, img1 = make_fast_thumbnail(analysis.get("image_url", ""), "Original MRI Scan")
    t2, img2 = make_fast_thumbnail(analysis.get("gradcam_url", ""), "Grad-CAM Heatmap")
    t3, img3 = make_fast_thumbnail(analysis.get("lime_url", ""), "LIME Feature Map")

    visual_titles = [t for t in [t1, t2, t3] if t is not None]
    visual_images = [i for i in [img1, img2, img3] if i is not None]

    if visual_images:
        story.append(Paragraph("Visual Explainability Analysis", section_style))
        col_w = 500 // len(visual_images)
        vis_table = Table([visual_titles, visual_images], colWidths=[col_w] * len(visual_images))
        vis_table.setStyle(
            TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(vis_table)
        story.append(Spacer(1, 8))

    # 4. Clinical Narrative
    story.append(Paragraph("Explainable AI Findings", section_style))
    story.append(
        Paragraph(
            f"The deep learning architecture evaluated brain MRI structural features and classified "
            f"biomarkers consistent with <b>{prediction}</b> at a confidence level of <b>{confidence:.2f}%</b>.<br/><br/>"
            f"• <b>Grad-CAM:</b> Highlights high-attention convolutional activation zones (red/yellow regions) "
            f"corresponding to structural tissue atrophy or ventricular expansion.<br/>"
            f"• <b>LIME:</b> Superpixel attribution isolates local pixel groups contributing positively (red) "
            f"or negatively (cyan) toward the classifier's diagnosis.",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "<b>DISCLAIMER:</b> This report is generated for research and decision-support purposes only.",
            disclaimer_style,
        )
    )

    doc.build(story)
    gc.collect()


# ============================================================
# REPORT ROUTE HANDLERS
# ============================================================

@app.post("/reports/{id}")
def create_report(
    id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    user_id = current_user["_id"]
    analysis = analyses_collection.find_one({"_id": id, "user_id": user_id})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    report_id = str(uuid.uuid4())
    original_name = os.path.basename(analysis.get("filename") or "MRI")
    pdf_filename = f"Report_{os.path.splitext(original_name)[0]}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, f"{report_id}.pdf")

    generate_pdf_report(report_id, analysis, pdf_path)

    created_at = datetime.utcnow().isoformat()
    report_doc = {
        "_id": report_id,
        "analysis_id": id,
        "user_id": user_id,
        "filename": pdf_filename,
        "content": "PDF Report generated successfully",
        "created_at": created_at,
    }

    reports_collection.insert_one(report_doc)
    gc.collect()

    return {
        "id": report_id,
        "analysis_id": id,
        "filename": pdf_filename,
        "content": "PDF Report generated successfully",
        "download_url": f"/reports/{report_id}/download",
        "created_at": created_at,
    }


@app.get("/reports/{id}/download")
def download_report(
    id: str,
    current_user: dict = Depends(auth.get_current_user),
):
    report = reports_collection.find_one({"_id": id, "user_id": current_user["_id"]})
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    pdf_path = os.path.join(REPORTS_DIR, f"{report['_id']}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(pdf_path, filename=report["filename"], media_type="application/pdf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)