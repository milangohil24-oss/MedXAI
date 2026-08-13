import os
import gc

# ============================================================
# TENSORFLOW MEMORY & CPU OPTIMIZATIONS
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"

import uuid
import json
import shutil
from typing import Optional

import tensorflow as tf

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

from sqlalchemy.orm import Session
from sqlalchemy import inspect, text

from .database import engine, Base, get_db
from . import models
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
# DATABASE SETUP
# ============================================================

Base.metadata.create_all(bind=engine)

try:
    user_columns = {
        column["name"]
        for column in inspect(engine).get_columns("users")
    }

    if "role" not in user_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE users "
                    "ADD COLUMN role VARCHAR(20) DEFAULT 'patient'"
                )
            )

        with engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE users "
                    "SET role = 'patient' "
                    "WHERE role IS NULL OR role = ''"
                )
            )

except Exception as role_schema_error:
    print(f"Role schema initialization warning: {role_schema_error}")


# ============================================================
# USER ROLE & PUBLIC USER
# ============================================================

def get_user_role(db: Session, user_id: str) -> str:
    try:
        value = db.execute(
            text("SELECT role FROM users WHERE id = :user_id"),
            {"user_id": user_id},
        ).scalar_one_or_none()

        if value in {"doctor", "patient"}:
            return value
        return "patient"
    except Exception:
        return "patient"


def public_user(db: Session, user: models.User) -> dict:
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": get_user_role(db, user.id),
        "created_at": (
            user.created_at.isoformat()
            if user.created_at
            else None
        ),
    }


# ============================================================
# FASTAPI APP & CORS
# ============================================================

app = FastAPI(
    title="MEDXAI - Explainable MRI Intelligence API",
    description="Backend API for Alzheimer's MRI Disease Detection.",
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
    return {"status": "ok", "service": "MEDXAI FastAPI Engine"}

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "MEDXAI FastAPI Engine"}


# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

def _register_user(data: dict, db: Session, forced_role: Optional[str] = None):
    name = str(data.get("name", "")).strip()
    email = str(data.get("email", "")).lower().strip()
    password = str(data.get("password", ""))
    role = str(forced_role or data.get("role", "patient")).lower().strip()

    if not name or not email or not password:
        raise HTTPException(status_code=400, detail="Name, email, and password are required")

    existing_user = db.query(models.User).filter(models.User.email == email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    hashed_password = auth.get_password_hash(password)

    db.execute(
        text("INSERT INTO users (id, name, email, password_hash, role) VALUES (:id, :name, :email, :password_hash, :role)"),
        {"id": user_id, "name": name, "email": email, "password_hash": hashed_password, "role": role}
    )
    db.commit()

    user = db.query(models.User).filter(models.User.id == user_id).first()
    access_token = auth.create_access_token(data={"sub": user.id, "email": user.email, "role": role})

    return {"access_token": access_token, "token_type": "bearer", "user": public_user(db, user)}


@app.post("/auth/register")
def register(data: dict, db: Session = Depends(get_db)):
    return _register_user(data, db)

@app.post("/auth/patient/register")
def patient_register(data: dict, db: Session = Depends(get_db)):
    return _register_user(data, db, "patient")

@app.post("/auth/doctor/register")
def doctor_register(data: dict, db: Session = Depends(get_db)):
    return _register_user(data, db, "doctor")


async def _perform_login(request: Request, db: Session, required_role: Optional[str] = None):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        data = await request.json()
        email = str(data.get("email", "")).lower().strip()
        password = str(data.get("password", ""))
        requested_role = str(data.get("role", "")).lower().strip()
    else:
        form = await request.form()
        email = str(form.get("username", form.get("email", ""))).lower().strip()
        password = str(form.get("password", ""))
        requested_role = str(form.get("role", "")).lower().strip()

    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not auth.verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    role = get_user_role(db, user.id)
    access_token = auth.create_access_token(data={"sub": user.id, "email": user.email, "role": role})

    return {"access_token": access_token, "token_type": "bearer", "user": public_user(db, user)}


@app.post("/auth/login")
async def login(request: Request, db: Session = Depends(get_db)):
    return await _perform_login(request, db)

@app.post("/auth/doctor/login")
async def doctor_login(request: Request, db: Session = Depends(get_db)):
    return await _perform_login(request, db, "doctor")

@app.post("/auth/patient/login")
async def patient_login(request: Request, db: Session = Depends(get_db)):
    return await _perform_login(request, db, "patient")

@app.get("/auth/me")
def get_me(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    return public_user(db, current_user)

@app.post("/auth/logout")
def logout(current_user: models.User = Depends(auth.get_current_user)):
    return {"message": "Logged out successfully"}

@app.get("/profile")
def get_profile(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(get_db)):
    return public_user(db, current_user)


# ============================================================
# MRI PREDICTION ROUTE
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
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
    lime_filename = f"lime_{filename}"
    gradcam_path = os.path.join(UPLOAD_DIR, gradcam_filename)
    lime_path = os.path.join(UPLOAD_DIR, lime_filename)

    model = get_model()
    gradcam_error = None

    try:
        generate_gradcam(image_path=file_path, model=model, output_path=gradcam_path)
    except Exception as error:
        gradcam_error = str(error)

    gradcam_url = f"/uploads/{gradcam_filename}" if gradcam_error is None and os.path.exists(gradcam_path) else None

    analysis_record = models.Analysis(
        id=analysis_id,
        user_id=current_user.id,
        filename=safe_filename,
        prediction=prediction_result["prediction"],
        confidence=prediction_result["confidence"],
        confidence_percentage=prediction_result["confidence_percentage"],
        probabilities=prediction_result.get("probabilities", {}),
        gradcam_url=gradcam_url,
        lime_url=None,
        image_url=f"/uploads/{filename}",
    )

    db.add(analysis_record)
    db.commit()
    db.refresh(analysis_record)

    tf.keras.backend.clear_session()
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
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves full analysis history for the logged-in user."""
    return (
        db.query(models.Analysis)
        .filter(models.Analysis.user_id == current_user.id)
        .order_by(models.Analysis.created_at.desc())
        .all()
    )


@app.get("/analyses/{id}")
def get_analysis_by_id(
    id: str,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    record = (
        db.query(models.Analysis)
        .filter(models.Analysis.id == id, models.Analysis.user_id == current_user.id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return record


# ============================================================
# LIME EXPLANATION ROUTE
# ============================================================

@app.post("/explain/lime")
def get_lime_explain(
    data: Optional[dict] = None,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    if data is None:
        data = {}

    analysis_id = data.get("analysis_id")

    if analysis_id:
        record = db.query(models.Analysis).filter(
            models.Analysis.id == analysis_id,
            models.Analysis.user_id == current_user.id,
        ).first()
    else:
        record = db.query(models.Analysis).filter(
            models.Analysis.user_id == current_user.id
        ).order_by(models.Analysis.created_at.desc()).first()

    if not record or not record.image_url:
        raise HTTPException(status_code=404, detail="Analysis record or image not found")

    image_filename = record.image_url.replace("/uploads/", "")
    image_path = os.path.join(UPLOAD_DIR, image_filename)
    lime_filename = f"lime_{image_filename}"
    lime_path = os.path.join(UPLOAD_DIR, lime_filename)

    model = get_model()

    try:
        generate_lime_explanation(model, image_path, lime_path)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"LIME generation failed: {str(error)}")

    record.lime_url = f"/uploads/{lime_filename}"
    db.commit()
    db.refresh(record)

    tf.keras.backend.clear_session()
    gc.collect()

    return {"analysis_id": record.id, "url": f"/uploads/{lime_filename}", "features": []}


# ============================================================
# HISTORY: GET ALL REPORTS
# ============================================================

@app.get("/reports")
def get_reports(
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    """Retrieves full PDF reports history for the logged-in user."""
    return (
        db.query(models.Report)
        .filter(models.Report.user_id == current_user.id)
        .order_by(models.Report.created_at.desc())
        .all()
    )


# ============================================================
# GENERATE PDF REPORT
# ============================================================

def generate_pdf_report(report_id: str, analysis: models.Analysis, pdf_path: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

    tf.keras.backend.clear_session()
    gc.collect()

    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=16, textColor=colors.HexColor("#0284c7"), spaceAfter=10)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=12, textColor=colors.HexColor("#0f172a"), spaceBefore=6, spaceAfter=4)
    body_style = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=13, textColor=colors.HexColor("#334155"), spaceAfter=6)
    disclaimer_style = ParagraphStyle("Disclaimer", parent=styles["Italic"], fontSize=7.5, leading=10, textColor=colors.HexColor("#64748b"))

    story.append(Paragraph("MEDXAI CLINICAL MRI RESEARCH REPORT", title_style))
    story.append(Spacer(1, 8))

    prediction = str(getattr(analysis, "prediction", "Unknown"))
    confidence = float(getattr(analysis, "confidence_percentage", 0) or 0)

    info_data = [
        [Paragraph("<b>Report ID:</b>", styles["Normal"]), Paragraph(str(report_id), styles["Normal"])],
        [Paragraph("<b>Scan Filename:</b>", styles["Normal"]), Paragraph(str(getattr(analysis, "filename", "")), styles["Normal"])],
        [Paragraph("<b>Classification:</b>", styles["Normal"]), Paragraph(f"<b>{prediction}</b>", styles["Normal"])],
        [Paragraph("<b>Confidence:</b>", styles["Normal"]), Paragraph(f"{confidence:.2f}%", styles["Normal"])],
    ]

    info_table = Table(info_data, colWidths=[140, 360])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))

    story.append(info_table)
    story.append(Spacer(1, 10))

    story.append(Paragraph("AI Diagnostic Summary", section_style))
    story.append(Paragraph(f"The MEDXAI model analyzed the scan as <b>{prediction}</b> at <b>{confidence:.2f}%</b> confidence.", body_style))

    # Add MRI
    image_rel = getattr(analysis, "image_url", "")
    if image_rel and image_rel.startswith("/uploads/"):
        image_path = os.path.join(UPLOAD_DIR, image_rel.replace("/uploads/", ""))
        if os.path.exists(image_path):
            story.append(Paragraph("Original MRI Scan", section_style))
            story.append(RLImage(image_path, width=150, height=150))
            story.append(Spacer(1, 8))

    # Add Grad-CAM
    gradcam_rel = getattr(analysis, "gradcam_url", "")
    if gradcam_rel and gradcam_rel.startswith("/uploads/"):
        gradcam_path = os.path.join(UPLOAD_DIR, gradcam_rel.replace("/uploads/", ""))
        if os.path.exists(gradcam_path) and os.path.getsize(gradcam_path) > 100:
            story.append(Paragraph("Grad-CAM Heatmap", section_style))
            story.append(RLImage(gradcam_path, width=150, height=150))
            story.append(Spacer(1, 8))

    story.append(Paragraph("<b>DISCLAIMER:</b> For clinical decision support only.", disclaimer_style))

    doc.build(story)
    gc.collect()


@app.post("/reports/{id}")
def create_report(
    id: str,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    analysis = db.query(models.Analysis).filter(models.Analysis.id == id, models.Analysis.user_id == current_user.id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    report_id = str(uuid.uuid4())
    original_name = os.path.basename(analysis.filename or "MRI")
    pdf_filename = f"Report_{os.path.splitext(original_name)[0]}.pdf"
    pdf_path = os.path.join(REPORTS_DIR, f"{report_id}.pdf")

    generate_pdf_report(report_id, analysis, pdf_path)

    report_obj = models.Report(
        id=report_id,
        analysis_id=id,
        user_id=current_user.id,
        filename=pdf_filename,
        content="PDF Report generated successfully",
    )

    db.add(report_obj)
    db.commit()
    db.refresh(report_obj)

    gc.collect()

    return {
        "id": report_obj.id,
        "analysis_id": report_obj.analysis_id,
        "filename": report_obj.filename,
        "content": report_obj.content,
        "download_url": f"/reports/{report_obj.id}/download",
        "created_at": report_obj.created_at.isoformat() if report_obj.created_at else None,
    }


@app.get("/reports/{id}/download")
def download_report(
    id: str,
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(get_db),
):
    report = db.query(models.Report).filter(models.Report.id == id, models.Report.user_id == current_user.id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")

    pdf_path = os.path.join(REPORTS_DIR, f"{report.id}.pdf")
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Report file not found")

    return FileResponse(pdf_path, filename=report.filename, media_type="application/pdf")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)