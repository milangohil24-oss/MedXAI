import os

# ============================================================
# TENSORFLOW MEMORY SETTINGS
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import uuid
import json
import shutil

from typing import Optional

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
# DATABASE
# ============================================================

Base.metadata.create_all(bind=engine)


# ============================================================
# DATABASE SCHEMA COMPATIBILITY
# ============================================================
# Older databases may not have the "role" column.
# Add it automatically if necessary.

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
    print(
        f"Role schema initialization warning: "
        f"{role_schema_error}"
    )


# ============================================================
# USER ROLE
# ============================================================

def get_user_role(
    db: Session,
    user_id: str,
) -> str:

    try:
        value = db.execute(
            text(
                "SELECT role "
                "FROM users "
                "WHERE id = :user_id"
            ),
            {
                "user_id": user_id
            },
        ).scalar_one_or_none()

        if value in {"doctor", "patient"}:
            return value

        return "patient"

    except Exception:
        return "patient"


# ============================================================
# PUBLIC USER RESPONSE
# ============================================================

def public_user(
    db: Session,
    user: models.User,
) -> dict:

    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "role": get_user_role(
            db,
            user.id,
        ),
        "created_at": (
            user.created_at.isoformat()
            if user.created_at
            else None
        ),
    }


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MEDXAI - Explainable MRI Intelligence API",
    description=(
        "Backend API for Alzheimer's MRI Disease Detection "
        "using EfficientNetB0, Grad-CAM and LIME."
    ),
    version="1.0.0",
)

# ============================================================
# CORS
# ============================================================

def _normalize_origin(value: str) -> str:
    """Normalize CORS origins from environment variables.

    This also repairs accidental Markdown-style values such as:
    [https://example.com](https://example.com)
    which can otherwise cause browsers to reject CORS requests.
    """

    value = str(value or "").strip()

    if not value:
        return ""

    # Remove surrounding quotes commonly introduced in env settings.
    value = value.strip('\"\'')

    # Convert Markdown link syntax to the actual origin.
    if value.startswith("[") and "](" in value:
        value = value[1:value.find("](")]

    return value.strip().rstrip("/")


frontend_url = _normalize_origin(
    os.getenv(
        "FRONTEND_URL",
        "https://medxai-frontend.onrender.com",
    )
)

configured_origins = os.getenv(
    "CORS_ORIGINS",
    "",
).strip()

# Explicit origins used by the deployed frontend and local development.
origins = [
    frontend_url,
    "https://medxai-frontend.onrender.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

# Add extra origins from the CORS_ORIGINS environment variable.
# Commas, semicolons, and newlines are accepted.
if configured_origins:
    for raw_origin in configured_origins.replace(";", ",").replace("\n", ",").split(","):
        normalized = _normalize_origin(raw_origin)
        if normalized:
            origins.append(normalized)

# Remove duplicates and empty values while preserving order.
origins = list(dict.fromkeys(
    origin for origin in origins if origin
))

print("CORS allowed origins:", origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Also allow local development ports and Render frontend subdomains.
    # This is useful when the frontend runs on localhost:3000 while the
    # backend is deployed on Render.
    allow_origin_regex=(
        r"^https://([a-zA-Z0-9-]+\.)?onrender\.com$"
        r"|^https?://(localhost|127\.0\.0\.1)(:\d+)?$"
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
    ],
    max_age=600,
)

# DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads",
)

REPORTS_DIR = os.path.join(
    BASE_DIR,
    "reports",
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)

os.makedirs(
    REPORTS_DIR,
    exist_ok=True,
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/uploads",
    StaticFiles(
        directory=UPLOAD_DIR
    ),
    name="uploads",
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "service": "MEDXAI FastAPI Engine",
    }


@app.get("/api/health")
def health_check():

    return {
        "status": "ok",
        "service": "MEDXAI FastAPI Engine",
    }


# ============================================================
# AUTHENTICATION
# REGISTER USER
# ============================================================

def _register_user(
    data: dict,
    db: Session,
    forced_role: Optional[str] = None,
):

    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    requested_role = (
        forced_role
        or data.get(
            "role",
            "patient",
        )
    )

    if not name or not email or not password:
        raise HTTPException(
            status_code=400,
            detail=(
                "Name, email, and password "
                "are required"
            ),
        )

    name = str(name).strip()
    email = str(email).lower().strip()
    role = str(requested_role).lower().strip()

    if role not in {
        "doctor",
        "patient",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Role must be either "
                "doctor or patient"
            ),
        )

    if len(name) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Name must be at least "
                "2 characters"
            ),
        )

    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail=(
                "Password must be at least "
                "6 characters"
            ),
        )

    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail=(
                "Password must be "
                "72 bytes or less"
            ),
        )

    existing_user = (
        db.query(models.User)
        .filter(
            models.User.email == email
        )
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered",
        )

    user_id = str(
        uuid.uuid4()
    )

    try:
        hashed_password = (
            auth.get_password_hash(
                password
            )
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "Password hashing failed: "
                f"{str(error)}"
            ),
        )

    try:

        db.execute(
            text(
                "INSERT INTO users "
                "(id, name, email, password_hash, role) "
                "VALUES "
                "(:id, :name, :email, "
                ":password_hash, :role)"
            ),
            {
                "id": user_id,
                "name": name,
                "email": email,
                "password_hash": hashed_password,
                "role": role,
            },
        )

        db.commit()

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Registration failed: "
                f"{str(error)}"
            ),
        )

    user = (
        db.query(models.User)
        .filter(
            models.User.id == user_id
        )
        .first()
    )

    if not user:
        raise HTTPException(
            status_code=500,
            detail=(
                "Registered user "
                "could not be loaded"
            ),
        )

    access_token = (
        auth.create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "role": role,
            }
        )
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": public_user(
            db,
            user,
        ),
    }


# ============================================================
# GENERAL REGISTER
# ============================================================

@app.post("/auth/register")
def register(
    data: dict,
    db: Session = Depends(get_db),
):

    return _register_user(
        data,
        db,
    )


# ============================================================
# PATIENT REGISTER
# ============================================================

@app.post("/auth/patient/register")
def patient_register(
    data: dict,
    db: Session = Depends(get_db),
):

    return _register_user(
        data,
        db,
        "patient",
    )


# ============================================================
# DOCTOR REGISTER
# ============================================================

@app.post("/auth/doctor/register")
def doctor_register(
    data: dict,
    db: Session = Depends(get_db),
):

    return _register_user(
        data,
        db,
        "doctor",
    )


# ============================================================
# LOGIN HELPER
# ============================================================

async def _perform_login(
    request: Request,
    db: Session,
    required_role: Optional[str] = None,
):

    content_type = (
        request.headers
        .get(
            "content-type",
            "",
        )
        .lower()
    )

    if "application/json" in content_type:

        try:
            data = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON body",
            )

        email = str(
            data.get(
                "email",
                "",
            )
        ).lower().strip()

        password = str(
            data.get(
                "password",
                "",
            )
        )

        requested_role = str(
            data.get(
                "role",
                "",
            )
        ).lower().strip()

    else:

        try:
            form = await request.form()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid form data",
            )

        email = str(
            form.get(
                "username",
                form.get(
                    "email",
                    "",
                ),
            )
        ).lower().strip()

        password = str(
            form.get(
                "password",
                "",
            )
        )

        requested_role = str(
            form.get(
                "role",
                "",
            )
        ).lower().strip()

    if not email or not password:

        raise HTTPException(
            status_code=400,
            detail=(
                "Email and password "
                "are required"
            ),
        )

    user = (
        db.query(models.User)
        .filter(
            models.User.email == email
        )
        .first()
    )

    if not user:

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid email or password"
            ),
        )

    try:

        password_valid = (
            auth.verify_password(
                password,
                user.password_hash,
            )
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Password verification "
                f"failed: {str(error)}"
            ),
        )

    if not password_valid:

        raise HTTPException(
            status_code=401,
            detail=(
                "Invalid email or password"
            ),
        )

    role = get_user_role(
        db,
        user.id,
    )

    expected_role = (
        required_role
        or requested_role
    )

    if expected_role in {
        "doctor",
        "patient",
    }:

        if role != expected_role:

            if expected_role == "doctor":

                message = (
                    "This account is a "
                    "patient account. "
                    "Please use Patient Login."
                )

            else:

                message = (
                    "This account is a "
                    "doctor account. "
                    "Please use Doctor Login."
                )

            raise HTTPException(
                status_code=403,
                detail=message,
            )

    access_token = (
        auth.create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "role": role,
            }
        )
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": public_user(
            db,
            user,
        ),
    }


# ============================================================
# GENERAL LOGIN
# ============================================================

@app.post("/auth/login")
async def login(
    request: Request,
    db: Session = Depends(get_db),
):

    return await _perform_login(
        request,
        db,
    )


# ============================================================
# DOCTOR LOGIN
# ============================================================

@app.post("/auth/doctor/login")
async def doctor_login(
    request: Request,
    db: Session = Depends(get_db),
):

    return await _perform_login(
        request,
        db,
        "doctor",
    )


# ============================================================
# PATIENT LOGIN
# ============================================================

@app.post("/auth/patient/login")
async def patient_login(
    request: Request,
    db: Session = Depends(get_db),
):

    return await _perform_login(
        request,
        db,
        "patient",
    )


# ============================================================
# CURRENT USER
# ============================================================

@app.get("/auth/me")
def get_me(
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    return public_user(
        db,
        current_user,
    )


# ============================================================
# LOGOUT
# ============================================================

@app.post("/auth/logout")
def logout(
    current_user: models.User = Depends(
        auth.get_current_user
    ),
):

    return {
        "message": (
            "Logged out successfully"
        ),
    }


# ============================================================
# CHANGE PASSWORD
# ============================================================

@app.post("/auth/change-password")
def change_password(
    data: dict,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    current_password = data.get(
        "current_password"
    )

    new_password = data.get(
        "new_password"
    )

    if not current_password or not new_password:

        raise HTTPException(
            status_code=400,
            detail=(
                "Current and new password "
                "are required"
            ),
        )

    if len(new_password) < 6:

        raise HTTPException(
            status_code=400,
            detail=(
                "New password must be "
                "at least 6 characters"
            ),
        )

    if len(
        new_password.encode("utf-8")
    ) > 72:

        raise HTTPException(
            status_code=400,
            detail=(
                "New password must be "
                "72 bytes or less"
            ),
        )

    if not auth.verify_password(
        current_password,
        current_user.password_hash,
    ):

        raise HTTPException(
            status_code=401,
            detail=(
                "Current password incorrect"
            ),
        )

    current_user.password_hash = (
        auth.get_password_hash(
            new_password
        )
    )

    db.commit()

    return {
        "message": (
            "Password updated successfully"
        ),
    }


# ============================================================
# PROFILE - GET
# ============================================================

@app.get("/profile")
def get_profile(
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    return public_user(
        db,
        current_user,
    )


# ============================================================
# PROFILE - UPDATE
# ============================================================

@app.put("/profile")
def update_profile(
    data: dict,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    name = data.get("name")
    email = data.get("email")

    if name:

        name = str(name).strip()

        if len(name) < 2:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Name must be at least "
                    "2 characters"
                ),
            )

        current_user.name = name

    if email:

        email = (
            str(email)
            .lower()
            .strip()
        )

        if email != current_user.email:

            existing_user = (
                db.query(models.User)
                .filter(
                    models.User.email == email,
                    models.User.id
                    != current_user.id,
                )
                .first()
            )

            if existing_user:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Email already registered"
                    ),
                )

            current_user.email = email

    db.commit()
    db.refresh(current_user)

    return public_user(
        db,
        current_user,
    )


# ============================================================
# MRI PREDICTION
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    # --------------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected",
        )

    allowed_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use JPG, JPEG, PNG, BMP or WEBP."
            ),
        )

    # --------------------------------------------------------
    # ANALYSIS ID
    # --------------------------------------------------------

    analysis_id = str(
        uuid.uuid4()
    )

    safe_filename = os.path.basename(
        file.filename
    )

    filename = (
        f"{analysis_id}_{safe_filename}"
    )

    file_path = os.path.join(
        UPLOAD_DIR,
        filename,
    )

    # --------------------------------------------------------
    # SAVE MRI
    # --------------------------------------------------------

    try:

        with open(
            file_path,
            "wb",
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "File upload failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    try:

        prediction_result = (
            predict_mri(
                file_path
            )
        )

    except Exception as error:

        if os.path.exists(file_path):

            try:
                os.remove(file_path)
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=(
                "MRI prediction failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # EXPLANATION FILE PATHS
    # --------------------------------------------------------

    gradcam_filename = (
        f"gradcam_{filename}"
    )

    lime_filename = (
        f"lime_{filename}"
    )

    gradcam_path = os.path.join(
        UPLOAD_DIR,
        gradcam_filename,
    )

    lime_path = os.path.join(
        UPLOAD_DIR,
        lime_filename,
    )

    # --------------------------------------------------------
    # REMOVE EXISTING EXPLANATION FILES
    # --------------------------------------------------------

    for explanation_file in [
        gradcam_path,
        lime_path,
    ]:

        if os.path.exists(
            explanation_file
        ):

            try:
                os.remove(
                    explanation_file
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model = get_model()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model loading failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    gradcam_error = None

    try:

        generate_gradcam(
            model,
            file_path,
            gradcam_path,
        )

        if not os.path.exists(
            gradcam_path
        ):

            raise RuntimeError(
                "Grad-CAM output file "
                "was not created"
            )

        if os.path.getsize(
            gradcam_path
        ) <= 100:

            raise RuntimeError(
                "Grad-CAM output file "
                "is empty"
            )

    except Exception as error:

        gradcam_error = str(error)

        print(
            "=================================================="
        )

        print(
            "GRAD-CAM ERROR:"
        )

        print(
            gradcam_error
        )

        print(
            "=================================================="
        )

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------
    # LIME is intentionally deferred from /predict.
    # This prevents excessive memory usage on small
    # Render instances.
    #
    # It can be generated using /explain/lime.

    lime_error = (
        "LIME generation deferred"
    )

    lime_url = None

    # --------------------------------------------------------
    # GRAD-CAM URL
    # --------------------------------------------------------

    gradcam_url = None

    if (
        gradcam_error is None
        and os.path.exists(
            gradcam_path
        )
    ):

        gradcam_url = (
            f"/uploads/{gradcam_filename}"
        )

    # --------------------------------------------------------
    # DATABASE RECORD
    # --------------------------------------------------------

    try:

        probabilities = (
            prediction_result.get(
                "probabilities",
                {},
            )
        )

        analysis_record = (
            models.Analysis(
                id=analysis_id,

                user_id=current_user.id,

                filename=safe_filename,

                prediction=(
                    prediction_result[
                        "prediction"
                    ]
                ),

                confidence=(
                    prediction_result[
                        "confidence"
                    ]
                ),

                confidence_percentage=(
                    prediction_result[
                        "confidence_percentage"
                    ]
                ),

                probabilities=probabilities,

                gradcam_url=gradcam_url,

                lime_url=lime_url,

                image_url=(
                    f"/uploads/{filename}"
                ),
            )
        )

        db.add(
            analysis_record
        )

        db.commit()

        db.refresh(
            analysis_record
        )

    except Exception as error:

        db.rollback()

        raise HTTPException(
            status_code=500,
            detail=(
                "Database save failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    response = {

        "analysis_id": analysis_id,

        "filename": safe_filename,

        "prediction": (
            prediction_result[
                "prediction"
            ]
        ),

        "confidence": (
            prediction_result[
                "confidence"
            ]
        ),

        "confidence_percentage": (
            prediction_result[
                "confidence_percentage"
            ]
        ),

        "probabilities": (
            prediction_result[
                "probabilities"
            ]
        ),

        "image_url": (
            f"/uploads/{filename}"
        ),

        "gradcam_url": gradcam_url,

        "lime_url": lime_url,

        "explanation_status": {

            "gradcam": (
                "success"
                if gradcam_url
                else "failed"
            ),

            "lime": "deferred",
        },
    }

    if gradcam_error:

        response[
            "gradcam_error"
        ] = gradcam_error

    if lime_error:

        response[
            "lime_error"
        ] = lime_error

    return response


# ============================================================
# ANALYSES - CURRENT USER
# ============================================================

@app.get("/analyses")
def get_analyses(
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    records = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.user_id
            == current_user.id
        )
        .order_by(
            models.Analysis.created_at.desc()
        )
        .all()
    )

    return records


# ============================================================
# SINGLE ANALYSIS
# ============================================================

@app.get("/analyses/{id}")
def get_analysis_by_id(
    id: str,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    record = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.id == id,
            models.Analysis.user_id
            == current_user.id,
        )
        .first()
    )

    if not record:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found",
        )

    return record


# ============================================================
# DELETE ANALYSIS
# ============================================================

@app.delete("/analyses/{id}")
def delete_analysis(
    id: str,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    record = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.id == id,
            models.Analysis.user_id
            == current_user.id,
        )
        .first()
    )

    if not record:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found",
        )

    # --------------------------------------------------------
    # DELETE ORIGINAL MRI
    # --------------------------------------------------------

    if record.image_url:

        image_filename = (
            record.image_url
            .replace(
                "/uploads/",
                "",
            )
        )

        image_path = os.path.join(
            UPLOAD_DIR,
            image_filename,
        )

        if os.path.exists(
            image_path
        ):

            try:
                os.remove(
                    image_path
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # DELETE GRAD-CAM
    # --------------------------------------------------------

    if record.gradcam_url:

        gradcam_filename = (
            record.gradcam_url
            .replace(
                "/uploads/",
                "",
            )
        )

        gradcam_path = os.path.join(
            UPLOAD_DIR,
            gradcam_filename,
        )

        if os.path.exists(
            gradcam_path
        ):

            try:
                os.remove(
                    gradcam_path
                )
            except Exception:
                pass

    # --------------------------------------------------------
    # DELETE LIME
    # --------------------------------------------------------

    if record.lime_url:

        lime_filename = (
            record.lime_url
            .replace(
                "/uploads/",
                "",
            )
        )

        lime_path = os.path.join(
            UPLOAD_DIR,
            lime_filename,
        )

        if os.path.exists(
            lime_path
        ):

            try:
                os.remove(
                    lime_path
                )
            except Exception:
                pass

    db.delete(
        record
    )

    db.commit()

    return {
        "message": (
            "Analysis deleted successfully"
        ),
    }


# ============================================================
# DASHBOARD STATISTICS
# ============================================================

@app.get("/dashboard/stats")
def get_dashboard_stats(
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    all_analyses = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.user_id
            == current_user.id
        )
        .order_by(
            models.Analysis.created_at.desc()
        )
        .all()
    )

    total = len(
        all_analyses
    )

    if total > 0:

        avg_confidence = (
            sum(
                float(
                    a.confidence_percentage
                    or 0
                )
                for a in all_analyses
            )
            / total
        )

    else:

        avg_confidence = 0.0

    recent = (
        all_analyses[:5]
    )

    latest_prediction = (
        recent[0].prediction
        if recent
        else None
    )

    return {

        "total_analyses": total,

        "average_confidence": round(
            avg_confidence,
            2,
        ),

        "latest_prediction": (
            latest_prediction
        ),

        "recent": recent,
    }


# ============================================================
# LIME EXPLANATION
# ============================================================

@app.post("/explain/lime")
def get_lime_explain(
    data: Optional[dict] = None,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    if data is None:
        data = {}

    analysis_id = data.get(
        "analysis_id"
    )

    # --------------------------------------------------------
    # FIND ANALYSIS
    # --------------------------------------------------------

    if analysis_id:

        record = (
            db.query(models.Analysis)
            .filter(
                models.Analysis.id
                == analysis_id,

                models.Analysis.user_id
                == current_user.id,
            )
            .first()
        )

    else:

        record = (
            db.query(models.Analysis)
            .filter(
                models.Analysis.user_id
                == current_user.id
            )
            .order_by(
                models.Analysis.created_at.desc()
            )
            .first()
        )

    if not record:

        raise HTTPException(
            status_code=404,
            detail=(
                "Analysis record "
                "not found"
            ),
        )

    # --------------------------------------------------------
    # FIND ORIGINAL IMAGE
    # --------------------------------------------------------

    if not record.image_url:

        raise HTTPException(
            status_code=404,
            detail=(
                "Original MRI image "
                "not found"
            ),
        )

    image_filename = (
        record.image_url
        .replace(
            "/uploads/",
            "",
        )
    )

    image_path = os.path.join(
        UPLOAD_DIR,
        image_filename,
    )

    if not os.path.exists(
        image_path
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Original MRI file "
                "not found"
            ),
        )

    # --------------------------------------------------------
    # LIME OUTPUT
    # --------------------------------------------------------

    lime_filename = (
        f"lime_{image_filename}"
    )

    lime_path = os.path.join(
        UPLOAD_DIR,
        lime_filename,
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model = get_model()

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model loading failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # GENERATE LIME
    # --------------------------------------------------------

    try:

        result = generate_lime_explanation(
            model,
            image_path,
            lime_path,
        )

        # Some implementations return
        # a path while others simply
        # create the file.

        if isinstance(
            result,
            str,
        ):

            if os.path.exists(
                result
            ):

                lime_path = result

        if not os.path.exists(
            lime_path
        ):

            raise RuntimeError(
                "LIME output file "
                "was not created"
            )

        if os.path.getsize(
            lime_path
        ) <= 100:

            raise RuntimeError(
                "LIME output file "
                "is empty"
            )

    except Exception as error:

        print(
            "=================================================="
        )

        print(
            "LIME ERROR:"
        )

        print(
            str(error)
        )

        print(
            "=================================================="
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "LIME generation failed: "
                f"{str(error)}"
            ),
        )

    # --------------------------------------------------------
    # UPDATE DATABASE
    # --------------------------------------------------------

    record.lime_url = (
        f"/uploads/{lime_filename}"
    )

    db.commit()
    db.refresh(
        record
    )

    return {

        "analysis_id": record.id,

        "url": (
            f"/uploads/{lime_filename}"
        ),

        "features": [],
    }


# ============================================================
# REPORTS - CURRENT USER
# ============================================================

@app.get("/reports")
def get_reports(
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    return (
        db.query(models.Report)
        .filter(
            models.Report.user_id
            == current_user.id
        )
        .order_by(
            models.Report.created_at.desc()
        )
        .all()
    )


# ============================================================
# PDF REPORT GENERATOR
# ============================================================

def generate_pdf_report(
    report_id: str,
    analysis: models.Analysis,
    pdf_path: str,
):

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

    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle,
    )

    # --------------------------------------------------------
    # DOCUMENT
    # --------------------------------------------------------

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = (
        getSampleStyleSheet()
    )

    story = []

    # --------------------------------------------------------
    # STYLES
    # --------------------------------------------------------

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor(
            "#0284c7"
        ),
        spaceAfter=12,
    )

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor(
            "#0f172a"
        ),
        spaceBefore=8,
        spaceAfter=6,
    )

    explanation_style = ParagraphStyle(
        "Explanation",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor(
            "#334155"
        ),
        spaceAfter=8,
    )

    finding_style = ParagraphStyle(
        "Finding",
        parent=styles["BodyText"],
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor(
            "#1e293b"
        ),
        leftIndent=8,
        spaceAfter=6,
    )

    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Italic"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor(
            "#64748b"
        ),
    )

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "MEDXAI CLINICAL MRI RESEARCH REPORT",
            title_style,
        )
    )

    story.append(
        Spacer(1, 10)
    )

    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    probabilities = (
        analysis.probabilities
        or {}
    )

    if isinstance(
        probabilities,
        str,
    ):

        try:

            probabilities = (
                json.loads(
                    probabilities
                )
            )

        except Exception:

            probabilities = {}

    probability_items = []

    if isinstance(
        probabilities,
        dict,
    ):

        for key, value in (
            probabilities.items()
        ):

            try:

                numeric_value = float(
                    value
                )

                if numeric_value > 1:

                    numeric_value /= 100

            except Exception:

                numeric_value = 0.0

            probability_items.append(
                (
                    str(key),
                    numeric_value,
                )
            )

    probability_items.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = str(
        getattr(
            analysis,
            "prediction",
            "Unknown",
        )
    )

    try:

        confidence = float(
            getattr(
                analysis,
                "confidence_percentage",
                0,
            )
            or 0
        )

    except Exception:

        confidence = 0.0

    # --------------------------------------------------------
    # REPORT INFORMATION
    # --------------------------------------------------------

    info_data = [

        [
            Paragraph(
                "<b>Report ID:</b>",
                styles["Normal"],
            ),

            Paragraph(
                str(report_id),
                styles["Normal"],
            ),
        ],

        [
            Paragraph(
                "<b>Date/Time:</b>",
                styles["Normal"],
            ),

            Paragraph(
                str(
                    getattr(
                        analysis,
                        "created_at",
                        "",
                    )
                ),
                styles["Normal"],
            ),
        ],

        [
            Paragraph(
                "<b>Scan Filename:</b>",
                styles["Normal"],
            ),

            Paragraph(
                str(
                    getattr(
                        analysis,
                        "filename",
                        "",
                    )
                ),
                styles["Normal"],
            ),
        ],

        [
            Paragraph(
                "<b>Diagnosis Classification:</b>",
                styles["Normal"],
            ),

            Paragraph(
                f"<b>{prediction}</b>",
                styles["Normal"],
            ),
        ],

        [
            Paragraph(
                "<b>Model Confidence:</b>",
                styles["Normal"],
            ),

            Paragraph(
                f"{confidence:.2f}%",
                styles["Normal"],
            ),
        ],
    ]

    info_table = Table(
        info_data,
        colWidths=[
            160,
            380,
        ],
    )

    info_table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(
                        "#f8fafc"
                    ),
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#cbd5e1"
                    ),
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
            ]
        )
    )

    story.append(
        info_table
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # PREDICTION SUMMARY
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "AI Prediction Summary",
            section_style,
        )
    )

    if confidence >= 80:

        confidence_description = (
            "The model produced a "
            "high-confidence classification "
            "for the submitted MRI image."
        )

    elif confidence >= 60:

        confidence_description = (
            "The model produced a "
            "moderate-to-high confidence "
            "classification. The result should "
            "be interpreted together with the "
            "probability distribution and "
            "explainability outputs."
        )

    elif confidence >= 40:

        confidence_description = (
            "The model produced an "
            "intermediate-confidence "
            "classification. The probability "
            "distribution indicates that "
            "alternative classes should also "
            "be considered."
        )

    else:

        confidence_description = (
            "The model produced a relatively "
            "low-confidence classification, "
            "indicating substantial uncertainty "
            "in the prediction."
        )

    prediction_text = (
        "The MEDXAI EfficientNetB0 model "
        "classified the submitted MRI image "
        f"as <b>{prediction}</b> with a model "
        f"confidence of <b>{confidence:.2f}%</b>. "
        f"{confidence_description}"
    )

    story.append(
        Paragraph(
            prediction_text,
            explanation_style,
        )
    )

    # --------------------------------------------------------
    # PROBABILITY ANALYSIS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Probability Analysis",
            section_style,
        )
    )

    if probability_items:

        top_class, top_probability = (
            probability_items[0]
        )

        probability_text = (
            "The highest model probability "
            "is associated with "
            f"<b>{top_class}</b> at "
            f"<b>{top_probability * 100:.2f}%</b>. "
        )

        if len(
            probability_items
        ) > 1:

            second_class, second_probability = (
                probability_items[1]
            )

            difference = (
                top_probability
                - second_probability
            )

            probability_text += (
                "The next highest probability "
                f"is <b>{second_class}</b> at "
                f"<b>{second_probability * 100:.2f}%</b>, "
                "giving a probability difference "
                f"of <b>{difference * 100:.2f} "
                "percentage points</b>. "
            )

            if difference >= 0.30:

                probability_text += (
                    "This indicates a comparatively "
                    "strong separation between the "
                    "leading class and the next "
                    "most likely class."
                )

            elif difference >= 0.10:

                probability_text += (
                    "This indicates a noticeable "
                    "but not decisive separation "
                    "between the leading classes."
                )

            else:

                probability_text += (
                    "The relatively small separation "
                    "indicates that the model has "
                    "meaningful uncertainty between "
                    "the leading classes."
                )

    else:

        probability_text = (
            "A probability distribution was "
            "not available for detailed "
            "class comparison."
        )

    story.append(
        Paragraph(
            probability_text,
            explanation_style,
        )
    )

    # --------------------------------------------------------
    # PROBABILITY TABLE
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Class Probability Breakdown",
            section_style,
        )
    )

    probability_table_data = [
        [
            "Class",
            "Probability",
        ]
    ]

    for key, value in (
        probability_items
    ):

        probability_table_data.append(
            [
                str(key),
                f"{value * 100:.2f}%",
            ]
        )

    if len(
        probability_table_data
    ) == 1:

        probability_table_data.append(
            [
                "Unavailable",
                "0.00%",
            ]
        )

    probability_table = Table(
        probability_table_data,
        colWidths=[
            300,
            240,
        ],
    )

    probability_table.setStyle(
        TableStyle(
            [

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#0284c7"
                    ),
                ),

                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor(
                        "#e2e8f0"
                    ),
                ),

                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),

                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
            ]
        )
    )

    story.append(
        probability_table
    )

    story.append(
        Spacer(1, 15)
    )

    # --------------------------------------------------------
    # ORIGINAL MRI
    # --------------------------------------------------------

    image_rel = getattr(
        analysis,
        "image_url",
        "",
    )

    if (
        image_rel
        and image_rel.startswith(
            "/uploads/"
        )
    ):

        image_filename = (
            image_rel.replace(
                "/uploads/",
                "",
            )
        )

        image_full_path = os.path.join(
            UPLOAD_DIR,
            image_filename,
        )

        if os.path.exists(
            image_full_path
        ):

            story.append(
                Paragraph(
                    "Original MRI Scan",
                    section_style,
                )
            )

            story.append(
                Spacer(1, 5)
            )

            story.append(
                RLImage(
                    image_full_path,
                    width=200,
                    height=200,
                )
            )

            story.append(
                Spacer(1, 15)
            )

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    gradcam_rel = getattr(
        analysis,
        "gradcam_url",
        "",
    )

    gradcam_available = False

    if (
        gradcam_rel
        and gradcam_rel.startswith(
            "/uploads/"
        )
    ):

        gradcam_filename = (
            gradcam_rel.replace(
                "/uploads/",
                "",
            )
        )

        gradcam_full_path = os.path.join(
            UPLOAD_DIR,
            gradcam_filename,
        )

        if (
            os.path.exists(
                gradcam_full_path
            )
            and os.path.getsize(
                gradcam_full_path
            ) > 100
        ):

            gradcam_available = True

            story.append(
                Paragraph(
                    "Grad-CAM Explainability Heatmap",
                    section_style,
                )
            )

            story.append(
                Spacer(1, 5)
            )

            story.append(
                RLImage(
                    gradcam_full_path,
                    width=200,
                    height=200,
                )
            )

            story.append(
                Spacer(1, 8)
            )

            story.append(
                Paragraph(
                    (
                        "The Grad-CAM visualization "
                        "provides a model-focused "
                        "representation of the image "
                        "regions that contributed most "
                        "strongly to the "
                        f"<b>{prediction}</b> "
                        "classification. Highlighted "
                        "regions should be interpreted "
                        "as areas receiving greater "
                        "influence from the neural "
                        "network rather than as "
                        "definitive evidence of disease."
                    ),
                    explanation_style,
                )
            )

            story.append(
                Spacer(1, 10)
            )

    # --------------------------------------------------------
    # LIME
    # --------------------------------------------------------

    lime_rel = getattr(
        analysis,
        "lime_url",
        "",
    )

    lime_available = False

    if (
        lime_rel
        and lime_rel.startswith(
            "/uploads/"
        )
    ):

        lime_filename = (
            lime_rel.replace(
                "/uploads/",
                "",
            )
        )

        lime_full_path = os.path.join(
            UPLOAD_DIR,
            lime_filename,
        )

        if (
            os.path.exists(
                lime_full_path
            )
            and os.path.getsize(
                lime_full_path
            ) > 100
        ):

            lime_available = True

            story.append(
                Paragraph(
                    "LIME Local Explanation",
                    section_style,
                )
            )

            story.append(
                Spacer(1, 5)
            )

            story.append(
                RLImage(
                    lime_full_path,
                    width=200,
                    height=200,
                )
            )

            story.append(
                Spacer(1, 8)
            )

            story.append(
                Paragraph(
                    (
                        "The LIME visualization "
                        "provides a local explanation "
                        "of the model decision by "
                        "identifying image regions "
                        "that contributed to the "
                        "prediction for this individual "
                        "MRI. These regions describe "
                        "the behavior of the AI model "
                        "and should not be interpreted "
                        "as a direct anatomical diagnosis."
                    ),
                    explanation_style,
                )
            )

            story.append(
                Spacer(1, 10)
            )

    # --------------------------------------------------------
    # OVERALL FINDINGS
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "Overall AI Findings",
            section_style,
        )
    )

    if (
        gradcam_available
        and lime_available
    ):

        explanation_status = (
            "Both Grad-CAM and LIME "
            "explainability outputs were "
            "successfully generated."
        )

    elif gradcam_available:

        explanation_status = (
            "Grad-CAM was successfully "
            "generated, while a usable "
            "LIME visualization was not "
            "available."
        )

    elif lime_available:

        explanation_status = (
            "LIME was successfully "
            "generated, while a usable "
            "Grad-CAM visualization was "
            "not available."
        )

    else:

        explanation_status = (
            "Neither Grad-CAM nor LIME "
            "produced a usable "
            "visualization for this analysis."
        )

    overall_text = (
        "The overall MEDXAI analysis "
        f"classified the MRI as "
        f"<b>{prediction}</b> with a "
        f"confidence of <b>{confidence:.2f}%</b>. "
        f"{explanation_status} "
        "The probability distribution "
        "describes how the model allocated "
        "prediction likelihood across the "
        "available classes, while Grad-CAM "
        "and LIME provide complementary "
        "views of the model's decision "
        "process. These explainability "
        "methods describe model behavior "
        "and do not independently establish "
        "the presence, absence, severity, "
        "or stage of Alzheimer's disease."
    )

    story.append(
        Paragraph(
            overall_text,
            explanation_style,
        )
    )

    # --------------------------------------------------------
    # FINDINGS
    # --------------------------------------------------------

    if probability_items:

        top_class, top_probability = (
            probability_items[0]
        )

        story.append(
            Paragraph(
                "• Leading classification: "
                f"<b>{top_class}</b>",
                finding_style,
            )
        )

        story.append(
            Paragraph(
                "• Leading probability: "
                f"<b>{top_probability * 100:.2f}%</b>",
                finding_style,
            )
        )

        story.append(
            Paragraph(
                "• Reported model confidence: "
                f"<b>{confidence:.2f}%</b>",
                finding_style,
            )
        )

    story.append(
        Paragraph(
            (
                "• Explainability interpretation: "
                "highlighted regions represent "
                "areas that influenced the AI "
                "model's decision and should not "
                "be treated as independently "
                "diagnostic."
            ),
            finding_style,
        )
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            (
                "<b>MEDICAL DISCLAIMER:</b> "
                "This AI-generated report is "
                "produced by the MEDXAI "
                "EfficientNetB0 Engine for "
                "research and clinical "
                "decision-support purposes. "
                "The prediction, probability "
                "values, Grad-CAM visualization, "
                "and LIME explanation describe "
                "the behavior of an artificial "
                "intelligence model and are not "
                "a medical diagnosis. This report "
                "must be reviewed by a qualified "
                "radiologist, neurologist, or "
                "other appropriately qualified "
                "healthcare professional before "
                "any clinical diagnosis or "
                "treatment decision is made."
            ),
            disclaimer_style,
        )
    )

    # --------------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------------

    doc.build(
        story
    )


# ============================================================
# CREATE REPORT
# ============================================================

@app.post("/reports/{id}")
def create_report(
    id: str,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    analysis = (
        db.query(models.Analysis)
        .filter(
            models.Analysis.id == id,
            models.Analysis.user_id
            == current_user.id,
        )
        .first()
    )

    if not analysis:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found",
        )

    report_id = str(
        uuid.uuid4()
    )

    original_name = (
        os.path.basename(
            analysis.filename
            or "MRI"
        )
    )

    pdf_filename = (
        f"Report_"
        f"{os.path.splitext(original_name)[0]}"
        f".pdf"
    )

    pdf_path = os.path.join(
        REPORTS_DIR,
        f"{report_id}.pdf",
    )

    try:

        generate_pdf_report(
            report_id,
            analysis,
            pdf_path,
        )

    except Exception as error:

        print(
            "ReportLab PDF generation error:",
            error,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF generation failed: "
                f"{str(error)}"
            ),
        )

    if not os.path.exists(
        pdf_path
    ):

        raise HTTPException(
            status_code=500,
            detail=(
                "PDF file was not created"
            ),
        )

    content = (
        "PDF Report generated successfully"
    )

    try:

        report_obj = models.Report(
            id=report_id,

            analysis_id=id,

            user_id=current_user.id,

            filename=pdf_filename,

            content=content,
        )

        db.add(
            report_obj
        )

        db.commit()

        db.refresh(
            report_obj
        )

    except Exception as error:

        db.rollback()

        if os.path.exists(
            pdf_path
        ):

            try:
                os.remove(
                    pdf_path
                )
            except Exception:
                pass

        raise HTTPException(
            status_code=500,
            detail=(
                "Report database save failed: "
                f"{str(error)}"
            ),
        )

    return {

        "id": report_obj.id,

        "analysis_id": (
            report_obj.analysis_id
        ),

        "filename": (
            report_obj.filename
        ),

        "content": (
            report_obj.content
        ),

        "download_url": (
            f"/reports/"
            f"{report_obj.id}"
            f"/download"
        ),

        "created_at": (
            report_obj.created_at.isoformat()
            if report_obj.created_at
            else None
        ),
    }


# ============================================================
# DOWNLOAD REPORT
# ============================================================

@app.get("/reports/{id}/download")
def download_report(
    id: str,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    report = (
        db.query(models.Report)
        .filter(
            models.Report.id == id,
            models.Report.user_id
            == current_user.id,
        )
        .first()
    )

    if not report:

        raise HTTPException(
            status_code=404,
            detail="Report not found",
        )

    pdf_path = os.path.join(
        REPORTS_DIR,
        f"{report.id}.pdf",
    )

    if not os.path.exists(
        pdf_path
    ):

        raise HTTPException(
            status_code=404,
            detail=(
                "Report file not found"
            ),
        )

    return FileResponse(
        pdf_path,
        filename=report.filename,
        media_type="application/pdf",
    )


# ============================================================
# DELETE REPORT
# ============================================================

@app.delete("/reports/{id}")
def delete_report(
    id: str,
    current_user: models.User = Depends(
        auth.get_current_user
    ),
    db: Session = Depends(get_db),
):

    report = (
        db.query(models.Report)
        .filter(
            models.Report.id == id,
            models.Report.user_id
            == current_user.id,
        )
        .first()
    )

    if not report:

        raise HTTPException(
            status_code=404,
            detail="Report not found",
        )

    pdf_path = os.path.join(
        REPORTS_DIR,
        f"{report.id}.pdf",
    )

    if os.path.exists(
        pdf_path
    ):

        try:
            os.remove(
                pdf_path
            )
        except Exception as error:

            print(
                "Could not delete PDF:",
                error,
            )

    db.delete(
        report
    )

    db.commit()

    return {
        "message": (
            "Report deleted successfully"
        ),
    }


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )