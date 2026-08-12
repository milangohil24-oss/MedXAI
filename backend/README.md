# MEDXAI Python FastAPI Backend & EfficientNetB0 Model Service

This directory contains the Python FastAPI backend service and AI explainability engine for MEDXAI.

## Directory Structure

```
backend/
├── main.py                     # FastAPI application entry point
├── auth.py                     # JWT authentication & password hashing
├── database.py                 # SQLite database connection setup
├── models.py                   # SQLAlchemy models (User, Analysis, Report)
├── model/
│   ├── alzheimer_efficientnetb0.keras  # Trained EfficientNetB0 Keras model
│   └── class_names.json        # Class labels mapping
├── services/
│   ├── prediction.py           # Model loading & MRI preprocessing/inference
│   ├── gradcam.py              # Grad-CAM heatmap overlay generator
│   └── lime_explainer.py       # LIME superpixel feature explainer
├── requirements.txt            # Python package dependencies
└── medxai.db                   # SQLite database
```

## Running the Backend Locally

1. Create a Python virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `alzheimer_efficientnetb0.keras` is placed inside `backend/model/`.

4. Start the FastAPI server on port 8000:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. Access API documentation at:
   `http://127.0.0.1:8000/docs`
