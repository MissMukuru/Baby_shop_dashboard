# Setup guide for the new MLOps layer

This adds a backend API, containerization, CI/CD, and a DVC pipeline on top
of your existing `Nila_baby_shop/` package. Copy these files into the repo
root (preserving folder structure) and follow the steps below.

## File map — where everything goes

```
Baby_shop_dashboard/
├── app/
│   └── main.py              ← NEW: FastAPI backend
├── tests/
│   └── test_api.py          ← NEW: API tests
├── .github/workflows/
│   └── ci.yml                ← NEW: CI/CD
├── Dockerfile.api             ← NEW
├── Dockerfile.frontend        ← NEW
├── docker-compose.yml         ← NEW
├── dvc.yaml                   ← NEW: pipeline definition
├── Nila_baby_shop/             ← already exists
├── models/                     ← already exists
└── requirements.txt            ← already exists, add: fastapi, uvicorn[standard], dvc
```

## 1. Run the API locally (no Docker needed)

```bash
pip install fastapi "uvicorn[standard]"
uvicorn app.main:app --reload --port 8000
# visit http://localhost:8000/docs for interactive Swagger UI
```

## 2. Run everything with one command (the demo-friendly path)

```bash
docker compose up --build
# API:       http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

This is what you point people to when demoing — "one command, two services,
done." The frontend depends on the API being healthy before it starts.

## 3. Wire up DVC (versioned data + reproducible pipeline)

```bash
pip install "dvc[s3]"        # swap [s3] for [gs]/[azure] depending on your cloud
dvc init
dvc remote add -d storage s3://your-bucket/nila-baby-shop
dvc repro                    # runs dataset -> features -> train -> predict
dvc push                     # uploads data + model versions to remote
```

Note: `dataset.py` is currently still the cookiecutter placeholder — you'll
need to replace its body with real ingestion logic before `dvc repro` does
anything meaningful for that stage.

## 4. CI/CD

`.github/workflows/ci.yml` runs lint + tests on every push/PR, and builds
both Docker images on merges to `main`. The deploy step is a placeholder —
add your actual cloud command once you've picked a target, e.g.:

```yaml
- name: Deploy to Cloud Run
  run: |
    gcloud run deploy nila-api --image nila-api:${{ github.sha }} --region us-central1
```

## Suggested next steps, in order

1. Replace `dataset.py`'s placeholder body with real raw→processed logic.
2. Add MLflow tracking inside `demand_forecast.py` (wrap the `.fit()` calls).
3. Point `dashboard_app.py` at the API (`requests.post(f"{API_URL}/v1/predict")`)
   instead of reading CSVs directly — this finishes the "frontend never
   touches the model directly" architecture.
4. Pick one cloud target and deploy for real (Cloud Run is the least setup).
5. Add drift monitoring (Evidently AI) once the API is logging predictions.
