# SensorLens — Manufacturing Anomaly Detection & LLM Explanation

AI-powered system that detects anomalous sensor readings in manufacturing equipment and explains them in plain English using LLaMA 3 70B.

**Live Demo:** [link — add after deploy]

## Architecture

```
┌─────────────────────────────┐
│   Streamlit Cloud (Frontend)│
│   frontend/app.py           │
└──────────────┬──────────────┘
               │ HTTP (REST)
┌──────────────▼──────────────┐
│   Render Free (Backend)     │
│   ├── detector.py           │
│   │    └── Isolation Forest │
│   └── explainer.py          │
│        └── Groq API         │
│             (LLaMA 3 70B)   │
└─────────────────────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| Backend framework | Python 3.11+, FastAPI |
| Anomaly detection | scikit-learn — Isolation Forest |
| LLM inference | Groq API — LLaMA 3.3 70B Versatile |
| Frontend | Streamlit |
| Dataset | AI4I 2020 Predictive Maintenance (10k rows) |
| Backend deploy | Render free tier |
| Frontend deploy | Streamlit Cloud |

## Features

- Configurable Isolation Forest with full parameter control
- Auto-explains all anomalies via LLaMA 3 70B in batches
- Natural language querying over anomaly data
- 4 visualization types: scatter, distributions, heatmap, time series
- Ground truth comparison against 5 labelled failure types
- CSV export of anomaly results

## Local Setup

```bash
git clone https://github.com/411sst/sensorlens
cd sensorlens

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — paste your GROQ_API_KEY (free key at https://console.groq.com)
# Leave BACKEND_URL=http://localhost:8000 for local dev

# Download the dataset from Kaggle and place it at data/ai4i2020.csv
# https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

# Start the backend (must run from inside backend/ due to relative imports)
cd backend
uvicorn main:app --reload

# In a new terminal — run from the project root
streamlit run frontend/app.py
```

## Dataset

AI4I 2020 Predictive Maintenance Dataset — https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

10,000 rows of manufacturing sensor readings with 5 failure type labels.
