import logging
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data_loader import load_dataset, FEATURE_COLUMNS
from detector import IsolationForestDetector
from explainer import GroqExplainer
from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ExplainRequest,
    ExplainResponse,
    QueryRequest,
    QueryResponse,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="SensorLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level state
df = None
detector = None
cache = {"params": None, "result": None}


@app.on_event("startup")
def startup() -> None:
    """Load the dataset and initialize the Isolation Forest detector on startup."""
    global df, detector
    df = load_dataset()
    detector = IsolationForestDetector(df)


@app.get("/health")
def health() -> dict:
    """Return a health check with current timestamp.

    Returns:
        Dict with status "ok" and ISO-format timestamp.
    """
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/dataset")
def dataset() -> list[dict]:
    """Return the first 100 rows of the dataset as a list of dicts.

    Returns:
        List of row dicts for the frontend table preview.
    """
    return df.head(100).to_dict(orient="records")


@app.get("/dataset/full")
def dataset_full() -> list[dict]:
    """Return all 10,000 rows of the dataset for visualizations.

    Returns:
        List of all row dicts.
    """
    return df.to_dict(orient="records")


@app.get("/dataset/stats")
def dataset_stats() -> dict:
    """Return full-dataset statistics for all 10,000 rows.

    Returns:
        Dict with total_rows, feature_means, and failure_rate.
    """
    feature_means = {col: float(df[col].mean()) for col in FEATURE_COLUMNS}
    failure_rate = float(df["machine_failure"].sum() / len(df) * 100)
    return {
        "total_rows": len(df),
        "feature_means": feature_means,
        "failure_rate": failure_rate,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest) -> dict:
    """Run Isolation Forest anomaly detection with caching.

    Args:
        req: Analysis parameters including features, contamination, and model config.

    Returns:
        AnalyzeResponse with anomaly rows, scores, and cache status.
    """
    cache_key = (
        tuple(sorted(req.features)),
        req.contamination,
        req.n_estimators,
        str(req.max_samples),
    )

    if cache["params"] == cache_key:
        result = cache["result"].copy()
        result["cached"] = True
        return result

    flags, scores = detector.detect(
        req.features, req.contamination, req.n_estimators, req.max_samples
    )
    anomalies = detector.build_anomaly_rows(flags, scores, df)

    result = {
        "total_rows": len(df),
        "anomaly_count": len(anomalies),
        "contamination_used": req.contamination,
        "features": req.features,
        "anomalies": anomalies,
        "all_scores": scores.tolist(),
        "cached": False,
    }

    cache["params"] = cache_key
    cache["result"] = result

    return result


@app.post("/explain", response_model=ExplainResponse)
def explain(req: ExplainRequest) -> dict:
    """Generate LLM explanations for anomaly rows via Groq.

    Args:
        req: Request containing list of anomaly row dicts.

    Returns:
        ExplainResponse with a list of {row_id, explanation} dicts.
    """
    try:
        explainer = GroqExplainer()
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    explanations = explainer.explain_anomalies(req.anomalies)
    return {"explanations": explanations}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    """Answer a natural language question using cached anomaly data.

    Args:
        req: Request with the question string and optional context_rows.

    Returns:
        QueryResponse with the LLM's answer.
    """
    if cache["result"] is None:
        raise HTTPException(status_code=400, detail="Run analysis first")
    try:
        explainer = GroqExplainer()
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        answer = explainer.answer_query(req.question, cache["result"], req.context_rows)
        return {"answer": answer}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error("Groq query endpoint failed: %s", e)
        raise HTTPException(status_code=502, detail=str(e))
