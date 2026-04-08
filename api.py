"""
London Rent Prediction — FastAPI Microservice
=============================================
Serves an Avg Ensemble (LightGBM + XGBoost + CatBoost) trained on
Rightmove London rental data.

Endpoints:
  POST /predict  → predicted rent + confidence score
  GET  /health   → liveness check
  GET  /         → service info

Run locally:
  uvicorn api:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Model download (runs if LFS pointers were cloned instead of real binaries)
# ---------------------------------------------------------------------------
RELEASE_BASE = "https://github.com/Tim-Wei-28/rent_prediction/releases/download/v1.0.0"
MODEL_FILES = [
    "models/model_lgb.pkl",
    "models/model_xgb.pkl",
    "models/model_cat.pkl",
    "models/model_meta.pkl",
    "models/encoding.pkl",
    "london_postcodes.csv",
]
LFS_POINTER_PREFIX = b"version https://git-lfs"

def _is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX
    except Exception:
        return False

def _ensure_models() -> None:
    (BASE_DIR / "models").mkdir(exist_ok=True)
    for rel in MODEL_FILES:
        dest = BASE_DIR / rel
        is_pointer = _is_lfs_pointer(dest) if dest.exists() else False
        print(f"[startup] {dest.name}: exists={dest.exists()}, lfs_pointer={is_pointer}")
        if not dest.exists() or is_pointer:
            filename = Path(rel).name
            url = f"{RELEASE_BASE}/{filename}"
            print(f"[startup] downloading {filename} from {url}")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"[startup] {filename} saved ({dest.stat().st_size / 1e6:.1f} MB)")
            except Exception as e:
                raise RuntimeError(f"Failed to download {filename}: {e}") from e

# ---------------------------------------------------------------------------
# Global state (populated at startup)
# ---------------------------------------------------------------------------
_models: dict = {}
_enc: dict = {}
_postcodes_df: Optional[pd.DataFrame] = None
_station_coords: Optional[np.ndarray] = None

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="London Rent Prediction API",
    description=(
        "Predicts monthly rent for a London property using an ensemble of "
        "gradient-boosting models (LightGBM + XGBoost + CatBoost)."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: load models and reference data
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup_event() -> None:
    global _models, _enc, _postcodes_df, _station_coords

    _ensure_models()

    models_dir = BASE_DIR / "models"
    _models["lgb"]  = joblib.load(models_dir / "model_lgb.pkl")
    _models["xgb"]  = joblib.load(models_dir / "model_xgb.pkl")
    _models["cat"]  = joblib.load(models_dir / "model_cat.pkl")
    _models["meta"] = joblib.load(models_dir / "model_meta.pkl")
    _enc = joblib.load(models_dir / "encoding.pkl")

    # Postcode lookup: keep only what we need to reduce memory
    pc_path = BASE_DIR / "london_postcodes.csv"
    _postcodes_df = pd.read_csv(pc_path, usecols=["Postcode", "Latitude", "Longitude"])
    _postcodes_df["Postcode"] = (
        _postcodes_df["Postcode"].str.strip().str.upper().str.replace(" ", "", regex=False)
    )
    _postcodes_df = _postcodes_df.set_index("Postcode")

    # Tube station coords in radians for vectorised Haversine
    stations = pd.read_csv(BASE_DIR / "london_stations.csv")
    _station_coords = np.radians(stations[["Latitude", "Longitude"]].values)

    print(f"[startup] models loaded | postcodes: {len(_postcodes_df):,} | stations: {len(stations)}")


# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
class PropertyInput(BaseModel):
    zip_code: str = Field(
        ...,
        description="UK postcode, e.g. 'SW1A 2AA'",
        examples=["SW1A 2AA"],
    )
    size_sqm: float = Field(
        ..., gt=0, description="Property size in square metres", examples=[65.0]
    )
    bedrooms: int = Field(..., ge=0, le=10, examples=[2])
    bathrooms: int = Field(..., ge=1, le=8, examples=[1])
    property_type: str = Field(
        "Flat",
        description="Flat / House / Studio / Maisonette / Bungalow",
        examples=["Flat"],
    )
    furnished: str = Field(
        "Unfurnished",
        description="Furnished / Part Furnished / Unfurnished / Flexible",
        examples=["Furnished"],
    )
    floor: Optional[int] = Field(None, ge=0, le=100, examples=[3])
    amenities: List[str] = Field(
        default_factory=list,
        description="Any of: Balcony, Gym, Concierge, Swimming Pool, Lift",
        examples=[["Balcony", "Lift"]],
    )


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
class RentPrediction(BaseModel):
    predicted_rent_gbp: int = Field(description="Avg ensemble point estimate (£/month)")
    band_low: int = Field(description="Lower end of confidence band (£/month)")
    band_high: int = Field(description="Upper end of confidence band (£/month)")
    confidence_pct: float = Field(
        description="Model disagreement as % of mean prediction — lower = more confident"
    )
    confidence_label: str = Field(description="High / Moderate / Low")
    subdistrict: str = Field(description="Resolved postcode district, e.g. SW1A")


# ---------------------------------------------------------------------------
# Feature-engineering helpers
# ---------------------------------------------------------------------------
def _resolve_postcode(postcode: str):
    """Return (lat, lon, subdistrict_code) or raise 422."""
    clean = postcode.strip().upper().replace(" ", "")
    row = _postcodes_df.loc[clean] if clean in _postcodes_df.index else None

    if row is None:
        # Fuzzy: try prefix of decreasing length
        for length in (6, 5, 4):
            prefix = clean[:length]
            candidates = _postcodes_df[_postcodes_df.index.str.startswith(prefix)]
            if not candidates.empty:
                row = candidates.iloc[0]
                break

    if row is None:
        raise HTTPException(
            status_code=422,
            detail=f"Postcode '{postcode}' not found in London postcode database.",
        )

    lat = float(row["Latitude"])
    lon = float(row["Longitude"])

    # Subdistrict = area code e.g. SW1A from SW1A2AA
    m = re.match(r"^([A-Z]{1,2}\d{1,2}[A-Z]?)", clean)
    subdistrict = m.group(1) if m else clean[:4]
    return lat, lon, subdistrict


def _min_tube_dist_km(lat: float, lon: float) -> float:
    query = np.radians([[lat, lon]])
    lat_diff = query[:, 0:1] - _station_coords[:, 0]
    lon_diff = query[:, 1:2] - _station_coords[:, 1]
    a = np.sin(lat_diff / 2) ** 2 + (
        np.cos(query[:, 0:1]) * np.cos(_station_coords[:, 0]) * np.sin(lon_diff / 2) ** 2
    )
    return float((6371 * 2 * np.arcsin(np.sqrt(a))).min())


def _knn_rent_mean(lat: float, lon: float) -> float:
    k = _enc["knn_k"]
    ref_lat = _enc["knn_train_lat"]
    ref_lon = _enc["knn_train_lon"]
    ref_y   = _enc["knn_train_y_log"]
    ref_rad   = np.radians(np.column_stack([ref_lat, ref_lon]))
    query_rad = np.radians([[lat, lon]])
    nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="haversine")
    nn.fit(ref_rad)
    _, indices = nn.kneighbors(query_rad)
    return float(ref_y[indices[0]].mean())


def _build_features(inp: PropertyInput) -> pd.DataFrame:
    enc = _enc
    lat, lon, subdistrict = _resolve_postcode(inp.zip_code)

    # sqm → sqft, clamp to training range
    size_sqft = inp.size_sqm * 10.764
    if not (100 <= size_sqft <= 3000):
        size_sqft = np.nan

    furnish_enc        = enc["furnish_map"].get(inp.furnished, np.nan)
    property_type_clean = enc["prop_type_map"].get(inp.property_type, "Other")

    a = {s.strip() for s in inp.amenities}
    has_balcony       = 1 if "Balcony"       in a else 0
    has_gym           = 1 if "Gym"           in a else 0
    has_concierge     = 1 if "Concierge"     in a else 0
    has_swimming_pool = 1 if "Swimming Pool" in a else 0
    has_lift          = 1 if "Lift"          in a else 0
    amenity_count     = has_balcony + has_gym + has_concierge + has_swimming_pool + has_lift

    bathrooms   = inp.bathrooms if inp.bathrooms <= 7 else np.nan
    floor_level = inp.floor if inp.floor is not None and inp.floor <= 100 else np.nan

    bedrooms_safe    = max(inp.bedrooms, 1)
    size_per_bedroom = size_sqft / bedrooms_safe if not np.isnan(size_sqft) else np.nan

    subdistrict_enc = enc["code_smoothed"].get(subdistrict, enc["global_mean_log"])
    tube_dist_km    = _min_tube_dist_km(lat, lon)
    knn_rent_mean   = _knn_rent_mean(lat, lon)

    row = {
        "bedrooms":            inp.bedrooms,
        "bathrooms":           bathrooms,
        "size_sqft":           size_sqft,
        "property_type_clean": property_type_clean,
        "furnish_enc":         furnish_enc,
        "has_balcony":         has_balcony,
        "has_gym":             has_gym,
        "has_concierge":       has_concierge,
        "has_swimming_pool":   has_swimming_pool,
        "has_lift":            has_lift,
        "floor_level":         floor_level,
        "size_per_bedroom":    size_per_bedroom,
        "amenity_count":       amenity_count,
        "subdistrict_enc":     subdistrict_enc,
        "tube_dist_km":        tube_dist_km,
        "knn_rent_mean":       knn_rent_mean,
        "latitude":            lat,
        "longitude":           lon,
    }
    return pd.DataFrame([row])[enc["features"]]


def _confidence_label(pct: float) -> str:
    if pct < 6:
        return "High"
    elif pct < 12:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=RentPrediction, summary="Predict monthly rent")
def predict(inp: PropertyInput) -> RentPrediction:
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")

    try:
        X = _build_features(inp)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {exc}") from exc

    try:
        p_lgb = float(np.exp(_models["lgb"].predict(X)[0]))
        p_xgb = float(np.exp(_models["xgb"].predict(X)[0]))
        p_cat = float(np.exp(_models["cat"].predict(X)[0]))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    p_avg = (p_lgb + p_xgb + p_cat) / 3
    p_std = float(np.std([p_lgb, p_xgb, p_cat]))
    confidence_pct = round(p_std / p_avg * 100, 1)

    _, _, subdistrict = _resolve_postcode(inp.zip_code)

    return RentPrediction(
        predicted_rent_gbp=round(p_avg),
        band_low=round(p_avg - p_std),
        band_high=round(p_avg + p_std),
        confidence_pct=confidence_pct,
        confidence_label=_confidence_label(confidence_pct),
        subdistrict=subdistrict,
    )


@app.get("/health", summary="Health check")
def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": bool(_models),
        "postcodes_loaded": _postcodes_df is not None,
    }


@app.get("/", summary="Service info")
def root() -> dict:
    return {
        "service": "London Rent Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
    }
