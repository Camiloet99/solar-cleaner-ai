# app.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models import PredictReq, PredictResp
from datetime import datetime, timezone
import numpy as np, torch, os

from featurizer import window_features, to_tensor_dict
from policy_loader import load_policy

USE_NN = os.getenv("USE_NN", "true").lower() == "true"
model = load_policy() if USE_NN else None

app = FastAPI(title="PV Cleaning AI", version="2.0.0-nn")

def _heuristic(points):
    dust = np.array([p["dustIndex"] for p in points], dtype=float)
    dust_mean = float(dust.mean())
    dust_trend = float(dust[-1] - dust[0])
    k = float(np.clip((dust_mean - 0.20)/0.60, 0.0, 1.0))
    need_now = (dust_mean > 0.40) or (dust_trend > 0.08)
    rec = "now" if need_now else "hold_20s"
    proposed = {
        "brushRpm": 0.5 + 0.5*k,
        "waterFlow": 0.4 + 0.6*k,
        "pressure":  0.5 + 0.5*k,
        "passes":    0.3 + 0.5*k,
        "detergentPct": 0.1*k,
        "route": {"keep":0.1*(1-k), "zigzag_thin":0.6*k+0.2, "zigzag_wide":0.2*(1-k), "spiral_focus":0.1*k}
    }
    s = sum(proposed["route"].values())
    for k2 in proposed["route"]: proposed["route"][k2] /= s
    return rec, proposed, dust_mean, dust_trend

@app.get("/health")
def health(): return {"status":"ok","version":app.version,"use_nn":USE_NN}

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq):
    if len(req.points) < 5:
        return JSONResponse(status_code=400, content={"error":"At least 5 points required"})
    pts = [p.model_dump() for p in req.points]

    if USE_NN and model is not None:
        feats = window_features(pts)
        x = to_tensor_dict(feats)                 # (5,)
        x = torch.tensor(x, dtype=torch.float32).view(1,1,-1)  # (1,1,5)
        with torch.no_grad():
            cont, route = model(x)
        brushRpm, waterFlow, pressure, passes, detergent = cont[0].tolist()
        keep, zigthin, zigwide, spiral = route[0].tolist()
        route_soft = {"keep":keep, "zigzag_thin":zigthin, "zigzag_wide":zigwide, "spiral_focus":spiral}
        s = sum(route_soft.values()) or 1.0
        for k in route_soft: route_soft[k] /= s

        proposed = {
            "brushRpm": float(brushRpm),
            "waterFlow": float(waterFlow),
            "pressure":  float(pressure),
            "passes":    float(passes),
            "detergentPct": float(detergent),
            "route": route_soft
        }
        dust_mean = feats["dust_mean"]; dust_trend = feats["dust_trend"]
        rec = "now" if (dust_mean>0.40 or dust_trend>0.08) else "hold_20s"
        pred_loss = float(min(0.45, 0.12 + 0.6*dust_mean + 0.15*max(0.0,dust_trend)))
        explain = f"NN policy | dust_mean={dust_mean:.2f}, trend={dust_trend:.2f}"
    else:
        rec, proposed, dust_mean, dust_trend = _heuristic(pts)
        pred_loss = float(min(0.35, 0.15 + 0.5*dust_mean))
        explain = f"Heuristic | dust_mean={dust_mean:.2f}, trend={dust_trend:.2f}"

    return {
        "sessionId": req.points[-1].sessionId,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictedEfficiencyLoss": round(pred_loss,3),
        "recommendedCleaningFrequency": rec,
        "cleaningRouteAdjustment": max(proposed["route"], key=proposed["route"].get),
        "alerts": ["dust_high"] if rec=="now" else [],
        "proposedCommands": proposed,
        "explain": explain
    }
