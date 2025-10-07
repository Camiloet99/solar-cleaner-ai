# featurizer.py
import numpy as np

FEATURES = [
  "dust_mean","dust_trend","temp_mean","hum_mean","power_mean"
]

def window_features(points: list[dict]) -> dict:
    dust = np.array([p["dustIndex"] for p in points], dtype=float)
    temp = np.array([p["temperature"] for p in points], dtype=float)
    hum  = np.array([p["humidity"]    for p in points], dtype=float)
    poww = np.array([p["powerOutput"] for p in points], dtype=float)

    return {
        "dust_mean": float(dust.mean()),
        "dust_trend": float(dust[-1]-dust[0]),
        "temp_mean": float(temp.mean()),
        "hum_mean": float(hum.mean()),
        "power_mean": float(poww.mean()),
    }

def to_tensor_dict(x: dict):
    # orden fijo para el modelo
    vec = np.array([x[k] for k in FEATURES], dtype=np.float32)
    return vec
# featurizer.py
import numpy as np

FEATURES = [
  "dust_mean","dust_trend","temp_mean","hum_mean","power_mean"
]

def window_features(points: list[dict]) -> dict:
    dust = np.array([p["dustIndex"] for p in points], dtype=float)
    temp = np.array([p["temperature"] for p in points], dtype=float)
    hum  = np.array([p["humidity"]    for p in points], dtype=float)
    poww = np.array([p["powerOutput"] for p in points], dtype=float)

    return {
        "dust_mean": float(dust.mean()),
        "dust_trend": float(dust[-1]-dust[0]),
        "temp_mean": float(temp.mean()),
        "hum_mean": float(hum.mean()),
        "power_mean": float(poww.mean()),
    }

def to_tensor_dict(x: dict):
    # orden fijo para el modelo
    vec = np.array([x[k] for k in FEATURES], dtype=np.float32)
    return vec
