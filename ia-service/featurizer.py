# featurizer.py
import numpy as np
from typing import Dict, List, Any

# Orden de características esperado por el modelo y por el scaler del training
FEATURES = [
    "dust_mean", "dust_trend", "temp_mean", "hum_mean", "power_mean",
    "dust_last", "dust_delta",
    "prev_rpm01", "prev_water01", "prev_press01", "prev_det01"
]

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def _norm_prev_to_01(p: Dict[str, Any]) -> Dict[str, float]:
    """
    Normaliza parámetros previos a 0..1 usando rangos físicos razonables.
    Acepta claves al nivel raíz o dentro de 'params'.
    - brushRpm       ~ [500, 1200]     -> prev_rpm01
    - waterFlowLpm   ~ [0.10, 0.60]    -> prev_water01
    - nozzlePressureBar ~ [1.2, 2.5]   -> prev_press01
    - detergentPct   ~ [0, 1]          -> prev_det01 (ya normalizado)
    """
    params = p.get("params") if isinstance(p, dict) else None
    src = params if isinstance(params, dict) else p if isinstance(p, dict) else {}

    prpm = float(src.get("brushRpm") or 0.0)
    pwat = float(src.get("waterFlowLpm") or src.get("waterFlow") or 0.0)
    ppres = float(src.get("nozzlePressureBar") or src.get("pressure") or 0.0)
    pdet = float(src.get("detergentPct") or 0.0)

    # Map a 0..1 aprox (clamp para robustez)
    prev_rpm01   = _clip01((prpm - 500.0) / 700.0)     # 500→0, 1200→1
    prev_water01 = _clip01((pwat - 0.10) / 0.50)       # 0.10→0, 0.60→1
    prev_press01 = _clip01((ppres - 1.2) / 1.3)        # 1.2→0, 2.5→1
    prev_det01   = _clip01(pdet)                       # ya 0..1

    return {
        "prev_rpm01": prev_rpm01,
        "prev_water01": prev_water01,
        "prev_press01": prev_press01,
        "prev_det01": prev_det01,
    }

def window_features(points: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calcula features agregadas de ventana + estado previo normalizado.
    Espera que cada punto tenga al menos:
      - dustIndex, temperature, humidity, powerOutput
    Puede incluir parámetros previos en el último punto (nivel raíz o en p['params']).
    """
    if not points:
        # vector neutro si vinieran 0 puntos (el caller valida mínimo 5)
        base = {k: 0.0 for k in FEATURES}
        return base

    # Arrays de ventana
    dust = np.array([float(p["dustIndex"]) for p in points], dtype=float)
    temp = np.array([float(p["temperature"]) for p in points], dtype=float)
    hum  = np.array([float(p["humidity"])    for p in points], dtype=float)
    poww = np.array([float(p["powerOutput"]) for p in points], dtype=float)

    dust_mean  = float(dust.mean())
    dust_trend = float(dust[-1] - dust[0])
    dust_last  = float(dust[-1])
    dust_delta = float(dust[-1] - dust[-2]) if len(dust) >= 2 else 0.0

    prev_feats = _norm_prev_to_01(points[-1])  # usa el último punto de la ventana

    out = {
        "dust_mean":  dust_mean,
        "dust_trend": dust_trend,
        "temp_mean":  float(temp.mean()),
        "hum_mean":   float(hum.mean()),
        "power_mean": float(poww.mean()),
        "dust_last":  dust_last,
        "dust_delta": dust_delta,
        **prev_feats,
    }
    return out

def to_tensor_dict(x: Dict[str, float]):
    """
    Serializa el dict de features a vector numpy en el ORDEN EXACTO de FEATURES.
    Cualquier clave ausente se rellena con 0.0 para robustez.
    """
    vec = np.array([float(x.get(k, 0.0)) for k in FEATURES], dtype=np.float32)
    return vec
