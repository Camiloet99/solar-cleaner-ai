from typing import Dict, List
import numpy as np

def _clip01(x): return float(np.clip(x, 0.0, 1.0))

def policy(points: List[dict]) -> Dict:
    dust = np.array([p["dustIndex"] for p in points], dtype=float)
    temp = np.array([p["temperature"] for p in points], dtype=float)
    hum  = np.array([p["humidity"]    for p in points], dtype=float)

    dust_mean  = float(np.mean(dust))
    dust_trend = float(dust[-1] - dust[0])   # ~0.13 en tu ejemplo
    temp_mean  = float(np.mean(temp))

    # Normalizaciones “contextuales”
    k_base   = _clip01((dust_mean - 0.20) / 0.60)          # 0.0 @0.2 → 1.0 @0.8
    k_trend  = _clip01(dust_trend / 0.20)                  # 0.0 @0 → 1.0 @0.20
    k_hum    = _clip01((hum.mean() - 0.30) / 0.50) if hum.size else 0.0  # opcional
    k_temp   = 0.0 if temp_mean < 55 else 0.3              # penal térmica suave

    # Curva no lineal (gamma<1 empuja hacia arriba)
    gamma    = 0.6
    k_effort = _clip01(0.45 + 0.55 * (k_base ** gamma) + 0.20 * k_trend - k_temp)
    # pisos por cada actuador (evita esfuerzos ridículos cuando sí hay polvo)
    rpm01    = max(0.55, k_effort)
    flow01   = max(0.50, _clip01(k_effort * 0.95 + 0.05*k_hum))
    press01  = max(0.55, k_effort)
    passes01 = max(0.45, _clip01(0.35 + 0.70*k_effort))
    det01    = _clip01(0.10 * k_effort)

    need_now = (dust_mean > 0.40) or (dust_trend > 0.08)
    rec      = "now" if need_now else "hold_20s"

    proposed = {
        "brushRpm": rpm01,
        "waterFlow": flow01,
        "pressure":  press01,
        "passes":    passes01,
        "detergentPct": det01,
        "route": {
            "keep":          max(0.0, 0.20*(1-k_effort)),
            "zigzag_thin":   0.50 + 0.40*k_effort,
            "zigzag_wide":   0.20*(1-k_effort),
            "spiral_focus":  0.10*k_effort
        }
    }
    # normaliza distribución de ruta a 1.0 (opcional)
    s = sum(proposed["route"].values())
    for k in proposed["route"]:
        proposed["route"][k] /= s

    predicted_loss = round(min(0.45, 0.12 + 0.6*dust_mean + 0.15*k_trend), 3)

    return {
        "recommend": rec,
        "predicted_loss": predicted_loss,
        "route": "zigzag_thin",
        "proposed": proposed,
        "explain": f"dust_mean={dust_mean:.2f}, trend={dust_trend:.2f}, temp={temp_mean:.1f}, eff={k_effort:.2f}"
    }
