from fastapi import FastAPI
from fastapi.responses import JSONResponse
from models import PredictReq, PredictResp
from datetime import datetime, timezone
import numpy as np, torch, os
from typing import Dict, Any

from featurizer import window_features, to_tensor_dict
from policy_loader import load_policy

# =========================
# Config por variables de entorno (con defaults sensatos)
# =========================
USE_NN = os.getenv("USE_NN", "true").lower() == "true"
TARGET_DUST_PCT = float(os.getenv("TARGET_DUST_PCT", "0.10"))              # objetivo intermedio (10%)
TARGET_DUST_PCT_FINAL = float(os.getenv("TARGET_DUST_PCT_FINAL", "0.02"))   # objetivo final (~2%)
TARGET_HYS = float(os.getenv("TARGET_HYS", "0.005"))                        # histéresis (0.5%)

# mínimos en 0..1 cuando el panel está por encima del target (para la máscara suave)
MIN_WATER01 = float(os.getenv("MIN_WATER01", "0.35"))
MIN_PRESS01 = float(os.getenv("MIN_PRESS01", "0.35"))
MIN_DET01   = float(os.getenv("MIN_DET01",   "0.25"))

# acople RPM→fluidos (ganancias en 0..1)
COUPLING_GAIN_RPM   = float(os.getenv("COUPLING_GAIN_RPM", "0.10"))      # agua por alza relativa de RPM
COUPLING_DET_FACTOR = float(os.getenv("COUPLING_DET_FACTOR", "0.60"))    # proporción para detergente

# Umbrales para "NOW"
NOW_DUST_THR   = float(os.getenv("NOW_DUST_THR", "0.40"))
NOW_TREND_THR  = float(os.getenv("NOW_TREND_THR", "0.08"))

# ====== Motor de escalamiento estatal ======
MIN_DELTA_EXPECTED = float(os.getenv("MIN_DELTA_EXPECTED", "0.05"))        # 5% mejora mínima por ventana
MAX_ESCALATION_LEVEL = int(os.getenv("MAX_ESCALATION_LEVEL", "6"))         # niveles máximos de escalamiento
STUCK_WINDOWS_FOR_ESCALATE = int(os.getenv("STUCK_WINDOWS_FOR_ESCALATE", "1"))

# Factores por nivel (0..N). Puedes tunear vía env si quieres.
# (water +, det ×, press +, contact +s, passes +, speed -)
ESCALATION_TABLE = [
    (0.00, 1.00, 0.00, 0, 0, 0.00),
    (0.05, 1.10, 0.10, 1, 0, 0.03),
    (0.08, 1.15, 0.15, 2, 1, 0.05),
    (0.12, 1.20, 0.20, 3, 1, 0.07),
    (0.15, 1.25, 0.25, 4, 2, 0.10),
    (0.20, 1.30, 0.30, 5, 2, 0.12),
    (0.25, 1.35, 0.35, 6, 2, 0.15),
]

# ====== Empuje específico de RPM ======
# Paso base de incremento de RPM (en espacio 0..1), y pisos por bandas de polvo
RPM_STEP_UP_BASE = float(os.getenv("RPM_STEP_UP_BASE", "0.08"))   # incremento mínimo por ventana si sigue sucio
RPM_STEP_LVL_GAIN = float(os.getenv("RPM_STEP_LVL_GAIN", "0.04")) # adicional por nivel de escalamiento
# pisos por banda (si dust_mean >= band -> rpm01 >= floor)
RPM_FLOOR_AT_20 = float(os.getenv("RPM_FLOOR_AT_20", "0.80"))     # si polvo ≥ 20% → rpm01 ≥ 0.80
RPM_FLOOR_AT_12 = float(os.getenv("RPM_FLOOR_AT_12", "0.70"))     # si polvo ≥ 12% → rpm01 ≥ 0.70

# =========================

model = load_policy() if USE_NN else None
app = FastAPI(title="PV Cleaning AI", version="2.2.0-nn-escalation-rpm")

# Estado por sesión (en memoria)
SESS: Dict[str, Dict[str, Any]] = {}

def _heuristic(points):
    dust = np.array([p["dustIndex"] for p in points], dtype=float)
    dust_mean = float(dust.mean())
    dust_trend = float(dust[-1] - dust[0])
    k = float(np.clip((dust_mean - 0.20)/0.60, 0.0, 1.0))
    need_now = (dust_mean > NOW_DUST_THR) or (dust_trend > NOW_TREND_THR)
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
def health():
    return {"status":"ok","version":app.version,"use_nn":USE_NN}

def _apply_nn_guard_and_coupling(a01: Dict[str,float], dust_mean: float) -> Dict[str,float]:
    """No bajar fluidos/pressión mientras siga sucio + coupling RPM→fluidos."""
    if dust_mean > (TARGET_DUST_PCT + TARGET_HYS):
        a01["water01"] = max(a01["water01"], MIN_WATER01)
        a01["press01"] = max(a01["press01"], MIN_PRESS01)
        a01["det01"]   = max(a01["det01"],   MIN_DET01)

    if a01["brushRpm01"] > 0.5 and dust_mean > (TARGET_DUST_PCT + TARGET_HYS):
        rpm_up = a01["brushRpm01"] - 0.5
        a01["water01"] = float(np.clip(a01["water01"] + COUPLING_GAIN_RPM * rpm_up, 0.0, 1.0))
        a01["det01"]   = float(np.clip(a01["det01"]   + COUPLING_DET_FACTOR * COUPLING_GAIN_RPM * rpm_up, 0.0, 1.0))
    return a01

def _rpm_floor_for_dust(dust_mean: float) -> float:
    """Pisos de RPM por bandas de polvo (0..1)."""
    if dust_mean >= 0.20:
        return RPM_FLOOR_AT_20
    if dust_mean >= 0.12:
        return RPM_FLOOR_AT_12
    return 0.0

def _escalate_towards_zero(a01: Dict[str,float], state: Dict[str,Any], dust_mean: float, prev_rpm01: float) -> Dict[str,float]:
    """
    Si Δ es pobre o seguimos por encima del objetivo final, sube agresividad por niveles.
    - Escala agua, detergente, presión, contacto y pases; reduce velocidad implícita (la decide el BE).
    - Empuja RPM con monotonicidad (no volver a bajar mientras siga sucio) y con pisos por banda.
    """
    lvl = int(state.get("escalation_level", 0))
    stuck = int(state.get("stuck_windows", 0))

    # Coalesce when prev_dust is None
    pd = state.get("prev_dust")
    prev_dust = float(dust_mean if pd is None else pd)
    delta = prev_dust - dust_mean  # mejora positiva cuando delta > 0

    need_escalate = (dust_mean > TARGET_DUST_PCT_FINAL) and (
        delta < MIN_DELTA_EXPECTED or stuck >= STUCK_WINDOWS_FOR_ESCALATE
    )
    if need_escalate and lvl < MAX_ESCALATION_LEVEL:
        lvl += 1
        stuck = 0

    # ---- Escalado de fluidos/pressión/contacto/pases/speed
    if dust_mean > TARGET_DUST_PCT_FINAL:
        water_add, det_mul, press_add, contact_add, passes_add, speed_down = ESCALATION_TABLE[min(lvl, len(ESCALATION_TABLE)-1)]
        a01["water01"] = float(np.clip(a01["water01"] + water_add, 0.0, 1.0))
        a01["det01"]   = float(np.clip(a01["det01"]   * det_mul,   0.0, 1.0))
        a01["press01"] = float(np.clip(a01["press01"] + press_add, 0.0, 1.0))
        state["contact_boost"] = int(state.get("contact_boost", 0)) + int(contact_add)
        state["passes_boost"]  = int(state.get("passes_boost", 0))  + int(passes_add)
        state["speed_down"]    = float(state.get("speed_down", 0.0)) + float(speed_down)

    # ---- Empuje específico de RPM (monotónico + pisos por polvo)
    if dust_mean > TARGET_DUST_PCT_FINAL:
        rpm_floor = _rpm_floor_for_dust(dust_mean)
        # Paso de aumento crece con el nivel para salir del "atasco"
        rpm_step  = RPM_STEP_UP_BASE + lvl * RPM_STEP_LVL_GAIN
        rpm_target = max(a01["brushRpm01"], prev_rpm01 + rpm_step, rpm_floor)
        a01["brushRpm01"] = float(np.clip(rpm_target, 0.0, 1.0))

    # ---- Actualiza estado
    if delta < MIN_DELTA_EXPECTED and dust_mean > TARGET_DUST_PCT_FINAL:
        stuck += 1
    else:
        stuck = 0

    state["escalation_level"] = lvl
    state["stuck_windows"] = stuck
    state["prev_dust"] = dust_mean

    return a01

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq):
    if len(req.points) < 5:
        return JSONResponse(status_code=400, content={"error":"At least 5 points required"})

    pts = [p.model_dump() for p in req.points]
    session_id = req.points[-1].sessionId
    st = SESS.setdefault(session_id, {"prev_dust": None, "stuck_windows": 0, "escalation_level": 0})

    if USE_NN and model is not None:
        # ===== Featurización =====
        feats = window_features(pts)
        dust_mean = float(feats.get("dust_mean", 0.0))
        dust_trend = float(feats.get("dust_trend", 0.0))

        # Inicializa prev_dust la primera vez
        if st.get("prev_dust") is None:
            st["prev_dust"] = dust_mean

        # Tomamos rpm previo normalizado del featurizer (0..1)
        prev_rpm01 = float(feats.get("prev_rpm01", 0.5))

        x = to_tensor_dict(feats)                             # (in_dim,)
        x = torch.tensor(x, dtype=torch.float32).view(1,1,-1) # (1,1,in_dim)

        # ===== Inferencia NN =====
        with torch.no_grad():
            cont, route = model(x)
        brushRpm01, water01, press01, passes01, det01 = cont[0].tolist()
        keep, zigthin, zigwide, spiral = route[0].tolist()

        # Ruta normalizada
        route_soft = {
            "keep": float(keep),
            "zigzag_thin": float(zigthin),
            "zigzag_wide": float(zigwide),
            "spiral_focus": float(spiral)
        }
        s = sum(route_soft.values()) or 1.0
        for k in route_soft: route_soft[k] /= s

        # Acción 0..1 base
        a01 = {
            "brushRpm01": float(brushRpm01),
            "water01":    float(water01),
            "press01":    float(press01),
            "passes01":   float(passes01),
            "det01":      float(det01)
        }

        # 1) Guard IA + coupling (no bajar fluidos/pressión + acople RPM→fluidos)
        a01 = _apply_nn_guard_and_coupling(a01, dust_mean)

        # 2) Escalamiento hacia 0 (drive-to-zero con memoria por sesión) + empuje RPM
        a01 = _escalate_towards_zero(a01, st, dust_mean, prev_rpm01)

        # Empaquetar propuesta 0..1 al BE
        proposed = {
            "brushRpm":     a01["brushRpm01"],
            "waterFlow":    a01["water01"],
            "pressure":     a01["press01"],
            "passes":       a01["passes01"],     # BE puede mapear a enteros si requiere
            "detergentPct": a01["det01"],
            "route":        route_soft
        }

        # Recomendación de frecuencia:
        # Mientras no estemos por debajo del objetivo FINAL con histéresis, seguimos "now"
        if dust_mean > (TARGET_DUST_PCT_FINAL + TARGET_HYS):
            rec = "now"
        else:
            rec = "hold_20s" if dust_trend <= 0 else "after_2_windows"

        # Pérdida estimada (cosmética)
        pred_loss = float(min(0.45, 0.12 + 0.6*dust_mean + 0.15*max(0.0, dust_trend)))

        explain = (
            f"NN+escalation | dust_mean={dust_mean:.3f}, trend={dust_trend:.3f}, "
            f"lvl={st.get('escalation_level')}, stuck={st.get('stuck_windows')}, "
            f"targets(final={TARGET_DUST_PCT_FINAL:.3f}), boosts(c={st.get('contact_boost',0)},"
            f" p={st.get('passes_boost',0)}, v-={st.get('speed_down',0.0):.2f})"
        )

    else:
        rec, proposed, dust_mean, dust_trend = _heuristic(pts)
        # Aun con heurística, drive-to-zero:
        if dust_mean > (TARGET_DUST_PCT_FINAL + TARGET_HYS):
            rec = "now"
        pred_loss = float(min(0.35, 0.15 + 0.5*dust_mean))
        explain = f"Heuristic | dust_mean={dust_mean:.2f}, trend={dust_trend:.2f}"

    return {
        "sessionId": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "predictedEfficiencyLoss": round(pred_loss, 3),
        "recommendedCleaningFrequency": rec,
        "cleaningRouteAdjustment": max(proposed["route"], key=proposed["route"].get),
        "alerts": ["dust_high"] if rec == "now" else [],
        "proposedCommands": proposed,
        "explain": explain
    }
