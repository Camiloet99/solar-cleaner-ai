# train_from_csv.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# =======================
# Features y salidas
# =======================
# Señales base + estado de la ventana + acciones previas normalizadas (0..1 aprox)
FEATURES = [
    "dust_mean","dust_trend","temp_mean","hum_mean","power_mean",
    "dust_last","dust_delta",
    "prev_rpm01","prev_water01","prev_press01","prev_det01"
]
CONT_OUT = ["brushRpm01","waterFlow01","pressure01","passes01","detergent01"]
ROUTE_OUT = ["route_keep","route_zigzag_thin","route_zigzag_wide","route_spiral_focus"]

# =======================
# Modelo
# =======================
class PolicyNet(nn.Module):
    def __init__(self, in_dim=len(FEATURES), hidden=64):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head_cont1 = nn.Linear(hidden, 64)
        self.head_cont2 = nn.Linear(64, 5)   # brush, water, press, passes, det
        self.head_route1 = nn.Linear(hidden, 64)
        self.head_route2 = nn.Linear(64, 4)  # keep, zig_thin, zig_wide, spiral

    def forward(self, x):  # x: (B,T,in_dim)
        h = torch.relu(self.enc1(x))
        h = torch.relu(self.enc2(h))
        h = self.ln(h)
        out, _ = self.gru(h)
        h_last = out[:, -1, :]
        cont = torch.sigmoid(self.head_cont2(torch.relu(self.head_cont1(h_last))))
        logits = self.head_route2(torch.relu(self.head_route1(h_last)))
        route = torch.softmax(logits, dim=-1)
        return cont, route

# =======================
# Utils
# =======================
def zscore_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd

def zscore_apply(X: np.ndarray, mu, sd):
    return (X - mu) / sd

def _normalize_prev_units_to_01(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte prev_* en 0..1 aprox usando rangos físicos razonables.
    Si ya existen prev_*01, los respeta.
    """
    if "prev_rpm01" not in df.columns and "prev_rpm" in df.columns:
        df["prev_rpm01"] = ((df["prev_rpm"] - 500.0) / 700.0).clip(0, 1)
    if "prev_water01" not in df.columns and "prev_water" in df.columns:
        df["prev_water01"] = ((df["prev_water"] - 0.10) / 0.50).clip(0, 1)
    if "prev_press01" not in df.columns and "prev_press" in df.columns:
        df["prev_press01"] = ((df["prev_press"] - 1.2) / 1.3).clip(0, 1)
    if "prev_det01" not in df.columns and "prev_detergent01" in df.columns:
        df["prev_det01"] = df["prev_detergent01"].clip(0, 1)
    if "prev_det01" not in df.columns and "prev_det" in df.columns:
        df["prev_det01"] = df["prev_det"].clip(0, 1)
    return df

def _ensure_prev_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea columnas prev_*01 si no existen, usando shift por sesión si hay 'sessionId',
    o shift global en su defecto. Se derivan desde las salidas Y (CONT_OUT).
    """
    need_prev = {"prev_rpm01","prev_water01","prev_press01","prev_det01"}
    missing = [c for c in need_prev if c not in df.columns]

    if not missing:
        return _normalize_prev_units_to_01(df)

    # Derivar prev_*01 desde las columnas de salida si existen
    # Mapeo: salida -> prev_ equivalente
    map_out_to_prev = {
        "brushRpm01": "prev_rpm01",
        "waterFlow01": "prev_water01",
        "pressure01": "prev_press01",
        "detergent01": "prev_det01"
    }

    # Orden por sesión si existe
    if "sessionId" in df.columns:
        df = df.sort_values(["sessionId","timestamp"] if "timestamp" in df.columns else ["sessionId"]).reset_index(drop=True)
        group_keys = ["sessionId"]
    else:
        df = df.reset_index(drop=True)
        group_keys = None

    def _shift_series(s: pd.Series) -> pd.Series:
        if group_keys:
            return s.groupby(df[group_keys].apply(lambda r: tuple(r), axis=1)).shift(1)
        else:
            return s.shift(1)

    for out_col, prev_col in map_out_to_prev.items():
        if prev_col not in df.columns:
            if out_col in df.columns:
                df[prev_col] = _shift_series(df[out_col]).fillna(df[out_col])  # primera fila toma su propio valor
            else:
                # si ni siquiera está la salida, crea default 0.5
                df[prev_col] = 0.5

    # Normaliza si vinieron en unidades
    df = _normalize_prev_units_to_01(df)

    # Señales de ventana (si faltan)
    if "dust_last" not in df.columns and "dust_mean" in df.columns and "dust_trend" in df.columns:
        # Mejor estimación si hay columnas de puntos; si no, aproximamos:
        # dust_last ≈ dust_mean + 0.5*dust_trend, dust_delta ≈ dust_trend
        df["dust_last"] = (df["dust_mean"] + 0.5*df["dust_trend"]).astype(float)
    if "dust_delta" not in df.columns and "dust_trend" in df.columns:
        df["dust_delta"] = df["dust_trend"].astype(float)

    return df

# =======================
# Entrenamiento
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="policy_trained.pt")
    ap.add_argument("--scaler", default="scaler.json")
    # hiperparámetros de la política
    ap.add_argument("--target", type=float, default=0.10, help="Objetivo de polvo (%) para dirty mask")
    ap.add_argument("--w_down", type=float, default=5.0, help="Peso penalización por bajar fluidos/pressión en sucio")
    ap.add_argument("--w_couple", type=float, default=3.0, help="Peso coupling RPM→fluidos")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)

    # Completa prev_*01 y señales de ventana si faltan
    df = _ensure_prev_columns(df)

    # Chequeo de columnas requeridas
    for col in FEATURES + CONT_OUT + ROUTE_OUT:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # Matrices
    X = df[FEATURES].to_numpy(np.float32)
    Yc = df[CONT_OUT].to_numpy(np.float32)      # targets continuos (0..1)
    Yr = df[ROUTE_OUT].to_numpy(np.float32)     # targets de ruta (soft)

    # Máscaras y prev de acuerdo al DF (sin normalizar por z-score)
    dust_mean_raw = df["dust_mean"].to_numpy(np.float32)
    dirty_mask_full = (dust_mean_raw > args.target).astype(np.float32)  # (N,)

    prev_rpm01_full   = df["prev_rpm01"].to_numpy(np.float32)
    prev_water01_full = df["prev_water01"].to_numpy(np.float32)
    prev_press01_full = df["prev_press01"].to_numpy(np.float32)
    prev_det01_full   = df["prev_det01"].to_numpy(np.float32)

    # Z-score solo a X
    mu, sd = zscore_fit(X)
    Xs = zscore_apply(X, mu, sd)

    # Tensores
    X_t  = torch.tensor(Xs).unsqueeze(1)  # (N,1,in_dim)
    Yc_t = torch.tensor(Yc)               # (N,5)
    Yr_t = torch.tensor(Yr)               # (N,4)

    dirty_mask_t    = torch.tensor(dirty_mask_full).view(-1, 1)        # (N,1)
    prev_rpm01_t    = torch.tensor(prev_rpm01_full).view(-1, 1)        # (N,1)
    prev_water01_t  = torch.tensor(prev_water01_full).view(-1, 1)      # (N,1)
    prev_press01_t  = torch.tensor(prev_press01_full).view(-1, 1)      # (N,1)
    prev_det01_t    = torch.tensor(prev_det01_full).view(-1, 1)        # (N,1)

    # Split train/val
    n = len(X_t)
    idx = np.arange(n)
    rng = np.random.default_rng(42)
    rng.shuffle(idx)
    split = int(n*0.9)
    tr, va = idx[:split], idx[split:]

    def take(ix):
        return (X_t[ix], Yc_t[ix], Yr_t[ix],
                dirty_mask_t[ix], prev_rpm01_t[ix],
                prev_water01_t[ix], prev_press01_t[ix], prev_det01_t[ix])

    Xtr, Yctr, Yrtr, Dtr, Prpm_tr, Pw_tr, Pp_tr, Pd_tr = take(tr)
    Xva, Ycva, Yrva, Dva, Prpm_va, Pw_va, Pp_va, Pd_va = take(va)

    # Modelo/optimizador
    model = PolicyNet(in_dim=len(FEATURES))
    opt = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction="batchmean")  # para targets soft en ruta

    best = 1e9
    patience, bad = 20, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()

        cont, route = model(Xtr)  # cont: (B,5), route: (B,4)

        # 1) pérdidas base
        loss_cont  = mse(cont, Yctr)
        loss_route = kld(torch.log(route + 1e-8), Yrtr)

        # 2) penalizar bajar agua/presión/detergente cuando está sucio (dirty_mask)
        # columnas: 0 brush, 1 water, 2 press, 3 passes, 4 det
        down_water = torch.relu(Pw_tr - cont[:, 1:2])   # (B,1)
        down_press = torch.relu(Pp_tr - cont[:, 2:3])
        down_det   = torch.relu(Pd_tr - cont[:, 4:5])
        penalty_down = (down_water + down_press + down_det) * Dtr  # (B,1)
        loss_policy = args.w_down * penalty_down.mean()

        # 3) coupling RPM→fluidos: si RPM↑, castigar drop de water/det
        rpm_up = torch.relu(cont[:, 0:1] - Prpm_tr)  # (B,1)
        fluid_drop = torch.relu(Pw_tr - cont[:, 1:2]) + torch.relu(Pd_tr - cont[:, 4:5])
        loss_coupling = args.w_couple * (rpm_up * (fluid_drop + 1e-3)).mean()

        loss = loss_cont + 0.5 * loss_route + loss_policy + loss_coupling
        loss.backward()
        opt.step()

        # ===== Validación =====
        model.eval()
        with torch.no_grad():
            c2, r2 = model(Xva)
            v_cont = mse(c2, Ycva).item()
            v_route = kld(torch.log(r2 + 1e-8), Yrva).item()

            # penalizaciones también en val para el early stopping
            vw_down = torch.relu(Pw_va - c2[:, 1:2])
            vp_down = torch.relu(Pp_va - c2[:, 2:3])
            vd_down = torch.relu(Pd_va - c2[:, 4:5])
            v_penalty_down = (vw_down + vp_down + vd_down) * Dva
            v_loss_policy = args.w_down * v_penalty_down.mean().item()

            v_rpm_up = torch.relu(c2[:, 0:1] - Prpm_va)
            v_fluid_drop = torch.relu(Pw_va - c2[:, 1:2]) + torch.relu(Pd_va - c2[:, 4:5])
            v_loss_coupling = args.w_couple * (v_rpm_up * (v_fluid_drop + 1e-3)).mean().item()

            v_total = v_cont + 0.5 * v_route + v_loss_policy + v_loss_coupling

        if v_total < best:
            best = v_total
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

        if epoch % 20 == 0 or epoch == 1:
            print(f"[{epoch:04d}] "
                  f"train={loss.item():.4f} (cont={loss_cont.item():.4f}, route={loss_route.item():.4f}, "
                  f"down={loss_policy.item():.4f}, couple={loss_coupling.item():.4f}) | "
                  f"val={v_total:.4f}")

    # Guardar
    torch.save(model.state_dict(), args.out)
    with open(args.scaler, "w") as f:
        json.dump({
            "means": {k: float(mu[i]) for i, k in enumerate(FEATURES)},
            "stds":  {k: float(sd[i]) for i, k in enumerate(FEATURES)},
            "features": FEATURES
        }, f, indent=2)

    print(f"Saved model to {args.out} and scaler to {args.scaler}")

if __name__ == "__main__":
    main()
