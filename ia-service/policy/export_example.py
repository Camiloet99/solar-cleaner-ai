# export_example.py
# Genera dos modelos .pt para /predict:
#  - policy_seed.pt         (heurística "normal")
#  - policy_aggressive.pt   (sale más alto en 0..1 → más rpm/flow/passes tras desnormalizar)

import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from pathlib import Path

# ======= Mismo modelo que nn_policy.py =======
class PolicyNet(nn.Module):
    def __init__(self, in_dim=5, hidden=64):
        super().__init__()
        self.enc1 = nn.Linear(in_dim, hidden)
        self.enc2 = nn.Linear(hidden, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head_cont1 = nn.Linear(hidden, 64)
        self.head_cont2 = nn.Linear(64, 5)
        self.head_route1 = nn.Linear(hidden, 64)
        self.head_route2 = nn.Linear(64, 4)

    def forward(self, x):  # x: (B,T,5)
        h = torch.relu(self.enc1(x))
        h = torch.relu(self.enc2(h))
        h = self.ln(h)
        out, _ = self.gru(h)
        h_last = out[:, -1, :]
        cont = torch.sigmoid(self.head_cont2(torch.relu(self.head_cont1(h_last))))  # 0..1
        logits = self.head_route2(torch.relu(self.head_route1(h_last)))
        route = torch.softmax(logits, dim=-1)  # distribución 4 clases
        return cont, route

# ======= Heurística que imitamos (targets) =======
def _clip01(x): return max(0.0, min(1.0, float(x)))

def targets_from_features(feat, aggressive=False):
    """
    feat = [dust_mean, dust_trend, temp_mean, hum_mean, power_mean]
    Devuelve:
      cont: [brushRpm01, waterFlow01, pressure01, passes01, detergent01]
      route: [keep, zigzag_thin, zigzag_wide, spiral_focus] (probabilidades)
    """
    dust_mean, dust_trend, temp_mean, hum_mean, _ = feat
    k_base   = _clip01((dust_mean - 0.20) / 0.60)
    k_trend  = _clip01(dust_trend / 0.20)
    k_temp   = 0.0 if temp_mean < 55 else 0.3

    gamma    = 0.6 if aggressive else 0.8
    offset   = 0.45 if aggressive else 0.35
    scale    = 0.55 if aggressive else 0.45
    k_eff    = _clip01(offset + scale*(k_base**gamma) + 0.20*k_trend - k_temp)

    floor_rpm   = 0.60 if aggressive else 0.50
    floor_flow  = 0.55 if aggressive else 0.45
    floor_press = 0.60 if aggressive else 0.50
    floor_pass  = 0.50 if aggressive else 0.35

    rpm01    = max(floor_rpm, k_eff)
    flow01   = max(floor_flow, _clip01(k_eff*0.95 + 0.05*_clip01((hum_mean-30)/50)))
    press01  = max(floor_press, k_eff)
    passes01 = max(floor_pass, _clip01(0.35 + 0.70*k_eff))
    det01    = _clip01(0.10 * k_eff)

    keep = max(0.0, (0.20 if aggressive else 0.30) * (1-k_eff))
    zigthin = 0.50 + 0.40*k_eff
    zigwide = 0.20*(1-k_eff)
    spiral  = 0.10*k_eff
    s = keep + zigthin + zigwide + spiral
    route = np.array([keep/s, zigthin/s, zigwide/s, spiral/s], dtype=np.float32)
    cont = np.array([rpm01, flow01, press01, passes01, det01], dtype=np.float32)
    return cont, route

# ======= Dataset sintético (simula ventanas agregadas) =======
def sample_feature():
    dust_mean  = np.random.beta(2, 2)                       # 0..1
    dust_trend = float(np.clip(np.random.normal(0.05, 0.05), -0.10, 0.30))
    temp_mean  = float(np.clip(np.random.normal(38, 8), 10, 75))
    hum_mean   = float(np.clip(np.random.normal(55, 15), 5, 95))
    power_mean = float(np.clip(np.random.normal(180, 25), 80, 320))
    return np.array([dust_mean, dust_trend, temp_mean, hum_mean, power_mean], dtype=np.float32)

def make_batch(N=2048, aggressive=False):
    X, Yc, Yr = [], [], []
    for _ in range(N):
        f = sample_feature()
        c, r = targets_from_features(f, aggressive=aggressive)
        X.append(f); Yc.append(c); Yr.append(r)
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1)  # (N,1,5)
    Yc = torch.tensor(np.array(Yc), dtype=torch.float32)             # (N,5)
    Yr = torch.tensor(np.array(Yr), dtype=torch.float32)             # (N,4)
    return X, Yc, Yr

# ======= Entrenamiento corto =======
def train_and_save(out_path: str, aggressive=False, epochs=200, lr=3e-3, seed=42):
    torch.manual_seed(seed + (1 if aggressive else 0))
    model = PolicyNet()
    opt = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction="batchmean")

    X, Yc, Yr = make_batch(N=4096, aggressive=aggressive)
    for e in range(epochs):
        model.train()
        opt.zero_grad()
        cont, route = model(X)
        loss_cont = mse(cont, Yc)
        loss_route = kld(torch.log(route + 1e-8), Yr)
        loss = loss_cont + 0.5*loss_route
        loss.backward()
        opt.step()
        # print(f"epoch {e+1}/{epochs} -> cont {loss_cont.item():.4f} route {loss_route.item():.4f}")

    Path("./").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")

    # Pequeño sanity check
    with torch.no_grad():
        test = torch.tensor([[ [0.50, 0.10, 35.0, 50.0, 180.0] ]], dtype=torch.float32)  # (1,1,5)
        c, r = model(test)
        print("Sample outputs (0..1):", c.squeeze(0).numpy().round(3).tolist(), "route:", r.squeeze(0).numpy().round(3).tolist())

if __name__ == "__main__":
    train_and_save("policy_seed.pt", aggressive=False, epochs=200, lr=3e-3, seed=42)
    train_and_save("policy_aggressive.pt", aggressive=True,  epochs=260, lr=3e-3, seed=1337)
