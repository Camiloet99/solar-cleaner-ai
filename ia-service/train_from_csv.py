
# train_from_csv.py
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

FEATURES = ["dust_mean","dust_trend","temp_mean","hum_mean","power_mean"]
CONT_OUT = ["brushRpm01","waterFlow01","pressure01","passes01","detergent01"]
ROUTE_OUT = ["route_keep","route_zigzag_thin","route_zigzag_wide","route_spiral_focus"]

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
        cont = torch.sigmoid(self.head_cont2(torch.relu(self.head_cont1(h_last))))
        logits = self.head_route2(torch.relu(self.head_route1(h_last)))
        route = torch.softmax(logits, dim=-1)
        return cont, route

def zscore_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    return mu, sd

def zscore_apply(X: np.ndarray, mu, sd):
    return (X - mu) / sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--out", default="policy_trained.pt")
    ap.add_argument("--scaler", default="scaler.json")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    X = df[FEATURES].to_numpy(np.float32)
    Yc = df[CONT_OUT].to_numpy(np.float32)
    Yr = df[ROUTE_OUT].to_numpy(np.float32)

    mu, sd = zscore_fit(X)
    Xs = zscore_apply(X, mu, sd)

    X_t = torch.tensor(Xs).unsqueeze(1)  # (N,1,5)
    Yc_t = torch.tensor(Yc)
    Yr_t = torch.tensor(Yr)

    n = len(X_t)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n*0.9)
    tr, va = idx[:split], idx[split:]
    Xtr, Yctr, Yrtr = X_t[tr], Yc_t[tr], Yr_t[tr]
    Xva, Ycva, Yrva = X_t[va], Yc_t[va], Yr_t[va]

    model = PolicyNet()
    opt = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction="batchmean")

    best = 1e9
    patience, bad = 20, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        opt.zero_grad()
        cont, route = model(Xtr)
        loss_cont = mse(cont, Yctr)
        loss_route = kld(torch.log(route + 1e-8), Yrtr)
        loss = loss_cont + 0.5*loss_route
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            c2, r2 = model(Xva)
            v_cont = mse(c2, Ycva).item()
            v_route = kld(torch.log(r2 + 1e-8), Yrva).item()
            v_total = v_cont + 0.5*v_route

        if v_total < best:
            best = v_total
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    torch.save(model.state_dict(), args.out)
    with open(args.scaler, "w") as f:
        json.dump({"means": {k: float(mu[i]) for i,k in enumerate(FEATURES)},
                   "stds":  {k: float(sd[i]) for i,k in enumerate(FEATURES)}}, f, indent=2)
    print(f"Saved model to {args.out} and scaler to {args.scaler}")

if __name__ == "__main__":
    main()
