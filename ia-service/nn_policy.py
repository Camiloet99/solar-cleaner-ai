# nn_policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    """
    Entrada: batch x T x 5 (features por tick o una sola ventana resumida)
    Salida:
      - 5 continuas: brushRpm, waterFlow, pressure, passes, detergentPct  (0..1 con Sigmoid)
      - 4 discretas (softmax): route logits -> {keep, zigzag_thin, zigzag_wide, spiral_focus}
    """
    def __init__(self, in_dim=11, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden), nn.GELU()
        )
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head_cont = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 5)
        )
        self.head_route = nn.Sequential(
            nn.Linear(hidden, 64), nn.GELU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):  # x: (B,T,5)
        b, t, d = x.shape
        x = self.enc(x)                 # (B,T,H)
        h0 = torch.zeros(1, b, x.size(-1), device=x.device)
        out, _ = self.gru(x, h0)        # (B,T,H)
        h = out[:, -1, :]               # último paso
        cont = torch.sigmoid(self.head_cont(h))      # (B,5) 0..1
        route_logits = self.head_route(h)            # (B,4)
        route = torch.softmax(route_logits, dim=-1)  # distrib
        return cont, route
