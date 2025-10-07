# policy_loader.py
import os, torch
from nn_policy import PolicyNet

def load_policy():
    path = os.getenv("POLICY_PATH", "").strip()
    model = PolicyNet()
    model.eval()
    if path and os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    return model
