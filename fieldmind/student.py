"""A tiny new policy with optional Intel XPU training/inference (not P2P weights)."""
from __future__ import annotations
import json
from pathlib import Path
from .core import ACTIONS, FEATURE_DIM


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("Install PyTorch (Intel XPU build if available) for student training") from exc
    return torch, nn


def choose_device(prefer_xpu: bool = True) -> str:
    torch, _ = _torch()
    return "xpu" if prefer_xpu and hasattr(torch,"xpu") and torch.xpu.is_available() else "cpu"


def create_model():
    _, nn = _torch()
    return nn.Sequential(nn.Linear(FEATURE_DIM, 96), nn.SiLU(),
                         nn.Linear(96, 64), nn.SiLU(), nn.Linear(64, len(ACTIONS)))


class Student:
    def __init__(self, checkpoint: str, prefer_xpu: bool = True):
        torch, _ = _torch()
        self.device = choose_device(prefer_xpu)
        self.torch = torch
        self.model = create_model().to(self.device)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload.get("actions") != list(ACTIONS) or payload.get("features") != FEATURE_DIM:
            raise ValueError("Checkpoint schema is incompatible")
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    def predict(self, observation: list[float]) -> tuple[str, float]:
        with self.torch.no_grad():
            x = self.torch.tensor([observation], dtype=self.torch.float32, device=self.device)
            prob = self.model(x).softmax(dim=1)[0]
            i = int(prob.argmax().item())
            return ACTIONS[i], float(prob[i].item())


def train(jsonl: str, output: str, epochs: int = 30, prefer_xpu: bool = True,
          batch_size: int = 128, lr: float = 1e-3) -> dict:
    """Distil expert action labels or supervised demonstrations (JSONL).

    Each line: {"features": [20 float values], "action": "pass"}.
    No claim of pretrained OpenP2P weights or online RL.
    """
    torch, nn = _torch()
    data = []
    for lineno, line in enumerate(Path(jsonl).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        item = json.loads(line)
        if len(item["features"]) != FEATURE_DIM or item["action"] not in ACTIONS:
            raise ValueError(f"Invalid example at line {lineno}")
        data.append((list(map(float,item["features"])), ACTIONS.index(item["action"])))
    if len(data) < 10:
        raise ValueError("Need at least 10 labeled states; real skill requires much more")
    device = choose_device(prefer_xpu)
    model = create_model().to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=lr)
    x = torch.tensor([d[0] for d in data], dtype=torch.float32)
    y = torch.tensor([d[1] for d in data], dtype=torch.long)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(data))
        for ids in order.split(batch_size):
            bx, by = x[ids].to(device), y[ids].to(device)
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(bx), by)
            loss.backward()
            opt.step()
    model.eval()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"actions":list(ACTIONS),"features":FEATURE_DIM,
                "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}},output)
    n = sum(p.numel() for p in model.parameters())
    return {"examples":len(data),"device":device,"parameters":n,"final_batch_loss":float(loss.item())}
