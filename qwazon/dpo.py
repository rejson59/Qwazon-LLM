"""
Qwazon DPO — Direct Preference Optimization dla ziemniaka.
Lżejsze niż PPO/RLHF, bez reward model, bez samplingu.

Idea (Rafailov et al. 2023):
  loss = -log sigmoid( beta * (log πθ(y_w|x) - log πref(y_w|x) - (log πθ(y_l|x) - log πref(y_l|x))) )

Gdzie y_w = preferred (wygrane), y_l = rejected (przegrane).
Uczeń uczy się preferować lepszy kod bez trenowania reward modela.

Użycie:
  from qwazon.dpo import DPOTrainer
  trainer = DPOTrainer(model, ref_model, beta=0.1)
  loss = trainer.step(batch)  # batch: {prompt, chosen, rejected}

Dane PL: UltraFeedback-PL-Code — pary (prompt, good_code, bad_code)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class DPOLoss(nn.Module):
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(self, logp_chosen, logp_rejected, ref_chosen, ref_rejected):
        """
        logp_*: log prob z modelu πθ
        ref_*: log prob z reference model πref (zamrożony)
        Wszystkie: [B]
        """
        # DPO logit
        # pi_logratios = logp - ref
        pi_chosen = logp_chosen - ref_chosen
        pi_rejected = logp_rejected - ref_rejected
        logits = self.beta * (pi_chosen - pi_rejected)
        # loss = -log sigmoid(logits)
        loss = -F.logsigmoid(logits).mean()
        # metrics
        acc = (logits > 0).float().mean()
        # chosen reward = beta * (logp_chosen - ref_chosen)
        reward_chosen = self.beta * pi_chosen
        reward_rejected = self.beta * pi_rejected
        return loss, {"acc": acc.item(), "reward_chosen": reward_chosen.mean().item(), "reward_rejected": reward_rejected.mean().item(), "logits": logits.mean().item()}

def get_logps(logits, labels):
    """
    Policz log prob dla labels.
    logits: [B, T, V], labels: [B, T] (z -100 ignorowanymi)
    Zwraca: [B] średni log prob per sequence (sum / len)
    """
    # shift jak w CE: logits dla next token
    # logits: [B, T, V], labels: [B, T]
    # Chcemy logp dla każdego tokenu != -100
    # Użyj cross_entropy z reduction none
    B, T, V = logits.shape
    # flatten
    logits_flat = logits.view(-1, V)  # [B*T, V]
    labels_flat = labels.view(-1)  # [B*T]
    # mask
    mask = labels_flat != -100
    if mask.sum() == 0:
        return torch.zeros(B, device=logits.device)
    # log_softmax
    log_probs = F.log_softmax(logits_flat, dim=-1)  # [B*T, V]
    # gather
    # labels_flat[mask] w [0, V)
    gathered = log_probs[mask, labels_flat[mask]]  # [N]
    # Teraz need per-sequence sum
    # Zbuduj [B, T] mask 2D
    mask_2d = (labels != -100)  # [B, T]
    # gathered to 1D, musimy podzielić per B
    # Prostsze: policz per batch sum
    logps = []
    idx = 0
    for b in range(B):
        n = mask_2d[b].sum().item()
        if n == 0:
            logps.append(torch.tensor(0.0, device=logits.device))
        else:
            seq_logps = gathered[idx:idx+n]
            # średni log prob per token (stabilniejsze) lub sum? DPO paper używa sum, ale my średnia
            logps.append(seq_logps.sum() / n)
            idx += n
    return torch.stack(logps)  # [B]

class DPOTrainer:
    def __init__(self, model: nn.Module, ref_model: Optional[nn.Module] = None, beta: float = 0.1, device: str = "cpu"):
        self.model = model
        self.ref_model = ref_model
        self.beta = beta
        self.device = device
        self.loss_fn = DPOLoss(beta=beta)
        if ref_model is not None:
            for p in ref_model.parameters():
                p.requires_grad = False
            ref_model.eval()

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict:
        """
        batch: {
          "chosen_input_ids": [B, T], "chosen_labels": [B, T], "chosen_attention_mask": [B, T],
          "rejected_input_ids": [B, T], ...
        }
        """
        self.model.train()
        # forward chosen
        out_chosen = self.model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
        out_rejected = self.model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])
        logp_chosen = get_logps(out_chosen["logits"], batch["chosen_labels"])
        logp_rejected = get_logps(out_rejected["logits"], batch["rejected_labels"])

        with torch.no_grad():
            if self.ref_model is not None:
                ref_chosen = self.ref_model(input_ids=batch["chosen_input_ids"], attention_mask=batch["chosen_attention_mask"])
                ref_rejected = self.ref_model(input_ids=batch["rejected_input_ids"], attention_mask=batch["rejected_attention_mask"])
                ref_logp_chosen = get_logps(ref_chosen["logits"], batch["chosen_labels"])
                ref_logp_rejected = get_logps(ref_rejected["logits"], batch["rejected_labels"])
            else:
                # bez ref = 0 (jak w SimPO)
                ref_logp_chosen = torch.zeros_like(logp_chosen)
                ref_logp_rejected = torch.zeros_like(logp_rejected)

        loss, metrics = self.loss_fn(logp_chosen, logp_rejected, ref_logp_chosen, ref_logp_rejected)
        return {"loss": loss, "metrics": metrics, "logp_chosen": logp_chosen.mean().item(), "logp_rejected": logp_rejected.mean().item()}

# Demo
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from qwazon.config import get_config
    from qwazon.model import QwazonForCausalLM
    cfg = get_config("qwazon-nano")
    model = QwazonForCausalLM(cfg)
    ref = QwazonForCausalLM(cfg)
    trainer = DPOTrainer(model, ref, beta=0.1)
    # dummy batch
    B, T, V = 2, 16, cfg.vocab_size
    batch = {
        "chosen_input_ids": torch.randint(0, V, (B, T)),
        "chosen_labels": torch.randint(0, V, (B, T)),
        "chosen_attention_mask": torch.ones(B, T, dtype=torch.long),
        "rejected_input_ids": torch.randint(0, V, (B, T)),
        "rejected_labels": torch.randint(0, V, (B, T)),
        "rejected_attention_mask": torch.ones(B, T, dtype=torch.long),
    }
    # ustaw -100 na pad
    batch["chosen_labels"][:, -2:] = -100
    batch["rejected_labels"][:, -2:] = -100
    out = trainer.step(batch)
    print("DPO loss:", out["loss"].item(), "acc:", out["metrics"]["acc"])
