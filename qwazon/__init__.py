"""
Qwazon LLM - Złoty środek między wydajnością a inteligencją.
Qwazon v0.3 — ziemniak trenuje dalej!
"""
from .config import QwazonConfig, QWAZON_VARIANTS, get_config
from .model import QwazonModel, QwazonForCausalLM

try:
    from .lora import LoRAConfig, apply_lora
except ImportError:
    LoRAConfig = None
    apply_lora = None

try:
    from .dpo import DPOTrainer
except ImportError:
    DPOTrainer = None

__version__ = "0.4.0"
__all__ = ["QwazonConfig", "QWAZON_VARIANTS", "get_config", "QwazonModel", "QwazonForCausalLM", "LoRAConfig", "apply_lora", "DPOTrainer"]
