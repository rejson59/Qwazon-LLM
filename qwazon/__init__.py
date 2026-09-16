"""
Qwazon LLM - Złoty środek między wydajnością a inteligencją.
Qwazon v0.1 — pierwsza wersja.
"""
from .config import QwazonConfig, QWAZON_VARIANTS
from .model import QwazonModel, QwazonForCausalLM

__version__ = "0.1.0"
__all__ = ["QwazonConfig", "QWAZON_VARIANTS", "QwazonModel", "QwazonForCausalLM"]
