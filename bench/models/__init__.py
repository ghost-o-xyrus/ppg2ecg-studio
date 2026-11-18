"""Built-in adapters (the app loads models through models/<id>/adapter.py)."""
from .base import ModelAdapter
from .template_baseline import TemplateBaseline
from .zhu_dct import ZhuDCT

__all__ = ["ModelAdapter", "TemplateBaseline", "ZhuDCT"]
