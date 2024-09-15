# -*- coding: utf-8 -*-
"""
This module provides functionality for building and using ResNet and attention-based models.
"""
from .models import build_attention_model, build_model_with_identity_blocks
from .network import make_prediction


__all__ = [
    "build_attention_model",
    "build_model_with_identity_blocks",
    "make_prediction",
]
