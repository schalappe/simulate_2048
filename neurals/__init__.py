# -*- coding: utf-8 -*-
"""
This module provides functionality for building and using ResNet and attention-based models.
It includes functions for model construction, prediction, and encoding.
"""
from .models import build_resnet_model, build_attention_model
from .network import make_prediction
from .encoded import encode, encode_flatten


__all__ = [
    "build_resnet_model",
    "build_attention_model",
    "make_prediction",
    "encode",
    "encode_flatten",
]
