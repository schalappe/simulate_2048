# -*- coding: utf-8 -*-
"""
Set of class for sharing the network between the self-play and training jobs.
"""
from dataclasses import dataclass

import tensorflow as tf


@dataclass
class NetworkCacher:
    """Class ot share network between self-play and training jobs."""

    network: tf.keras.Model
    optimizer: tf.keras.optimizers.Optimizer
