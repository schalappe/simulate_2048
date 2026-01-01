"""
Hardware accelerator detection and distribution strategy for training.

This module provides automatic detection and configuration for:
- TPU (Tensor Processing Units) - available on Kaggle/Colab
- GPU (NVIDIA CUDA devices)
- CPU fallback

The AcceleratorStrategy class handles:
1. Hardware detection in priority order: TPU > GPU > CPU
2. Distribution strategy setup for TensorFlow/Keras
3. Recommended batch size multipliers for each hardware type
4. Model building under the appropriate distribution scope

Reference: TensorFlow Distribution Strategy Guide
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import tensorflow as tf
from keras import Model

logger = logging.getLogger(__name__)


class AcceleratorType(Enum):
    """Available hardware accelerator types."""

    CPU = 'cpu'
    GPU = 'gpu'
    TPU = 'tpu'


@dataclass
class AcceleratorInfo:
    """
    Information about detected accelerator.

    Attributes
    ----------
    accelerator_type : AcceleratorType
        The type of accelerator detected.
    device_count : int
        Number of devices available.
    memory_per_device : int | None
        Memory per device in MB (if available).
    name : str
        Human-readable name of the accelerator.
    """

    accelerator_type: AcceleratorType
    device_count: int
    memory_per_device: int | None
    name: str


class AcceleratorStrategy:
    """
    Manages hardware acceleration for training and inference.

    This class automatically detects available hardware and configures
    the appropriate TensorFlow distribution strategy.

    Attributes
    ----------
    accelerator : AcceleratorType
        The detected accelerator type.
    strategy : tf.distribute.Strategy
        The configured distribution strategy.
    info : AcceleratorInfo
        Detailed information about the accelerator.

    Examples
    --------
    >>> strategy = AcceleratorStrategy(prefer_tpu=True)
    >>> print(f"Using: {strategy.accelerator.value}")
    >>> model = strategy.build_model(build_my_model)
    """

    def __init__(self, prefer_tpu: bool = True, force_cpu: bool = False):
        """
        Initialize accelerator strategy with automatic detection.

        Parameters
        ----------
        prefer_tpu : bool
            Whether to prefer TPU over GPU when both are available.
        force_cpu : bool
            Force CPU usage regardless of available accelerators.
        """
        self._prefer_tpu = prefer_tpu
        self._force_cpu = force_cpu

        if force_cpu:
            self.accelerator = AcceleratorType.CPU
            self.info = AcceleratorInfo(
                accelerator_type=AcceleratorType.CPU, device_count=1, memory_per_device=None, name='CPU (forced)'
            )
            self.strategy = self._setup_strategy()
        else:
            self.accelerator, self.info = self._detect_accelerator()
            self.strategy = self._setup_strategy()

        logger.info('Accelerator: %s (%d devices)', self.info.name, self.info.device_count)

    def _detect_accelerator(self) -> tuple[AcceleratorType, AcceleratorInfo]:
        """
        Detect the best available hardware accelerator.

        Returns
        -------
        tuple[AcceleratorType, AcceleratorInfo]
            The detected accelerator type and its information.
        """
        # ##>: Check TPU first if preferred (common in Kaggle/Colab).
        if self._prefer_tpu:
            tpu_info = self._detect_tpu()
            if tpu_info is not None:
                return AcceleratorType.TPU, tpu_info

        # ##>: Check GPU availability.
        gpu_info = self._detect_gpu()
        if gpu_info is not None:
            return AcceleratorType.GPU, gpu_info

        # ##>: Check TPU if not preferred but GPU unavailable.
        if not self._prefer_tpu:
            tpu_info = self._detect_tpu()
            if tpu_info is not None:
                return AcceleratorType.TPU, tpu_info

        # ##>: Fallback to CPU.
        return AcceleratorType.CPU, AcceleratorInfo(
            accelerator_type=AcceleratorType.CPU, device_count=1, memory_per_device=None, name='CPU'
        )

    def _detect_tpu(self) -> AcceleratorInfo | None:
        """
        Detect TPU availability (Kaggle/Colab environments).

        Returns
        -------
        AcceleratorInfo | None
            TPU information if available, None otherwise.
        """
        try:
            # ##>: Try to resolve TPU cluster (works on Kaggle/Colab).
            tpu_address = os.environ.get('TPU_NAME')
            if tpu_address:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
            else:
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()

            # ##>: Connect and initialize TPU.
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)

            # ##>: Get TPU topology for device count.
            # ##&: TPU v3-8 on Kaggle has 8 cores.
            tpu_topology = resolver.cluster_spec().as_dict()
            device_count = len(tpu_topology.get('worker', [1])) * 8  # Each worker has 8 cores

            return AcceleratorInfo(
                accelerator_type=AcceleratorType.TPU,
                device_count=device_count,
                memory_per_device=16 * 1024,  # TPU v3 has 16GB per core
                name=f'TPU v3 ({device_count} cores)',
            )

        except (ValueError, RuntimeError) as e:
            logger.debug('TPU not available: %s', e)
            return None
        except ImportError:
            logger.debug('TensorFlow not available for TPU detection')
            return None

    def _detect_gpu(self) -> AcceleratorInfo | None:
        """
        Detect GPU availability.

        Returns
        -------
        AcceleratorInfo | None
            GPU information if available, None otherwise.
        """
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                return None

            # ##>: Get GPU memory info if available.
            memory_per_device = None
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                # ##&: Memory info requires CUDA runtime.
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                if 'memory_limit' in gpu_details:
                    memory_per_device = int(gpu_details['memory_limit'] / (1024 * 1024))
            except (RuntimeError, KeyError):
                pass

            # ##>: Build device name.
            try:
                gpu_details = tf.config.experimental.get_device_details(gpus[0])
                gpu_name = gpu_details.get('device_name', 'GPU')
            except (RuntimeError, KeyError):
                gpu_name = 'GPU'

            return AcceleratorInfo(
                accelerator_type=AcceleratorType.GPU,
                device_count=len(gpus),
                memory_per_device=memory_per_device,
                name=f'{gpu_name} ({len(gpus)} device{"s" if len(gpus) > 1 else ""})',
            )

        except ImportError:
            logger.debug('TensorFlow not available for GPU detection')
            return None

    def _setup_strategy(self):
        """
        Configure distribution strategy for detected hardware.

        Returns
        -------
        tf.distribute.Strategy
            The configured distribution strategy.
        """
        if self.accelerator == AcceleratorType.TPU:
            try:
                tpu_address = os.environ.get('TPU_NAME')
                if tpu_address:
                    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
                else:
                    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                return tf.distribute.TPUStrategy(resolver)
            except (ValueError, RuntimeError):
                logger.warning('Failed to create TPU strategy, falling back to default')
                return tf.distribute.get_strategy()

        if self.accelerator == AcceleratorType.GPU:
            gpus = tf.config.list_physical_devices('GPU')
            if len(gpus) > 1:
                # ##>: MirroredStrategy for multi-GPU training.
                return tf.distribute.MirroredStrategy()
            # ##>: Single GPU uses default strategy.
            return tf.distribute.get_strategy()

        # ##>: CPU fallback.
        return tf.distribute.get_strategy()

    def build_model(self, model_fn: Callable[[], Model]) -> Model:
        """
        Build a model under the appropriate distribution scope.

        Parameters
        ----------
        model_fn : Callable[[], Model]
            A function that creates and returns a Keras model.

        Returns
        -------
        Model
            The model built under the distribution strategy scope.

        Examples
        --------
        >>> def create_model():
        ...     return keras.Sequential([keras.layers.Dense(10)])
        >>> model = strategy.build_model(create_model)
        """
        with self.strategy.scope():
            return model_fn()

    @property
    def batch_size_multiplier(self) -> int:
        """
        Recommended batch size multiplier for the hardware.

        TPUs work best with large batches (128+), GPUs benefit from
        moderate increases, CPUs should use smaller batches.

        Returns
        -------
        int
            Multiplier to apply to base batch size.
        """
        if self.accelerator == AcceleratorType.TPU:
            return 8  # e.g., base 16 -> 128
        if self.accelerator == AcceleratorType.GPU:
            return 2  # e.g., base 16 -> 32
        return 1  # CPU keeps base batch size

    @property
    def recommended_batch_size(self) -> int:
        """
        Recommended batch size for MCTS parallel trajectories.

        Returns
        -------
        int
            Recommended number of parallel trajectories.
        """
        base_batch = 16
        return base_batch * self.batch_size_multiplier

    def get_compile_kwargs(self) -> dict:
        """
        Get recommended compile kwargs for the accelerator.

        Returns
        -------
        dict
            Keyword arguments to pass to model.compile().
        """
        if self.accelerator == AcceleratorType.TPU:
            # ##>: TPUs require specific settings for optimal performance.
            return {'jit_compile': True, 'steps_per_execution': 32}
        if self.accelerator == AcceleratorType.GPU:
            return {'jit_compile': True}
        return {}


def get_default_accelerator() -> AcceleratorStrategy:
    """
    Get a default accelerator strategy with auto-detection.

    Returns
    -------
    AcceleratorStrategy
        Strategy with automatic hardware detection.
    """
    return AcceleratorStrategy(prefer_tpu=True, force_cpu=False)
