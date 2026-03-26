"""
势场数学 GPU 后端模块

提供多后端支持：
- PyTorch后端：通用CPU/GPU支持
- Triton后端：极致GPU优化（可选）
- 硬件自动检测
"""

from .base import Backend
from .detection import detect_hardware, get_optimal_backend
from .pytorch import PyTorchBackend

__all__ = [
    'Backend',
    'detect_hardware',
    'get_optimal_backend',
    'PyTorchBackend',
]

# 尝试导入Triton后端
try:
    from .triton import TritonBackend
    __all__.append('TritonBackend')
except ImportError:
    pass