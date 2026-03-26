"""硬件检测与后端选择"""

import torch
import warnings
from typing import Dict, Any
from .base import Backend


def detect_hardware() -> Dict[str, Any]:
    """
    检测硬件并返回最优配置
    
    返回:
        dict: 包含硬件信息和推荐配置
    """
    config = {
        'backend': 'pytorch',
        'device': 'cpu',
        'dtype': torch.float32,
        'block_size': 512,
        'supports_fp16': False,
        'supports_tf32': False,
        'vendor': 'unknown',
        'device_name': 'CPU',
        'memory_gb': 0,
    }
    
    # 检测CUDA (NVIDIA)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        config.update({
            'backend': 'cuda',
            'device': 'cuda',
            'vendor': 'nvidia',
            'device_name': props.name,
            'memory_gb': props.total_memory / 1e9,
            'supports_fp16': props.major >= 7,
            'supports_tf32': props.major >= 8,
            'block_size': 1024 if props.major >= 8 else 512,
            'dtype': torch.float16 if props.major >= 7 else torch.float32,
        })
        return config
    
    # 检测MPS (Apple)
    if hasattr(torch, 'mps') and torch.mps.is_available():
        config.update({
            'backend': 'mps',
            'device': 'mps',
            'vendor': 'apple',
            'device_name': 'Apple Silicon',
            'supports_fp16': True,
            'block_size': 256,
            'dtype': torch.float16,
        })
        return config
    
    # 检测XPU (Intel)
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        config.update({
            'backend': 'xpu',
            'device': 'xpu',
            'vendor': 'intel',
            'device_name': 'Intel GPU',
            'block_size': 512,
        })
        return config
    
    # 检测ROCm (AMD)
    if hasattr(torch, 'hip') and torch.hip.is_available():
        config.update({
            'backend': 'rocm',
            'device': 'cuda',  # ROCm使用CUDA API
            'vendor': 'amd',
            'device_name': 'AMD GPU',
            'block_size': 512,
        })
        return config
    
    return config


def get_optimal_backend() -> Backend:
    """
    获取最优后端实例
    
    优先级：
    1. Triton (如果有且CUDA可用)
    2. PyTorch (通用)
    """
    from .pytorch import PyTorchBackend
    
    config = detect_hardware()
    
    # 尝试使用Triton后端（仅CUDA）
    if config['backend'] == 'cuda':
        try:
            from .triton import TritonBackend
            if TritonBackend.is_available():
                return TritonBackend(config)
        except ImportError:
            pass
    
    # 回退到PyTorch后端
    return PyTorchBackend(config)


def get_backend_by_name(name: str, config: Dict = None) -> Backend:
    """
    根据名称获取后端
    
    参数:
        name: 后端名称 ('pytorch', 'triton', 'cuda')
        config: 配置字典（可选）
    """
    if config is None:
        config = detect_hardware()
    
    if name == 'triton':
        from .triton import TritonBackend
        return TritonBackend(config)
    elif name == 'pytorch':
        from .pytorch import PyTorchBackend
        return PyTorchBackend(config)
    elif name == 'cuda':
        config['backend'] = 'cuda'
        from .pytorch import PyTorchBackend
        return PyTorchBackend(config)
    else:
        raise ValueError(f"Unknown backend: {name}")
