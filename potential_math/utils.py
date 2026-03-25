"""
辅助工具函数

提供随机种子设置、梯度统计、参数范数计算等实用功能。
"""

import torch
import numpy as np
from typing import Dict, Optional


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_gradient_statistics(model: torch.nn.Module) -> Dict[str, float]:
    """计算梯度统计信息"""
    grad_norms = []
    grad_max = 0.0
    zero_grad_count = 0
    total_params = 0
    
    for param in model.parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            grad_max = max(grad_max, param.grad.abs().max().item())
            
            zero_grad_count += (param.grad.abs() < 1e-8).sum().item()
            total_params += param.numel()
    
    if grad_norms:
        mean_norm = np.mean(grad_norms)
        std_norm = np.std(grad_norms)
        sparsity = zero_grad_count / total_params if total_params > 0 else 0.0
    else:
        mean_norm = 0.0
        std_norm = 0.0
        sparsity = 0.0
    
    return {
        'mean_grad_norm': mean_norm,
        'std_grad_norm': std_norm,
        'max_grad': grad_max,
        'grad_sparsity': sparsity,
    }


def compute_param_norm(model: torch.nn.Module) -> float:
    """计算模型参数范数"""
    total_norm_sq = 0.0
    for param in model.parameters():
        total_norm_sq += param.norm().item() ** 2
    return total_norm_sq ** 0.5


def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """获取可用设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_device(data, device: torch.device, _depth: int = 0) -> any:
    """
    将数据移动到指定设备（修复递归深度版）
    
    参数:
        data: 数据（tensor、list、dict或tuple）
        device: 目标设备
        _depth: 递归深度（内部使用）
    
    返回:
        移动后的数据
    """
    MAX_DEPTH = 100
    if _depth > MAX_DEPTH:
        raise RecursionError(f"数据嵌套深度超过{MAX_DEPTH}")
    
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: to_device(v, device, _depth + 1) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(x, device, _depth + 1) for x in data)
    return data