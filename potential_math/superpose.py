"""
势场叠加公式：Φ_total = -c² ln(∑ e^{-Φ_i/c²})
"""

import torch
import numpy as np
from typing import List, Union, Callable, Optional


def potential_superpose(
    potentials: List[Union[torch.Tensor, float, Callable]],
    c: float = 1.0,
    points: Optional[torch.Tensor] = None,
    return_tensor: bool = True
) -> Union[torch.Tensor, float, Callable]:
    """
    势场叠加函数
    """
    if return_tensor:
        if points is None:
            raise ValueError("当return_tensor=True时，必须提供points")
        
        device = points.device if hasattr(points, 'device') else torch.device('cpu')
        exp_sum = torch.zeros(points.shape[0], device=device)
        
        for pot in potentials:
            if callable(pot):
                phi_vals = pot(points)
            elif isinstance(pot, torch.Tensor):
                phi_vals = pot
            else:
                phi_vals = torch.full((points.shape[0],), pot, device=device)
            
            exp_sum += torch.exp(-phi_vals / c**2)
        
        return -c**2 * torch.log(exp_sum + 1e-8)
    
    else:
        def superposed_function(x):
            exp_sum = 0.0
            for pot in potentials:
                if callable(pot):
                    phi_val = pot(x)
                elif isinstance(pot, (int, float)):
                    phi_val = pot
                else:
                    phi_val = pot
                exp_sum += np.exp(-phi_val / c**2)
            return -c**2 * np.log(exp_sum + 1e-8)
        
        return superposed_function


def stable_potential_superpose(
    potentials: List[Union[torch.Tensor, float]],
    c: float = 1.0,
    axis: int = 0
) -> torch.Tensor:
    """
    数值稳定的势场叠加版本（修复形状广播版）
    """
    # 转换为Tensor
    phi_tensors = []
    for p in potentials:
        if isinstance(p, (int, float)):
            p = torch.tensor([p]) if axis == 0 else torch.tensor([[p]])
        elif not isinstance(p, torch.Tensor):
            p = torch.tensor(p)
        phi_tensors.append(p)
    
    # 检查并处理形状不一致
    shapes = [t.shape for t in phi_tensors]
    if len(set(shapes)) > 1:
        target_shape = torch.broadcast_shapes(*shapes)
        phi_tensors_fixed = []
        for t in phi_tensors:
            if t.shape == target_shape:
                phi_tensors_fixed.append(t)
            else:
                # 使用expand并确保连续内存
                phi_tensors_fixed.append(t.expand(target_shape).contiguous())
        phi_tensors = phi_tensors_fixed
    
    # 堆叠并找到最大值
    stacked = torch.stack(phi_tensors, dim=axis)
    max_val = stacked.max(dim=axis, keepdim=True)[0]
    
    # 稳定计算
    exp_sum = torch.exp(-(stacked - max_val) / c**2).sum(dim=axis)
    result = -c**2 * (torch.log(exp_sum + 1e-8) - max_val.squeeze() / c**2)
    
    return result


def potential_superpose_2d(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """二维势场叠加"""
    return stable_potential_superpose([phi1, phi2], c=c, axis=0)


def potential_superpose_3d(
    phi1: torch.Tensor,
    phi2: torch.Tensor,
    phi3: torch.Tensor,
    c: float = 1.0
) -> torch.Tensor:
    """三维势场叠加"""
    return stable_potential_superpose([phi1, phi2, phi3], c=c, axis=0)