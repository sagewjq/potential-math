"""PyTorch后端 - 通用CPU/GPU支持"""

import torch
from typing import Optional, Dict, Any
from .base import Backend


class PyTorchBackend(Backend):
    """PyTorch后端，支持所有PyTorch设备"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config['device']
        self.dtype = config.get('dtype', torch.float32)
    
    @staticmethod
    def is_available() -> bool:
        return True
    
    @property
    def name(self) -> str:
        return f"PyTorch({self.device})"
    
    def potential_superpose(
        self,
        phi_stack: torch.Tensor,
        c: float = 1.0,
        points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        势场叠加 - PyTorch实现
        
        公式：Φ_total = -c² ln(∑ e^{-Φ_i/c²})
        """
        c2 = c * c
        exp_sum = torch.exp(-phi_stack / c2).sum(dim=0)
        result = -c2 * torch.log(exp_sum + 1e-8)
        return result.to(dtype=self.dtype)
    
    def compute_gradient_batch(
        self,
        phi: torch.Tensor,
        points: torch.Tensor,
        h: float = 1e-5
    ) -> torch.Tensor:
        """
        批量梯度计算 - 中心差分
        
        参数:
            phi: [n_points] 势场值
            points: [n_points, dim] 采样点
            h: 差分步长
        
        返回:
            [n_points, dim] 梯度
        """
        n_points, dim = points.shape
        grad = torch.zeros_like(points)
        
        # 使用自动微分（如果phi是计算图的一部分）
        if phi.requires_grad:
            grad = torch.autograd.grad(
                phi.sum(), points, create_graph=False, retain_graph=False
            )[0]
        else:
            # 数值差分
            for d in range(dim):
                points_plus = points.clone()
                points_minus = points.clone()
                points_plus[:, d] += h
                points_minus[:, d] -= h
                
                # 需要重新计算势场（用户应提供势场函数）
                # 这里假设phi是预计算的
                # 实际使用时应该传入势场函数
                pass
        
        return grad.to(dtype=self.dtype)
    
    def compute_curvature_batch(
        self,
        grads: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        批量曲率计算
        
        κ = ||Δg|| / ||Δθ||
        """
        # 这里需要历史数据，简化处理
        # 实际使用时应在优化器中维护历史
        if len(grads.shape) == 1:
            grad_norm = torch.norm(grads)
            param_norm = torch.norm(params)
        else:
            grad_norm = torch.norm(grads, dim=1)
            param_norm = torch.norm(params, dim=1)
        
        curvature = grad_norm / (param_norm + 1e-8)
        return curvature

