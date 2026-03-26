"""后端抽象基类"""

from abc import ABC, abstractmethod
import torch
from typing import Optional, Union, List, Callable


class Backend(ABC):
    """势场计算后端抽象基类"""
    
    @abstractmethod
    def potential_superpose(
        self,
        phi_stack: torch.Tensor,
        c: float = 1.0,
        points: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        势场叠加
        
        参数:
            phi_stack: [n_potentials, n_points] 势场值
            c: 耦合常数
            points: 采样点（可选）
        
        返回:
            [n_points] 叠加后的势场
        """
        pass
    
    @abstractmethod
    def compute_gradient_batch(
        self,
        phi: torch.Tensor,
        points: torch.Tensor,
        h: float = 1e-5
    ) -> torch.Tensor:
        """
        批量梯度计算
        
        参数:
            phi: [n_points] 势场值
            points: [n_points, dim] 采样点
            h: 差分步长
        
        返回:
            [n_points, dim] 梯度
        """
        pass
    
    @abstractmethod
    def compute_curvature_batch(
        self,
        grads: torch.Tensor,
        params: torch.Tensor
    ) -> torch.Tensor:
        """
        批量曲率计算
        
        参数:
            grads: [n_params] 梯度
            params: [n_params] 参数
        
        返回:
            float 曲率值
        """
        pass
    
    @staticmethod
    def is_available() -> bool:
        """检查后端是否可用"""
        return False
    
    @property
    def name(self) -> str:
        """后端名称"""
        return self.__class__.__name__