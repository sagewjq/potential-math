"""
势场统计模块 - 第五篇论文实现

基于势场数学第五篇：熵、温度、分布、宏观量的势场起源
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings

# 物理常数（可配置）
K_B = 1.380649e-23  # 玻尔兹曼常数
H_P = 6.62607015e-34  # 普朗克常数
C = 299792458.0  # 光速


class PotentialStatistics:
    """
    势场统计 - 基于第五篇论文
    
    核心公式：
    - 熵：S = -k⟨Φ⟩
    - 温度：kT = ⟨hν⟩
    - 玻尔兹曼分布：ρ(r) ∝ e^{Φ_total}
    - 力的统计本质：⟨F⟩ = -⟨E∇Φ⟩
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        c: float = C,
        k: float = K_B,
        h: float = H_P
    ):
        """
        参数:
            model: PyTorch模型
            optimizer: 优化器（用于获取参数历史）
            c: 光速
            k: 玻尔兹曼常数
            h: 普朗克常数
        """
        self.model = model
        self.optimizer = optimizer
        self.c = c
        self.k = k
        self.h = h
        
        self._energy_history: List[float] = []
        self._potential_history: List[float] = []
        self._gradient_history: List[torch.Tensor] = []
    
    def record_state(self, loss: float, gradients: Optional[List[torch.Tensor]] = None):
        """记录当前状态"""
        self._energy_history.append(loss)
        
        # 计算势场 Φ = ln(E / E₀)
        # 其中 E₀ = hν₀，这里用第一个能量作为参考
        if len(self._energy_history) == 1:
            self._potential_history.append(0.0)
        else:
            e0 = self._energy_history[0]
            potential = np.log(loss / e0) if loss > 0 else 0.0
            self._potential_history.append(potential)
        
        if gradients is not None:
            grad_norm = sum(g.norm().item() ** 2 for g in gradients if g is not None) ** 0.5
            self._gradient_history.append(grad_norm)
    
    def compute_entropy(self) -> float:
        """
        熵的势场定义
        
        论文公式：S = -k⟨Φ⟩
        
        物理意义：熵是平均势场的负值，势场越均匀，熵越大
        """
        if not self._potential_history:
            return 0.0
        
        mean_potential = np.mean(self._potential_history)
        return -self.k * mean_potential
    
    def compute_temperature(self) -> float:
        """
        温度的频率本质
        
        论文公式：kT = ⟨hν⟩ = ⟨E⟩
        
        物理意义：温度是平均能量的度量，能量越高，温度越高
        """
        if not self._energy_history:
            return 0.0
        
        mean_energy = np.mean(self._energy_history)
        return mean_energy / self.k
    
    def compute_free_energy(self) -> float:
        """
        自由能
        
        公式：F = ⟨E⟩ - TS
        """
        mean_energy = np.mean(self._energy_history) if self._energy_history else 0.0
        entropy = self.compute_entropy()
        temperature = self.compute_temperature()
        
        return mean_energy - temperature * entropy
    
    def boltzmann_distribution(
        self,
        potential: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        玻尔兹曼分布的势场起源
        
        论文公式：ρ(r) ∝ e^{Φ_total} ∝ e^{-U/(kT)}
        
        参数:
            potential: 势场值 Φ
            temperature: 温度（None则使用计算的温度）
        
        返回:
            概率分布
        """
        if temperature is None:
            temperature = self.compute_temperature()
        
        if temperature <= 0:
            return torch.ones_like(potential) / potential.numel()
        
        beta = 1.0 / (self.k * temperature)
        # Φ_total = -βU（在非相对论近似下）
        weights = torch.exp(-beta * potential)
        
        return weights / weights.sum()
    
    def compute_statistical_force(
        self,
        grad_phi: torch.Tensor,
        energies: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        力的统计本质
        
        论文公式：⟨F⟩ = -⟨E∇Φ⟩
        
        参数:
            grad_phi: 势场梯度 [n_samples, dim]
            energies: 能量（None则从历史计算）
        
        返回:
            统计力 [dim]
        """
        if energies is None and self._energy_history:
            energies = torch.tensor(self._energy_history, device=grad_phi.device)
        
        if energies is None:
            # 使用平均能量
            mean_energy = np.mean(self._energy_history) if self._energy_history else 1.0
            energies = torch.full((grad_phi.shape[0],), mean_energy, device=grad_phi.device)
        
        # F = -E∇Φ
        forces = -energies.unsqueeze(-1) * grad_phi
        
        return forces.mean(dim=0)
    
    def compute_statistical_pressure(
        self,
        forces: torch.Tensor,
        positions: torch.Tensor,
        volume: float
    ) -> float:
        """
        压力的统计计算
        
        公式：P = (1/3V) ⟨∑ r·F⟩
        
        参数:
            forces: 力 [n_particles, dim]
            positions: 位置 [n_particles, dim]
            volume: 体积
        
        返回:
            压力
        """
        virial = (positions * forces).sum().item()
        return virial / (3.0 * volume)
    
    def compute_angular_momentum(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        energies: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        角动量的统计投影
        
        论文公式：⟨h⟩ = √(GM⟨r⟩)（特定条件下）
        
        参数:
            positions: 位置 [n_particles, dim]
            velocities: 速度 [n_particles, dim]
            energies: 能量（None则从历史计算）
        
        返回:
            平均角动量 [dim]
        """
        if energies is None and self._energy_history:
            energies = torch.tensor(self._energy_history[-1], device=positions.device)
        elif energies is None:
            energies = torch.ones(positions.shape[0], device=positions.device)
        
        # p = (E/c²) v
        momentum = (energies.unsqueeze(-1) / (self.c ** 2)) * velocities
        
        # L = r × p
        angular_momenta = torch.cross(positions, momentum)
        
        return angular_momenta.mean(dim=0)
    
    def compute_correlation_function(
        self,
        values: torch.Tensor,
        max_lag: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算关联函数
        
        参数:
            values: 时间序列 [n_steps]
            max_lag: 最大延迟
        
        返回:
            (lags, correlations)
        """
        n = len(values)
        mean = values.mean()
        var = values.var()
        
        correlations = []
        for lag in range(min(max_lag, n // 2)):
            if lag == 0:
                corr = 1.0
            else:
                cross = (values[lag:] - mean) * (values[:-lag] - mean)
                corr = cross.mean() / var
            correlations.append(corr)
        
        return np.arange(len(correlations)), np.array(correlations)
    
    def get_statistics_summary(self) -> Dict[str, float]:
        """获取统计摘要"""
        return {
            'entropy': self.compute_entropy(),
            'temperature': self.compute_temperature(),
            'free_energy': self.compute_free_energy(),
            'mean_energy': np.mean(self._energy_history) if self._energy_history else 0.0,
            'energy_std': np.std(self._energy_history) if self._energy_history else 0.0,
            'mean_potential': np.mean(self._potential_history) if self._potential_history else 0.0,
            'potential_std': np.std(self._potential_history) if self._potential_history else 0.0,
            'gradient_norm_mean': np.mean(self._gradient_history) if self._gradient_history else 0.0,
        }
    
    def reset(self):
        """重置历史"""
        self._energy_history.clear()
        self._potential_history.clear()
        self._gradient_history.clear()
    
    def get_equipartition_ratio(self) -> float:
        """
        能量均分比
        
        检查 ⟨Φ_degree⟩ / (½⟨Φ_th⟩) 是否接近 1
        """
        if not self._energy_history:
            return 0.0
        
        # 简化实现
        mean_energy = np.mean(self._energy_history)
        temperature = self.compute_temperature()
        
        # 每个自由度的平均能量应为 ½kT
        expected_per_degree = 0.5 * self.k * temperature
        
        if expected_per_degree == 0:
            return 0.0
        
        return mean_energy / expected_per_degree

