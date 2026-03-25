"""
势场数学库 
Version: 0.2.0

基于频率调制统一理论，提供地形感知的优化器、损失地形分析、势场叠加等功能。
"""

from .optimizer import PotentialOptimizer
from .landscape import LossLandscape
from .superpose import potential_superpose, stable_potential_superpose
from .utils import set_seed, compute_gradient_statistics, compute_param_norm

__all__ = [
    'PotentialOptimizer', 
    'LossLandscape', 
    'potential_superpose',
    'stable_potential_superpose',
    'set_seed',
    'compute_gradient_statistics',
    'compute_param_norm',
]
__version__ = '0.2.0'