"""
势场数学库 
Version: 0.4.0

基于频率调制统一理论，提供地形感知的优化器、损失地形分析、势场叠加等功能。

v0.4.0 新增:
- GPU后端支持（自动检测硬件）
- 势场统计模块（熵、温度、分布）
- 地形增强分析（Morse指数、特征地形）
"""

from .optimizer import PotentialOptimizer
from .landscape import LossLandscape
from .superpose import (
    potential_superpose, 
    stable_potential_superpose,
    potential_superpose_2d,  
    potential_superpose_3d,
    potential_superpose_weighted,
    visualize_potential_superpose,
    set_gpu_backend,           # 新增
    get_gpu_backend,           # 新增
)
from .utils import set_seed, compute_gradient_statistics, compute_param_norm

# 新增：统计模块
from .statistics import PotentialStatistics

# 新增：GPU后端
from .backends import detect_hardware, get_optimal_backend

__all__ = [
    'PotentialOptimizer', 
    'LossLandscape', 
    'potential_superpose',
    'stable_potential_superpose',
    'potential_superpose_2d',  
    'potential_superpose_3d',
    'potential_superpose_weighted',
    'visualize_potential_superpose',
    'set_seed',
    'compute_gradient_statistics',
    'compute_param_norm',
    # 新增
    'PotentialStatistics',
    'detect_hardware',
    'get_optimal_backend',
    'set_gpu_backend',
    'get_gpu_backend',
]

__version__ = '0.4.0'