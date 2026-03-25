"""
势场优化器 - 基于损失地形曲率的自适应优化

理论基础：频率调制统一理论 F = -E∇Φ
- 曲率大的区域（陡峭）→ 小步长，避免震荡
- 曲率小的区域（平坦）→ 大步长，加速收敛
- 鞍点区域 → 自动逃逸

核心公式：η_adaptive = η₀ / (1 + α·κ)
其中 κ = ||Δg|| / ||Δθ|| 是局部曲率近似
"""

import torch
from collections import OrderedDict
from typing import Optional, Callable, Dict, Any


class PotentialOptimizer(torch.optim.Optimizer):
    """
    势场优化器 - 地形感知的自适应优化器
    
    参数:
        params: 模型参数
        lr: 学习率 (默认: 0.01)
        momentum: 动量因子 (默认: 0.0，不启用)
        curvature_sensitivity: 曲率敏感度 (默认: 1.0)
        saddle_escape: 是否启用鞍点逃逸 (默认: True)
        saddle_threshold: 鞍点检测阈值 (默认: 0.01)
        noise_scale: 鞍点逃逸噪声尺度 (默认: 0.01)
        grad_clip: 梯度裁剪阈值 (默认: None，不裁剪)
        max_history_size: 梯度历史最大大小 (默认: 1000)
    """
    
    def __init__(
        self,
        params,
        lr: float = 0.01,
        momentum: float = 0.0,
        curvature_sensitivity: float = 1.0,
        saddle_escape: bool = True,
        saddle_threshold: float = 0.01,
        noise_scale: float = 0.01,
        grad_clip: Optional[float] = None,
        max_history_size: int = 1000,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            curvature_sensitivity=curvature_sensitivity,
            saddle_escape=saddle_escape,
            saddle_threshold=saddle_threshold,
            noise_scale=noise_scale,
            grad_clip=grad_clip,
            max_history_size=max_history_size,
        )
        super().__init__(params, defaults)
        
        # 使用OrderedDict实现LRU缓存
        self._grad_history: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._momentum_buffer: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._grad_directions: OrderedDict[int, torch.Tensor] = OrderedDict()  # 用于方向检查
        self._ema_curvature: Dict[int, float] = {}  # EMA平滑曲率
        self._step_count = 0
    
    def _add_to_history(self, param_id: int, grad: torch.Tensor):
        """添加梯度到历史，自动限制大小"""
        self._grad_history[param_id] = grad.clone()
        
        # 限制历史记录大小（LRU淘汰）
        max_size = self.param_groups[0].get('max_history_size', 1000)
        if len(self._grad_history) > max_size:
            oldest_key = next(iter(self._grad_history))
            del self._grad_history[oldest_key]
            if oldest_key in self._momentum_buffer:
                del self._momentum_buffer[oldest_key]
            if oldest_key in self._grad_directions:
                del self._grad_directions[oldest_key]
            if oldest_key in self._ema_curvature:
                del self._ema_curvature[oldest_key]
    
    def _compute_smooth_curvature(
        self, 
        param_id: int, 
        grad_norm: float, 
        prev_grad_norm: float, 
        param_norm: float
    ) -> float:
        """计算EMA平滑的曲率"""
        raw_curvature = abs(grad_norm - prev_grad_norm) / (param_norm + 1e-8)
        
        # EMA平滑
        if param_id not in self._ema_curvature:
            self._ema_curvature[param_id] = raw_curvature
        else:
            self._ema_curvature[param_id] = (
                0.9 * self._ema_curvature[param_id] + 0.1 * raw_curvature
            )
        
        return self._ema_curvature[param_id]
    
    def _is_saddle_enhanced(
        self,
        param_id: int,
        grad: torch.Tensor,
        curvature: float,
        grad_norm: float,
        saddle_threshold: float
    ) -> bool:
        """增强的鞍点检测，包含方向变化检查"""
        # 基础检测
        if not (curvature < saddle_threshold and grad_norm < saddle_threshold):
            return False
        
        # 方向变化检测
        if param_id in self._grad_directions:
            prev_grad_dir = self._grad_directions[param_id]
            grad_flat = grad.flatten()
            prev_flat = prev_grad_dir.flatten()
            cos_sim = torch.dot(grad_flat, prev_flat) / (
                grad_flat.norm() * prev_flat.norm() + 1e-8
            )
            cos_sim = cos_sim.item()
            direction_stable = abs(cos_sim) > 0.5
            
            # 方向稳定说明可能不是鞍点（可能在局部极小附近）
            if direction_stable:
                return False
        
        return True
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """
        执行单步优化
        
        参数:
            closure: 重新计算损失的函数（可选）
        
        返回值:
            如果提供了closure，返回loss值；否则返回None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        self._step_count += 1
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            curvature_sensitivity = group['curvature_sensitivity']
            saddle_escape = group['saddle_escape']
            saddle_threshold = group['saddle_threshold']
            noise_scale = group['noise_scale']
            grad_clip = group['grad_clip']
            
            for param in group['params']:
                if param.grad is None:
                    continue
                
                # 保存原始梯度（用于历史和动量）
                original_grad = param.grad.clone()
                param_id = id(param)
                
                # 梯度裁剪（在原始梯度上）
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(param, grad_clip)
                    original_grad = param.grad.clone()
                
                # 计算当前梯度范数
                grad_norm = original_grad.norm().item()
                
                # 计算曲率（使用原始梯度）
                curvature = 0.0
                if param_id in self._grad_history:
                    prev_grad = self._grad_history[param_id]
                    prev_grad_norm = prev_grad.norm().item()
                    param_norm = param.norm().item() + 1e-8
                    curvature = self._compute_smooth_curvature(
                        param_id, grad_norm, prev_grad_norm, param_norm
                    )
                
                # 增强的鞍点检测
                is_saddle = False
                if saddle_escape:
                    is_saddle = self._is_saddle_enhanced(
                        param_id, original_grad, curvature, grad_norm, saddle_threshold
                    )
                
                # 准备用于更新的梯度
                grad_for_update = original_grad.clone()
                
                # 自适应步长
                if is_saddle:
                    adaptive_lr = lr * 2.0
                    # 添加随机扰动帮助逃逸（仅鞍点时）
                    noise = torch.randn_like(grad_for_update) * noise_scale
                    grad_for_update = grad_for_update + noise
                else:
                    adaptive_lr = lr / (1.0 + curvature_sensitivity * curvature)
                
                # 限制步长范围
                adaptive_lr = max(adaptive_lr, lr * 0.01)
                adaptive_lr = min(adaptive_lr, lr * 10.0)
                
                # 动量更新（使用原始梯度）
                if momentum > 0:
                    if param_id not in self._momentum_buffer:
                        self._momentum_buffer[param_id] = torch.zeros_like(original_grad)
                    buf = self._momentum_buffer[param_id]
                    buf.mul_(momentum).add_(original_grad, alpha=1 - momentum)
                    grad_for_update = buf
                
                # 更新参数
                param.add_(grad_for_update, alpha=-adaptive_lr)
                
                # 保存原始梯度历史（不是修改后的）
                self._add_to_history(param_id, original_grad)
                
                # 保存梯度方向用于鞍点检测
                self._grad_directions[param_id] = original_grad.clone()
        
        return loss
    
    def get_curvature_stats(self) -> Dict[str, float]:
        """
        获取曲率统计信息（用于调试）
        
        返回:
            dict: 包含 mean_curvature, max_curvature, min_curvature
        """
        curvatures = list(self._ema_curvature.values())
        
        if curvatures:
            return {
                'mean_curvature': sum(curvatures) / len(curvatures),
                'max_curvature': max(curvatures),
                'min_curvature': min(curvatures),
            }
        return {'mean_curvature': 0.0, 'max_curvature': 0.0, 'min_curvature': 0.0}
    
    def reset_gradient_history(self):
        """重置梯度历史（用于新的训练阶段）"""
        self._grad_history.clear()
        self._momentum_buffer.clear()
        self._grad_directions.clear()
        self._ema_curvature.clear()