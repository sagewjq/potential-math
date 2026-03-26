"""
势场叠加公式（v0.4.0）

修复内容：
1. 修复 torch.broadcast_tensors 返回值未捕获的问题
2. 增强数值稳定性
3. 添加更好的错误处理
4. 新增GPU后端支持（v0.4.0）
5. 新增批量势场叠加优化（v0.4.0）

理论基础：
    Φ_total = -c² ln(∑ e^{-Φ_i/c²})
"""

import torch
import numpy as np
from typing import List, Union, Callable, Optional, Tuple
import warnings

# 导入GPU后端（可选）
_BACKEND = None
_USE_GPU = True


def set_gpu_backend(use_gpu: bool = True, backend_name: str = 'auto'):
    """
    设置GPU后端
    
    参数:
        use_gpu: 是否使用GPU加速（默认True）
        backend_name: 后端名称 ('auto', 'pytorch', 'triton')
    
    示例:
        >>> from potential_math import set_gpu_backend
        >>> set_gpu_backend(True)  # 启用GPU加速
        >>> set_gpu_backend(False)  # 禁用GPU，使用CPU
    """
    global _BACKEND, _USE_GPU
    _USE_GPU = use_gpu
    _BACKEND = None  # 重置后端
    
    if use_gpu and backend_name != 'auto':
        try:
            from .backends import get_backend_by_name, detect_hardware
            config = detect_hardware()
            _BACKEND = get_backend_by_name(backend_name, config)
        except (ImportError, AttributeError) as e:
            warnings.warn(f"Failed to load backend '{backend_name}': {e}")


def get_gpu_backend():
    """获取当前GPU后端实例"""
    global _BACKEND, _USE_GPU
    
    if not _USE_GPU:
        return None
    
    if _BACKEND is None:
        try:
            from .backends import get_optimal_backend
            _BACKEND = get_optimal_backend()
        except (ImportError, AttributeError):
            pass
    
    return _BACKEND


def potential_superpose(
    potentials: List[Union[torch.Tensor, float, Callable]],
    c: float = 1.0,
    points: Optional[torch.Tensor] = None,
    return_tensor: bool = True,
    use_gpu: bool = True,
) -> Union[torch.Tensor, float, Callable]:
    """
    势场叠加函数
    
    基于频率调制统一理论：Φ_total = -c² ln(∑ e^{-Φ_i/c²})
    
    参数:
        potentials: 势场列表，可以是Tensor、常数或函数
        c: 耦合常数（默认: 1.0）
        points: 采样点（当return_tensor=True时必须提供）
        return_tensor: 是否返回Tensor（否则返回函数）
        use_gpu: 是否尝试使用GPU加速（默认: True）
    
    返回:
        叠加后的势场值或函数
    
    示例:
        >>> def gaussian(x): return torch.exp(-x**2)
        >>> x = torch.linspace(-3, 3, 100)
        >>> phi = potential_superpose([gaussian, 0.5], points=x)
        
        >>> # GPU加速版本
        >>> phi_gpu = potential_superpose([gaussian, 0.5], points=x.cuda(), use_gpu=True)
    """
    if return_tensor:
        if points is None:
            raise ValueError("When return_tensor=True, points must be provided")
        
        # 获取设备信息
        device = points.device if hasattr(points, 'device') else torch.device('cpu')
        
        # 收集所有势场值
        phi_vals_list = []
        for i, pot in enumerate(potentials):
            if callable(pot):
                try:
                    phi_vals = pot(points)
                except Exception as e:
                    raise ValueError(f"Failed to evaluate potential {i}: {e}")
            elif isinstance(pot, torch.Tensor):
                phi_vals = pot.to(device)
            else:
                # 常数势场
                phi_vals = torch.full((points.shape[0],), pot, dtype=torch.float32, device=device)
            
            # 确保形状一致
            if phi_vals.shape != points.shape[:1]:
                raise ValueError(
                    f"Potential {i} output shape {phi_vals.shape} "
                    f"does not match points shape {points.shape[:1]}"
                )
            
            phi_vals_list.append(phi_vals)
        
        # 尝试GPU加速
        if use_gpu and len(phi_vals_list) > 1:
            try:
                backend = get_gpu_backend()
                if backend is not None:
                    phi_stack = torch.stack(phi_vals_list, dim=0)
                    return backend.potential_superpose(phi_stack, c, points)
            except Exception as e:
                warnings.warn(f"GPU acceleration failed, falling back to CPU: {e}")
        
        # CPU实现（数值稳定版本）
        c2 = c * c
        exp_sum = torch.zeros(points.shape[0], device=device, dtype=torch.float32)
        
        for phi_vals in phi_vals_list:
            # 数值稳定：减去最大值防止溢出
            phi_max = phi_vals.max()
            exp_sum += torch.exp(-(phi_vals - phi_max) / c2)
            # 恢复最大值
            exp_sum = exp_sum * torch.exp(-phi_max / c2)
        
        # 避免log(0)
        exp_sum = torch.clamp(exp_sum, min=1e-8)
        result = -c2 * torch.log(exp_sum)
        
        return result
    
    else:
        # 返回函数版本（适用于非Tensor输入）
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
            
            # 避免log(0)
            exp_sum = max(exp_sum, 1e-8)
            return -c**2 * np.log(exp_sum)
        
        return superposed_function


def stable_potential_superpose(
    potentials: List[Union[torch.Tensor, float]],
    c: float = 1.0,
    axis: int = 0,
    use_gpu: bool = True,
) -> torch.Tensor:
    """
    数值稳定的势场叠加版本
    
    使用log-sum-exp技巧避免数值溢出
    
    参数:
        potentials: 势场Tensor列表
        c: 耦合常数
        axis: 堆叠维度
        use_gpu: 是否使用GPU加速（默认: True）
    
    返回:
        torch.Tensor: 叠加后的势场
    
    示例:
        >>> phi1 = torch.tensor([1.0, 2.0, 3.0])
        >>> phi2 = torch.tensor([0.5, 1.0, 1.5])
        >>> phi_total = stable_potential_superpose([phi1, phi2])
    """
    if not potentials:
        raise ValueError("potentials list cannot be empty")
    
    # 转换为Tensor
    phi_tensors = []
    for i, p in enumerate(potentials):
        if isinstance(p, (int, float)):
            p = torch.tensor([p], dtype=torch.float32)
        elif not isinstance(p, torch.Tensor):
            try:
                p = torch.tensor(p, dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"Potential {i} cannot be converted to tensor: {e}")
        phi_tensors.append(p)
    
    # 广播所有张量到相同形状
    try:
        broadcasted_tensors = torch.broadcast_tensors(*phi_tensors)
    except RuntimeError as e:
        shapes = [t.shape for t in phi_tensors]
        raise ValueError(
            f"Cannot broadcast potentials with shapes {shapes}. "
            f"Make sure all tensors are broadcastable. Error: {e}"
        )
    
    # 堆叠
    stacked = torch.stack(broadcasted_tensors, dim=axis)
    
    # 尝试GPU加速
    if use_gpu:
        try:
            backend = get_gpu_backend()
            if backend is not None:
                # 重塑为 [n_potentials, -1] 以适配后端接口
                original_shape = stacked.shape
                n_potentials = original_shape[axis]
                n_points = stacked.numel() // n_potentials
                
                # 展平
                flat_stacked = stacked.reshape(n_potentials, n_points)
                flat_result = backend.potential_superpose(flat_stacked, c, None)
                
                # 恢复形状
                result_shape = list(original_shape)
                result_shape.pop(axis)
                result = flat_result.reshape(result_shape)
                return result
        except Exception as e:
            warnings.warn(f"GPU acceleration failed, falling back to CPU: {e}")
    
    # CPU实现（数值稳定的log-sum-exp）
    max_val = stacked.max(dim=axis, keepdim=True)[0]
    scaled = -(stacked - max_val) / (c**2 + 1e-8)
    exp_sum = torch.exp(scaled).sum(dim=axis)
    result = -c**2 * torch.log(exp_sum + 1e-8) + max_val.squeeze(axis)
    
    return result


def potential_superpose_2d(
    phi1: Union[torch.Tensor, float],
    phi2: Union[torch.Tensor, float],
    c: float = 1.0,
    use_gpu: bool = True,
) -> torch.Tensor:
    """
    二维势场叠加便捷函数
    
    参数:
        phi1: 第一个势场
        phi2: 第二个势场
        c: 耦合常数
        use_gpu: 是否使用GPU加速
    
    返回:
        torch.Tensor: 叠加后的势场
    """
    return stable_potential_superpose([phi1, phi2], c=c, axis=0, use_gpu=use_gpu)


def potential_superpose_3d(
    phi1: Union[torch.Tensor, float],
    phi2: Union[torch.Tensor, float],
    phi3: Union[torch.Tensor, float],
    c: float = 1.0,
    use_gpu: bool = True,
) -> torch.Tensor:
    """
    三维势场叠加便捷函数
    
    参数:
        phi1: 第一个势场
        phi2: 第二个势场
        phi3: 第三个势场
        c: 耦合常数
        use_gpu: 是否使用GPU加速
    
    返回:
        torch.Tensor: 叠加后的势场
    """
    return stable_potential_superpose([phi1, phi2, phi3], c=c, axis=0, use_gpu=use_gpu)


def potential_superpose_weighted(
    potentials: List[Union[torch.Tensor, float]],
    weights: List[float],
    c: float = 1.0,
    axis: int = 0,
    use_gpu: bool = True,
) -> torch.Tensor:
    """
    带权重的势场叠加
    
    公式：Φ_total = -c² ln(∑ w_i e^{-Φ_i/c²})
    
    参数:
        potentials: 势场Tensor列表
        weights: 权重列表（必须与potentials长度相同）
        c: 耦合常数
        axis: 堆叠维度
        use_gpu: 是否使用GPU加速
    
    返回:
        torch.Tensor: 叠加后的势场
    """
    if len(potentials) != len(weights):
        raise ValueError(
            f"Length mismatch: {len(potentials)} potentials vs {len(weights)} weights"
        )
    
    # 标准化权重
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    weights_tensor = weights_tensor / weights_tensor.sum()
    
    # 转换势场为Tensor
    phi_tensors = []
    for p in potentials:
        if isinstance(p, (int, float)):
            p = torch.tensor([p], dtype=torch.float32)
        elif not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        phi_tensors.append(p)
    
    # 广播
    broadcasted_tensors = torch.broadcast_tensors(*phi_tensors)
    
    # 堆叠
    stacked = torch.stack(broadcasted_tensors, dim=axis)
    max_val = stacked.max(dim=axis, keepdim=True)[0]
    
    # 将权重移到相同设备
    device = stacked.device
    weights_tensor = weights_tensor.to(device)
    
    # 添加维度以匹配广播
    for _ in range(stacked.dim() - 1):
        weights_tensor = weights_tensor.unsqueeze(-1)
    
    # 尝试GPU加速
    if use_gpu:
        try:
            backend = get_gpu_backend()
            if backend is not None:
                # 使用后端实现（如果支持加权）
                n_potentials, n_points = stacked.shape[0], stacked.numel() // stacked.shape[0]
                flat_stacked = stacked.reshape(n_potentials, n_points)
                
                # 应用权重
                weighted = flat_stacked + torch.log(weights_tensor.reshape(-1, 1))
                
                # 使用log-sum-exp
                max_vals = weighted.max(dim=0)[0]
                exp_sum = torch.exp(weighted - max_vals).sum(dim=0)
                flat_result = -c**2 * torch.log(exp_sum + 1e-8) + max_vals
                
                result_shape = list(stacked.shape)
                result_shape.pop(axis)
                return flat_result.reshape(result_shape)
        except Exception as e:
            warnings.warn(f"GPU acceleration failed in weighted superpose: {e}")
    
    # CPU实现
    scaled = -(stacked - max_val) / (c**2 + 1e-8)
    exp_sum = (weights_tensor * torch.exp(scaled)).sum(dim=axis)
    result = -c**2 * torch.log(exp_sum + 1e-8) + max_val.squeeze(axis)
    
    return result


def visualize_potential_superpose(
    potentials: List[Callable],
    x_range: Tuple[float, float] = (-3, 3),
    n_points: int = 100,
    c: float = 1.0,
    save_path: Optional[str] = None,
    use_gpu: bool = False,  # 可视化通常不需要GPU
):
    """
    可视化势场叠加效果（需要matplotlib）
    
    参数:
        potentials: 势场函数列表
        x_range: x轴范围
        n_points: 采样点数
        c: 耦合常数
        save_path: 保存路径（可选）
        use_gpu: 是否使用GPU（默认False，可视化不需要）
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization")
    
    # 生成采样点
    x = torch.linspace(x_range[0], x_range[1], n_points)
    
    # 计算各个势场
    phi_values = []
    for pot in potentials:
        phi = pot(x)
        if isinstance(phi, torch.Tensor):
            phi = phi.cpu().numpy()
        phi_values.append(phi)
    
    # 计算叠加后的势场
    phi_total = potential_superpose(potentials, c=c, points=x, use_gpu=use_gpu)
    if isinstance(phi_total, torch.Tensor):
        phi_total = phi_total.cpu().numpy()
    x_np = x.cpu().numpy()
    
    # 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制各个势场（虚线）
    for i, phi in enumerate(phi_values):
        plt.plot(x_np, phi, '--', label=f'Φ_{i+1}', alpha=0.7)
    
    # 绘制总势场（实线）
    plt.plot(x_np, phi_total, '-', linewidth=2, label='Φ_total', color='black')
    
    plt.xlabel('x')
    plt.ylabel('Φ(x)')
    plt.title(f'势场叠加 (c = {c})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def batch_potential_superpose(
    phi_stack: torch.Tensor,
    c: float = 1.0,
    use_gpu: bool = True,
) -> torch.Tensor:
    """
    批量势场叠加（高性能版本）
    
    参数:
        phi_stack: [batch_size, n_potentials, n_points] 或 [n_potentials, n_points]
        c: 耦合常数
        use_gpu: 是否使用GPU加速
    
    返回:
        torch.Tensor: [batch_size, n_points] 叠加后的势场
    
    示例:
        >>> # 批量处理多个样本
        >>> phi_stack = torch.randn(32, 10, 1000)  # 32个样本，10个势场，1000个点
        >>> result = batch_potential_superpose(phi_stack, c=1.0)
    """
    if phi_stack.dim() not in [2, 3]:
        raise ValueError(f"Expected 2D or 3D input, got {phi_stack.dim()}D")
    
    original_shape = phi_stack.shape
    
    # 确保形状为 [n_potentials, n_points] 或 [batch, n_potentials, n_points]
    if phi_stack.dim() == 2:
        n_potentials, n_points = phi_stack.shape
        batch_size = 1
        phi_reshaped = phi_stack.unsqueeze(0)  # [1, n_potentials, n_points]
    else:
        batch_size, n_potentials, n_points = phi_stack.shape
        phi_reshaped = phi_stack
    
    # 尝试GPU加速
    if use_gpu:
        try:
            backend = get_gpu_backend()
            if backend is not None:
                results = []
                for b in range(batch_size):
                    result = backend.potential_superpose(phi_reshaped[b], c, None)
                    results.append(result)
                return torch.stack(results) if batch_size > 1 else results[0]
        except Exception as e:
            warnings.warn(f"GPU acceleration failed in batch superpose: {e}")
    
    # CPU实现（向量化）
    c2 = c * c
    
    # 数值稳定的log-sum-exp
    max_vals = phi_reshaped.max(dim=1, keepdim=True)[0]  # [batch, 1, n_points]
    scaled = -(phi_reshaped - max_vals) / c2
    exp_sum = torch.exp(scaled).sum(dim=1)  # [batch, n_points]
    result = -c2 * torch.log(exp_sum + 1e-8) + max_vals.squeeze(1)
    
    # 恢复形状
    if original_shape == phi_stack.shape and phi_stack.dim() == 2:
        result = result.squeeze(0)
    
    return result


# 导出公共接口
__all__ = [
    'potential_superpose',
    'stable_potential_superpose',
    'potential_superpose_2d',
    'potential_superpose_3d',
    'potential_superpose_weighted',
    'batch_potential_superpose',
    'visualize_potential_superpose',
    'set_gpu_backend',
    'get_gpu_backend',
]

