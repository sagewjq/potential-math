"""
损失地形分析工具（修复版 v0.3.2）

修复内容：
1. 修复梯度累积污染问题 - 在分析前保存并恢复原始梯度
2. 添加梯度上下文管理器
3. 优化内存使用
4. 增强Hessian计算的数值稳定性
"""

import torch
import warnings
from typing import Optional, List, Dict, Any, Callable, Union
from contextlib import contextmanager
import numpy as np


class LossLandscape:
    """
    损失地形分析器（修复版 v0.3.2）
    
    修复内容：
    1. 修复梯度累积污染问题 - 分析前保存并恢复原始梯度
    2. 添加梯度隔离上下文管理器
    3. 修复Hessian计算的内存泄漏
    4. 优化梯度计算性能
    
    注意：
        - 所有分析方法都会保存并恢复原始梯度状态
        - 可以在训练循环中安全使用，不会污染训练梯度
        - Hessian相关计算非常消耗内存，仅适用于小模型
    
    参数:
        model: PyTorch模型
        loss_fn: 损失函数（默认: MSELoss）
    """
    
    def __init__(self, model: torch.nn.Module, loss_fn: Optional[Callable] = None):
        self.model = model
        self.loss_fn = loss_fn or torch.nn.MSELoss()
        self._device = next(model.parameters()).device
    
    @contextmanager
    def _preserve_gradients(self):
        """
        梯度保存和恢复的上下文管理器
        
        用法：
            with self._preserve_gradients():
                # 执行会修改梯度的操作
                pass
            # 退出后梯度自动恢复
        """
        # 保存原始梯度
        original_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()
        
        try:
            yield
        finally:
            # 清除当前梯度
            self.model.zero_grad()
            # 恢复原始梯度
            for name, param in self.model.named_parameters():
                if name in original_grads:
                    param.grad = original_grads[name]
    
    def _safe_compute_gradients(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        安全计算梯度（不影响原始梯度状态）
        
        返回:
            tuple: (loss, gradients)
        """
        with self._preserve_gradients():
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            # 收集梯度
            gradients = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    gradients[name] = param.grad.clone()
            
            return loss.item(), gradients
    
    def compute_curvature(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        param_filter: Optional[List[torch.nn.Parameter]] = None
    ) -> float:
        """
        计算损失函数在当前参数处的平均曲率
        
        曲率定义：κ = ||∇L|| / ||θ||
        用于快速判断地形陡峭程度
        
        ✅ 修复：不会污染模型的原始梯度
        
        参数:
            inputs: 输入数据
            targets: 目标值
            param_filter: 指定要分析的参数（可选）
        
        返回:
            float: 平均曲率值
        """
        params = param_filter or list(self.model.parameters())
        
        # ✅ 使用梯度隔离上下文
        with self._preserve_gradients():
            # 计算当前梯度
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            grad_norm_sq = 0.0
            param_norm_sq = 0.0
            
            for param in params:
                if param.grad is not None:
                    grad_norm_sq += param.grad.norm().item() ** 2
                    param_norm_sq += param.norm().item() ** 2
            
            curvature = (grad_norm_sq ** 0.5) / (param_norm_sq ** 0.5 + 1e-8)
            
            # 上下文退出时会自动清理梯度并恢复原始状态
            
        return curvature
    
    def compute_hessian_diag(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        param_filter: Optional[List[torch.nn.Parameter]] = None,
        warn_large_model: bool = True
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """
        计算Hessian矩阵的对角线（修复梯度污染版）
        
        ✅ 修复：不会污染模型的原始梯度
        ✅ 修复：内存泄漏问题
        
        警告：
            - 此方法非常消耗内存（O(n_params)）
            - 对于大模型（>1M参数），可能导致OOM
            - 建议仅用于调试和小模型（<100K参数）
        
        参数:
            inputs: 输入数据
            targets: 目标值
            param_filter: 指定要分析的参数（可选）
            warn_large_model: 是否显示大模型警告
        
        返回:
            dict: 参数到Hessian对角线的映射
        """
        params = param_filter or list(self.model.parameters())
        
        # 检查模型大小并发出警告
        total_params = sum(p.numel() for p in params)
        if warn_large_model and total_params > 100000:
            warnings.warn(
                f"Model has {total_params:,} parameters. "
                f"Hessian diagonal computation may cause OOM. "
                f"Consider using compute_curvature() instead.",
                UserWarning
            )
        
        hessian_diag = {}
        
        # ✅ 使用梯度隔离上下文
        with self._preserve_gradients():
            for param in params:
                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                
                # 计算一阶梯度
                grad = torch.autograd.grad(
                    loss, param, 
                    create_graph=True,  # 需要计算图用于二阶导数
                    retain_graph=False
                )[0]
                
                if grad is not None:
                    # 计算二阶导数（Hessian对角线）
                    try:
                        # 对每个元素分别计算二阶导数
                        hessian = torch.autograd.grad(
                            grad.sum(), param,
                            retain_graph=False,
                            create_graph=False
                        )[0]
                        hessian_diag[param] = hessian.detach().cpu()
                    except RuntimeError as e:
                        warnings.warn(f"Failed to compute Hessian for {param}: {e}")
                        hessian_diag[param] = None
            
            # 上下文退出时会自动清理梯度并恢复原始状态
        
        return hessian_diag
    
    def _estimate_hessian_trace(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_samples: int = 10,
        warn_large_model: bool = True
    ) -> float:
        """
        使用 Hutchinson 方法估计 Hessian 的迹（内存友好）
        
        ✅ 修复：不会污染模型的原始梯度
        
        参数:
            inputs: 输入数据
            targets: 目标值
            num_samples: 采样次数
            warn_large_model: 是否显示大模型警告
        
        返回:
            float: Hessian迹的估计值
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        if warn_large_model and total_params > 1000000:
            warnings.warn(
                f"Model has {total_params:,} parameters. "
                f"Hessian trace estimation may be slow.",
                UserWarning
            )
        
        # ✅ 使用梯度隔离上下文
        with self._preserve_gradients():
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # 计算梯度
            grads = torch.autograd.grad(
                loss, self.model.parameters(), 
                create_graph=True,
                retain_graph=False
            )
            
            # 过滤掉 None 梯度
            valid_grads = [g for g in grads if g is not None]
            if not valid_grads:
                return 0.0
            
            grad_vec = torch.cat([g.view(-1) for g in valid_grads])
            
            trace_estimate = 0.0
            for _ in range(num_samples):
                # 生成 Rademacher 随机向量
                v = torch.randint(0, 2, grad_vec.shape, device=grad_vec.device).float()
                v = v * 2 - 1
                
                grad_dot_v = (grad_vec * v).sum()
                
                # 计算 Hv
                hv = torch.autograd.grad(
                    grad_dot_v, self.model.parameters(),
                    retain_graph=False,
                    create_graph=False
                )
                
                hv_vec = torch.cat([h.view(-1) for h in hv if h is not None])
                trace_estimate += (v * hv_vec).sum().item()
            
            # 上下文退出时会自动清理梯度并恢复原始状态
        
        return trace_estimate / num_samples
    
    def is_saddle_point(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.01,
        use_hessian: bool = False
    ) -> bool:
        """
        判断当前位置是否可能是鞍点
        
        ✅ 修复：不会污染模型的原始梯度
        
        参数:
            inputs: 输入数据
            targets: 目标值
            threshold: 梯度/曲率阈值
            use_hessian: 是否使用Hessian迹进行精确判断（更准确但更慢）
        
        返回:
            bool: 是否可能是鞍点
        """
        # ✅ 使用梯度隔离上下文
        with self._preserve_gradients():
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            # 检查梯度是否接近零
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            if grad_norm > threshold:
                return False
            
            curvature = self.compute_curvature(inputs, targets)
            
            if curvature > threshold:
                return False
            
            # 可选：使用Hessian进行精确判断
            if use_hessian:
                try:
                    hessian_trace = self._estimate_hessian_trace(inputs, targets)
                    return abs(hessian_trace) < threshold * 10
                except Exception as e:
                    warnings.warn(f"Hessian estimation failed: {e}")
                    return True
            
            return True
    
    def get_landscape_info(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """
        获取地形信息摘要（快速版本）
        
        ✅ 修复：不会污染模型的原始梯度
        
        返回:
            dict: 包含 loss, grad_norm, curvature, terrain_type, is_saddle
        """
        # ✅ 使用梯度隔离上下文
        with self._preserve_gradients():
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item() ** 2
            grad_norm = grad_norm ** 0.5
            
            curvature = self.compute_curvature(inputs, targets)
            
            # 地形分类
            if grad_norm < 0.01 and curvature < 0.01:
                terrain_type = 'saddle_or_plateau'
            elif curvature > 1.0:
                terrain_type = 'steep'
            elif curvature > 0.1:
                terrain_type = 'moderate'
            else:
                terrain_type = 'flat'
            
            return {
                'loss': loss.item(),
                'grad_norm': grad_norm,
                'curvature': curvature,
                'terrain_type': terrain_type,
                'is_saddle': grad_norm < 0.01 and curvature < 0.01,
            }
    
    def get_landscape_info_detailed(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        estimate_hessian: bool = True
    ) -> Dict[str, Any]:
        """
        获取详细的地形信息（包含Hessian迹估计）
        
        ✅ 修复：不会污染模型的原始梯度
        
        参数:
            inputs: 输入数据
            targets: 目标值
            estimate_hessian: 是否估计Hessian迹（较慢但更准确）
        
        返回:
            dict: 详细的地形信息
        """
        base_info = self.get_landscape_info(inputs, targets)
        
        if estimate_hessian:
            try:
                hessian_trace = self._estimate_hessian_trace(inputs, targets)
                base_info['hessian_trace'] = hessian_trace
                
                # 更新地形分类（基于Hessian迹）
                if base_info['is_saddle']:
                    if hessian_trace > 0.01:
                        base_info['terrain_type'] = 'local_minimum'
                    elif hessian_trace < -0.01:
                        base_info['terrain_type'] = 'local_maximum'
                    else:
                        base_info['terrain_type'] = 'saddle_point'
            except Exception as e:
                base_info['hessian_trace'] = None
                base_info['hessian_error'] = str(e)
        
        return base_info
    
    def visualize_landscape_1d(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        param: torch.nn.Parameter,
        direction: torch.Tensor = None,
        n_points: int = 50,
        scale: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        在1D方向上可视化损失地形（用于调试）
        
        ✅ 修复：不会污染模型的原始梯度
        
        参数:
            inputs: 输入数据
            targets: 目标值
            param: 要分析的参数
            direction: 扰动方向（None则随机）
            n_points: 采样点数
            scale: 扰动范围
        
        返回:
            dict: 包含 positions, losses
        """
        if direction is None:
            direction = torch.randn_like(param)
            direction = direction / direction.norm()
        
        original_param = param.data.clone()
        positions = torch.linspace(-scale, scale, n_points)
        losses = []
        
        # 保存原始梯度状态
        with self._preserve_gradients():
            for pos in positions:
                param.data = original_param + pos * direction
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                losses.append(loss.item())
        
        # 恢复原始参数
        param.data = original_param
        
        return {
            'positions': positions,
            'losses': torch.tensor(losses),
        }

# potential_math/landscape.py - 在原有代码末尾添加

# ... 原有代码保持不变 ...

# ========== 新增：地形增强分析 ==========

#class LossLandscape:
    # ... 原有代码保持不变 ...
    
    # ========== 新增方法 ==========

    def compute_morse_index(
        self, 
        inputs: torch.Tensor,
        targets: torch.Tensor,
        param_point: Optional[torch.Tensor] = None
    ) -> int:
        """
        计算临界点的Morse指数
    
        Morse指数 = Hessian矩阵负特征值的个数
        表示负曲率方向的数量（鞍点的指数）
    
        参数:
            inputs: 输入数据
            targets: 目标值
            param_point: 可选，指定参数位置（展平的参数向量）
                    如果为None，使用当前模型参数
    
        返回:
            int: Morse指数
    
        示例:
            >>> # 计算当前点的Morse指数
            >>> morse = landscape.compute_morse_index(X, y)
            >>> 
            >>> # 计算指定点的Morse指数
            >>> point = torch.cat([p.view(-1) for p in model.parameters()])
            >>> morse = landscape.compute_morse_index(X, y, point)
        """
        params = list(self.model.parameters())
    
        # 验证参数点形状
        if param_point is not None:
            total_params = sum(p.numel() for p in params)
            if param_point.numel() != total_params:
                raise ValueError(
                    f"param_point has {param_point.numel()} elements, "
                    f"but model has {total_params} parameters"
                )
    
        # 保存原始状态
        original_states = [p.data.clone() for p in params]
    
        try:
            # 设置到指定点
            if param_point is not None:
                offset = 0
                for p in params:
                    n = p.numel()
                    p.data = param_point[offset:offset+n].view_as(p)
                    offset += n
        
            # 计算Hessian（内部已有梯度隔离）
            hessian_diag = self.compute_hessian_diag(inputs, targets)
        
            # 统计负特征值
            negative_count = 0
            for h in hessian_diag.values():
                if h is not None:
                    negative_count += (h < 0).sum().item()
        
            return negative_count
        
        finally:
            # 恢复原始参数
            for p, orig in zip(params, original_states):
                p.data = orig
    
    def analyze_terrain_shape(self, inputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
        """
        特征地形分析（第三篇）
        
        分析损失地形的形状特征：
        - 条件数：地形狭长度
        - 曲率分布：各向异性程度
        - 地形类型：峡谷/盆地/高原
        
        返回:
            dict: 地形特征
        """
        with self._preserve_gradients():
            # 计算梯度和Hessian
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            
            # 收集梯度
            grads = []
            for param in self.model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
            
            if not grads:
                return {'error': 'No gradients'}
            
            grad_vec = torch.cat(grads)
            grad_norm = grad_vec.norm().item()
            
            # 计算曲率分布
            curvatures = []
            for param in self.model.parameters():
                if param.grad is not None:
                    # 近似曲率
                    param_norm = param.norm().item()
                    grad_norm_p = param.grad.norm().item()
                    if param_norm > 0:
                        curvatures.append(grad_norm_p / param_norm)
            
            if curvatures:
                mean_curv = np.mean(curvatures)
                max_curv = np.max(curvatures)
                min_curv = np.min(curvatures)
                condition = max_curv / (min_curv + 1e-8)
            else:
                mean_curv = max_curv = min_curv = condition = 0.0
            
            # 地形分类
            if condition > 100:
                terrain_type = 'extreme_ridge'
            elif condition > 10:
                terrain_type = 'ridge'
            elif grad_norm < 0.01 and mean_curv < 0.01:
                terrain_type = 'plateau'
            elif mean_curv > 1.0:
                terrain_type = 'steep_basin'
            else:
                terrain_type = 'gentle_basin'
            
            return {
                'gradient_norm': grad_norm,
                'mean_curvature': mean_curv,
                'max_curvature': max_curv,
                'min_curvature': min_curv,
                'condition_number': condition,
                'terrain_type': terrain_type,
                'is_isotropic': condition < 10,
                'is_anisotropic': condition >= 10,
            }
    
    def get_persistence_analysis(
        self,
        loss_trajectory: List[float],
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        持久性分析（第四篇）
        
        分析损失轨迹中的拓扑特征持久性
        
        参数:
            loss_trajectory: 损失值历史
            threshold: 显著性阈值
        
        返回:
            dict: 持久性分析结果
        """
        if len(loss_trajectory) < 3:
            return {'error': 'Trajectory too short'}
        
        # 检测局部极值
        minima = []
        maxima = []
        
        for i in range(1, len(loss_trajectory) - 1):
            if loss_trajectory[i] < loss_trajectory[i-1] and loss_trajectory[i] < loss_trajectory[i+1]:
                minima.append((i, loss_trajectory[i]))
            elif loss_trajectory[i] > loss_trajectory[i-1] and loss_trajectory[i] > loss_trajectory[i+1]:
                maxima.append((i, loss_trajectory[i]))
        
        # 计算持久性
        persistent_features = []
        for i, val in minima:
            # 找到该极小点消失的阈值
            death = None
            for j, (_, max_val) in enumerate(maxima):
                if max_val > val:
                    death = max_val
                    break
            
            if death is None:
                death = max(loss_trajectory)
            
            persistence = death - val
            if persistence > threshold:
                persistent_features.append({
                    'step': i,
                    'value': val,
                    'death': death,
                    'persistence': persistence,
                    'type': 'minimum'
                })
        
        return {
            'n_minima': len(minima),
            'n_maxima': len(maxima),
            'persistent_features': persistent_features,
            'significant_features': len(persistent_features),
            'noise_estimate': threshold,
            'topological_complexity': len(minima) + len(maxima),
        }

