"""
损失地形分析工具

用于分析损失函数的地形特征：曲率、鞍点、局部极小等
"""

import torch
from typing import Optional, List, Dict, Any, Callable


class LossLandscape:
    """
    损失地形分析器
    
    参数:
        model: PyTorch模型
        loss_fn: 损失函数（默认: MSELoss）
    """
    
    def __init__(self, model: torch.nn.Module, loss_fn: Optional[Callable] = None):
        self.model = model
        self.loss_fn = loss_fn or torch.nn.MSELoss()
    
    def compute_curvature(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        param_filter: Optional[List[torch.nn.Parameter]] = None
    ) -> float:
        """
        计算损失函数在当前参数处的平均曲率
        """
        params = param_filter or list(self.model.parameters())
        
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
        
        return curvature
    
    def compute_hessian_diag(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        param_filter: Optional[List[torch.nn.Parameter]] = None
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """
        计算Hessian矩阵的对角线（修复内存泄漏版）
        
        注意：只适用于小模型，大模型会很慢
        """
        params = param_filter or list(self.model.parameters())
        hessian_diag = {}
        
        for param in params:
            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            
            # 对每个参数分别计算，避免保留整个图
            grad = torch.autograd.grad(loss, param, create_graph=True)[0]
            if grad is not None:
                # ✅ 不保留计算图
                hessian = torch.autograd.grad(grad.sum(), param)[0]
                hessian_diag[param] = hessian
        
        return hessian_diag
    
    def _estimate_hessian_trace(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_samples: int = 10
    ) -> float:
        """
        使用 Hutchinson 方法估计 Hessian 的迹（修复内存泄漏版）
        """
        self.model.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        
        # 只创建一次计算图
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.view(-1) for g in grads if g is not None])
        
        trace_estimate = 0.0
        for _ in range(num_samples):
            # 随机 Rademacher 向量（±1）
            v = torch.randint(0, 2, grad_vec.shape, device=grad_vec.device).float()
            v = v * 2 - 1
            
            grad_dot_v = (grad_vec * v).sum()
            # ✅ 每次迭代不保留图
            hv = torch.autograd.grad(grad_dot_v, self.model.parameters(), retain_graph=False)
            hv_vec = torch.cat([h.view(-1) for h in hv if h is not None])
            
            trace_estimate += (v * hv_vec).sum().item()
        
        return trace_estimate / num_samples
    
    def is_saddle_point(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.01
    ) -> bool:
        """判断当前位置是否可能是鞍点"""
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
        
        return curvature < threshold
    
    def get_landscape_info(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """获取地形信息摘要"""
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
        targets: torch.Tensor
    ) -> Dict[str, Any]:
        """获取详细的地形信息（包含Hessian迹估计）"""
        base_info = self.get_landscape_info(inputs, targets)
        
        try:
            hessian_trace = self._estimate_hessian_trace(inputs, targets)
            base_info['hessian_trace'] = hessian_trace
            
            if base_info['is_saddle']:
                if hessian_trace > 0:
                    base_info['terrain_type'] = 'local_minimum'
                elif hessian_trace < 0:
                    base_info['terrain_type'] = 'local_maximum'
                else:
                    base_info['terrain_type'] = 'saddle_point'
        except Exception as e:
            base_info['hessian_trace'] = None
        
        return base_info