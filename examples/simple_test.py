"""
简单测试：验证势场优化器在简单任务上的效果

运行方式：
    python simple_test.py
"""

import torch
import torch.nn as nn
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from potential_math import PotentialOptimizer, LossLandscape, set_seed

# 配置matplotlib后端（解决无显示器环境问题）
import matplotlib
import os
if 'DISPLAY' not in os.environ or os.environ.get('MPLBACKEND') == 'agg':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def test_on_simple_problem():
    """在简单回归问题上测试优化器"""
    
    set_seed(42)
    
    # 创建简单数据集：y = 2*x + noise
    X = torch.randn(100, 1)
    y = 2 * X + 0.1 * torch.randn(100, 1)
    
    # 优化器配置
    optimizer_configs = {
        'Adam': lambda params: torch.optim.Adam(params, lr=0.1),
        'SGD': lambda params: torch.optim.SGD(params, lr=0.1),
        'Potential': lambda params: PotentialOptimizer(params, lr=0.1)
    }
    
    results = {}
    
    for name, optimizer_factory in optimizer_configs.items():
        # 重新初始化模型
        model = nn.Linear(1, 1)
        criterion = nn.MSELoss()
        optimizer = optimizer_factory(model.parameters())
        
        losses = []
        
        # 训练100步
        for step in range(100):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        results[name] = losses
        print(f"{name}: final loss = {losses[-1]:.6f}")
    
    # 绘制对比图
    plt.figure(figsize=(10, 6))
    for name, losses in results.items():
        plt.plot(losses, label=name)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Optimizer Comparison on Simple Regression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    print("Figure saved to optimizer_comparison.png")
    plt.close()
    
    return results


def test_with_landscape_analysis():
    """测试地形分析功能"""
    
    set_seed(42)
    
    # 创建更复杂的模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    # 创建数据
    X = torch.randn(50, 10)
    y = torch.randn(50, 1)
    
    # 分析地形
    landscape = LossLandscape(model)
    info = landscape.get_landscape_info(X, y)
    
    print("\n=== 地形分析结果 ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # 获取详细地形信息
    info_detailed = landscape.get_landscape_info_detailed(X, y)
    print("\n=== 详细地形分析 ===")
    for key, value in info_detailed.items():
        if value is not None:
            print(f"{key}: {value}")
    
    return info


def test_potential_superpose():
    """测试势场叠加功能"""
    
    from potential_math import potential_superpose, stable_potential_superpose
    
    print("\n=== 势场叠加测试 ===")
    
    # 创建两个高斯势场
    def gaussian1(x):
        return torch.exp(-x**2)
    
    def gaussian2(x):
        return 0.5 * torch.exp(-(x-1)**2)
    
    x = torch.linspace(-3, 4, 100)
    
    # 计算叠加
    phi_total = potential_superpose([gaussian1, gaussian2], c=1.0, points=x)
    
    print(f"叠加后的势场范围: [{phi_total.min().item():.4f}, {phi_total.max().item():.4f}]")
    print("势场叠加测试完成 ✓")
    
    # 测试稳定版本
    phi1 = torch.exp(-x**2)
    phi2 = 0.5 * torch.exp(-(x-1)**2)
    phi_stable = stable_potential_superpose([phi1, phi2], c=1.0)
    
    print(f"稳定叠加后的势场范围: [{phi_stable.min().item():.4f}, {phi_stable.max().item():.4f}]")
    
    return phi_total


if __name__ == '__main__':
    print("=" * 50)
    print("势场数学库 - 修复版测试 v0.3.1")
    print("=" * 50)
    
    # 测试1：优化器对比
    print("\n[测试1] 优化器对比")
    test_on_simple_problem()
    
    # 测试2：地形分析
    print("\n[测试2] 地形分析")
    test_with_landscape_analysis()
    
    # 测试3：势场叠加
    print("\n[测试3] 势场叠加")
    test_potential_superpose()
    
    print("\n✅ 所有测试完成")
