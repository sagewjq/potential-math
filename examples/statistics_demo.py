"""
势场统计演示

演示熵、温度、分布等统计量的计算
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from potential_math import PotentialOptimizer, PotentialStatistics, set_seed


def demo_statistics():
    """演示统计模块"""
    
    print("=" * 60)
    print("势场统计演示")
    print("=" * 60)
    
    set_seed(42)
    
    # 创建模型和数据
    model = nn.Linear(10, 1)
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    
    # 创建优化器和统计模块
    optimizer = PotentialOptimizer(model.parameters(), lr=0.01)
    statistics = PotentialStatistics(model, optimizer)
    
    losses = []
    gradients = []
    
    print("\n训练中记录统计信息...")
    
    for step in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        
        # 记录梯度
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        statistics.record_state(loss.item(), grads)
        
        optimizer.step()
        losses.append(loss.item())
    
    # 获取统计摘要
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)
    
    summary = statistics.get_statistics_summary()
    for key, value in summary.items():
        print(f"  {key}: {value:.6e}")
    
    print("\n" + "=" * 60)
    print("物理量解读")
    print("=" * 60)
    
    entropy = summary['entropy']
    temperature = summary['temperature']
    free_energy = summary['free_energy']
    
    print(f"\n熵 S = {entropy:.6e} J/K")
    print(f"  解读: 系统混乱程度，值越大越无序")
    
    print(f"\n温度 T = {temperature:.2f} K")
    print(f"  解读: 系统热运动剧烈程度")
    
    print(f"\n自由能 F = {free_energy:.6e} J")
    print(f"  解读: 系统可用能量")
    
    # 玻尔兹曼分布
    print("\n" + "=" * 60)
    print("玻尔兹曼分布")
    print("=" * 60)
    
    # 生成测试势场
    test_points = torch.linspace(-5, 5, 100)
    potential = test_points ** 2  # 简谐振子势
    
    distribution = statistics.boltzmann_distribution(potential)
    
    print(f"势场范围: [{potential.min():.2f}, {potential.max():.2f}]")
    print(f"分布范围: [{distribution.min():.4f}, {distribution.max():.4f}]")
    print("  解读: 低势能区域概率高，符合物理直觉")
    
    # 能量均分
    equipartition = statistics.get_equipartition_ratio()
    print(f"\n能量均分比: {equipartition:.3f}")
    if abs(equipartition - 1.0) < 0.3:
        print("  解读: 系统接近热平衡状态")
    else:
        print("  解读: 系统远离热平衡")
    
    return statistics


def demo_terrain_analysis():
    """演示地形增强分析"""
    
    print("\n" + "=" * 60)
    print("地形增强分析演示")
    print("=" * 60)
    
    from potential_math import LossLandscape
    
    set_seed(42)
    
    # 创建复杂模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    
    X = torch.randn(50, 10)
    y = torch.randn(50, 1)
    
    landscape = LossLandscape(model)
    
    # 地形形状分析
    terrain = landscape.analyze_terrain_shape(X, y)
    
    print("\n地形形状特征:")
    for key, value in terrain.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 临界点检测
    is_saddle = landscape.is_saddle_point(X, y)
    print(f"\n是否鞍点: {is_saddle}")
    
    # 地形信息
    info = landscape.get_landscape_info(X, y)
    print(f"\n地形信息:")
    print(f"  损失: {info['loss']:.6f}")
    print(f"  梯度范数: {info['grad_norm']:.6f}")
    print(f"  曲率: {info['curvature']:.6f}")
    print(f"  地形类型: {info['terrain_type']}")


if __name__ == '__main__':
    stats = demo_statistics()
    demo_terrain_analysis()
    
    print("\n✅ 势场统计演示完成")

