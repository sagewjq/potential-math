"""
GPU加速演示

演示势场数学的GPU加速效果
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from potential_math import potential_superpose, detect_hardware, set_gpu_backend


def benchmark_gpu_superpose():
    """测试GPU势场叠加性能"""
    
    print("=" * 60)
    print("GPU势场叠加性能测试")
    print("=" * 60)
    
    # 检测硬件
    config = detect_hardware()
    print(f"\n硬件信息:")
    print(f"  后端: {config['backend']}")
    print(f"  设备: {config['device']}")
    print(f"  设备名: {config.get('device_name', 'Unknown')}")
    print(f"  显存: {config.get('memory_gb', 0):.1f} GB")
    
    # 测试配置
    n_potentials_list = [10, 50, 100]
    n_points_list = [1000, 10000, 100000]
    
    # 生成测试数据
    def gaussian_potential(x, center):
        return torch.exp(-(x - center) ** 2)
    
    for n_points in n_points_list:
        print(f"\n--- 采样点数: {n_points:,} ---")
        x = torch.linspace(-5, 5, n_points)
        
        for n_potentials in n_potentials_list:
            # 生成势场
            potentials = [
                lambda x, i=i: gaussian_potential(x, -3 + 6 * i / n_potentials)
                for i in range(n_potentials)
            ]
            
            # CPU版本
            x_cpu = x.cpu()
            start = time.time()
            phi_cpu = potential_superpose(potentials, c=1.0, points=x_cpu, use_gpu=False)
            cpu_time = time.time() - start
            
            # GPU版本
            if torch.cuda.is_available():
                x_gpu = x.cuda()
                start = time.time()
                phi_gpu = potential_superpose(potentials, c=1.0, points=x_gpu, use_gpu=True)
                gpu_time = time.time() - start
                
                speedup = cpu_time / gpu_time
                
                # 验证结果
                diff = torch.abs(phi_cpu - phi_gpu.cpu()).max().item()
                
                print(f"  {n_potentials:3d}个势场: CPU={cpu_time*1000:.2f}ms, "
                      f"GPU={gpu_time*1000:.2f}ms, "
                      f"加速比={speedup:.1f}x, "
                      f"误差={diff:.2e}")
            else:
                print(f"  {n_potentials:3d}个势场: CPU={cpu_time*1000:.2f}ms, GPU不可用")


def benchmark_gpu_optimizer():
    """测试GPU优化器性能"""
    
    print("\n" + "=" * 60)
    print("GPU优化器性能测试")
    print("=" * 60)
    
    import torch.nn as nn
    from potential_math import PotentialOptimizer
    
    # 创建大模型
    model = nn.Sequential(
        nn.Linear(1000, 2000),
        nn.ReLU(),
        nn.Linear(2000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 创建优化器
    optimizer = PotentialOptimizer(model.parameters(), lr=0.001)
    
    # 生成数据
    X = torch.randn(100, 1000)
    y = torch.randn(100, 1)
    
    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()
    
    # 预热
    for _ in range(10):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()
    
    # 测试
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    for step in range(100):
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(X), y)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    print(f"\n100步训练耗时: {elapsed*1000:.2f}ms")
    print(f"平均每步: {elapsed*10:.2f}ms")


if __name__ == '__main__':
    benchmark_gpu_superpose()
    benchmark_gpu_optimizer()
    
    print("\n✅ GPU测试完成")

