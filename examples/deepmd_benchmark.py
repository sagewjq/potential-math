"""
DeePMD-kit集成示例

说明：
    势场优化器在分子动力学领域的应用示例

    本示例展示如何使用势场优化器训练 DeePMD 风格的模型。
    势场优化器是通用的，这只是其中一个应用场景。

    这是一个集成示例，展示如何在DeePMD-kit中使用势场优化器。
    实际使用时需要根据DeePMD-kit的API进行调整。

运行方式（需要安装DeePMD-kit）：
    python deepmd_benchmark.py --data water --epochs 100
"""

import argparse
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from potential_math import PotentialOptimizer, set_seed, compute_gradient_statistics


class MockDeepMDTrainer:
    """模拟DeePMD-kit训练器，用于演示集成方式"""
    
    def __init__(self, model, optimizer_type='adam', lr=0.001, **kwargs):
        self.model = model
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.kwargs = kwargs
        self._setup_optimizer()
        self.step_count = 0
    
    def _setup_optimizer(self):
        if self.optimizer_type == 'potential':
            self.optimizer = PotentialOptimizer(
                self.model.parameters(),
                lr=self.lr,
                curvature_sensitivity=self.kwargs.get('curvature_sensitivity', 1.0),
                saddle_escape=self.kwargs.get('saddle_escape', True),
                saddle_threshold=self.kwargs.get('saddle_threshold', 0.01),
                noise_scale=self.kwargs.get('noise_scale', 0.01),
                momentum=self.kwargs.get('momentum', 0.9)
            )
        elif self.optimizer_type == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.kwargs.get('momentum', 0.9)
            )
        else:  # adam
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr
            )
    
    def train_step(self, batch):
        """单步训练"""
        self.optimizer.zero_grad()
        loss = self._compute_loss(batch)
        loss.backward()
        
        # 梯度裁剪（可选）
        if self.kwargs.get('grad_clip'):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.kwargs['grad_clip'])
        
        self.optimizer.step()
        self.step_count += 1
        
        # 获取地形信息（仅势场优化器支持）
        terrain_info = {}
        if self.optimizer_type == 'potential' and hasattr(self.optimizer, 'get_curvature_stats'):
            terrain_info = self.optimizer.get_curvature_stats()
        
        # 获取梯度统计
        grad_stats = compute_gradient_statistics(self.model)
        
        return loss.item(), terrain_info, grad_stats
    
    def _compute_loss(self, batch):
        """模拟损失计算"""
        x, y = batch
        output = self.model(x)
        return torch.nn.MSELoss()(output, y)
    
    def train(self, dataloader, epochs=10):
        """完整训练循环"""
        losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                loss, terrain, grad_stats = self.train_step(batch)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"[{self.optimizer_type}] Epoch {epoch}: loss={avg_loss:.6f}")

        elapsed = time.time() - start_time
        return losses, elapsed


def create_mock_data(n_samples=1000, n_features=10):
    """创建模拟数据"""
    X = torch.randn(n_samples, n_features)
    y = torch.randn(n_samples, 1)
    return [(X[i:i+32], y[i:i+32]) for i in range(0, n_samples, 32)]


def integrate_with_deepmd():
    """
    集成到DeePMD-kit的方法
    """
    print("=" * 60)
    print("DeePMD-kit 势场优化器集成指南")
    print("=" * 60)
    print()
    print("要将势场优化器集成到DeePMD-kit，需要修改以下文件：")
    print()
    print("1. deepmd/optimizer.py - 添加势场优化器类")
    print("2. deepmd/trainer.py - 修改优化器初始化")
    print("3. deepmd/config.py - 添加optimizer_type配置")
    print()
    print("示例配置：")
    print('''
    {
        "optimizer": {
            "type": "potential",
            "learning_rate": 0.001,
            "curvature_sensitivity": 1.0,
            "saddle_escape": true,
            "saddle_threshold": 0.01,
            "noise_scale": 0.01
        }
    }
    ''')
    print()
    print("集成后，可以通过以下命令运行：")
    print("    dp train input.json --optimizer potential")
    print()
    
    # 模拟训练对比
    print("\n" + "=" * 60)
    print("模拟训练对比（DeePMD风格）")
    print("=" * 60)
    
    set_seed(42)
    
    # 创建简单模型（模拟DeePMD的势能网络）
    class MockPotentialNet(torch.nn.Module):
        def __init__(self, n_features=10, n_hidden=64):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(n_features, n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(n_hidden, 1)
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = MockPotentialNet()
    dataloader = create_mock_data(n_samples=2000, n_features=10)
    
    # 对比不同优化器
    results = {}
    for opt_type in ['adam', 'sgd', 'potential']:
        print(f"\n训练 {opt_type.upper()} 优化器...")
        
        # 重新初始化模型
        model = MockPotentialNet()
        trainer = MockDeepMDTrainer(
            model, 
            optimizer_type=opt_type, 
            lr=0.001,
            curvature_sensitivity=1.0,
            saddle_escape=True
        )
        
        losses, elapsed = trainer.train(dataloader, epochs=50)
        results[opt_type] = {'losses': losses, 'time': elapsed, 'final_loss': losses[-1]}
        
        print(f"[{opt_type}] Final loss: {losses[-1]:.6f}, Time: {elapsed:.2f}s")
    
    # 结果汇总
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"{'优化器':<12} {'最终损失':<12} {'训练时间':<12} {'相对加速':<12}")
    print("-" * 50)
    
    baseline_time = results['adam']['time']
    for opt_type, res in results.items():
        speedup = baseline_time / res['time']
        print(f"{opt_type:<12} {res['final_loss']:<12.6f} {res['time']:<12.2f}s {speedup:<12.2f}x")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='water', help='数据集名称')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--optimizer', type=str, default='both', 
                        choices=['adam', 'sgd', 'potential', 'both'])
    args = parser.parse_args()
    
    integrate_with_deepmd()