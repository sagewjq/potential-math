# ⚡ 势场数学库 (Potential Math)

[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![Gitee](https://gitee.com/yourname/potential-math/badge/star.svg?theme=dark)](https://gitee.com/yourname/potential-math)

基于**频率调制统一理论**的地形感知优化器。用物理规律指导深度学习优化。

---

## 🎯 核心特性

- **地形感知**：根据损失曲率自适应调整学习率
- **鞍点逃逸**：主动检测并逃离鞍点，避免陷入局部最优
- **零成本集成**：只需一行代码替换现有优化器
- **可解释性**：实时获取曲率统计信息
- **理论支撑**：频率调制统一理论，有完整的物理基础

---

## 📦 安装

### 方式一：从 Gitee 安装（国内推荐）
```bash
pip install git+https://gitee.com/sageapollo/potential-math.git
```

方式二：从 GitHub 安装

```bash
pip install git+https://github.com/sagewjq/potential-math
```
方式三：本地安装（开发模式）

```bash
git clone https://gitee.com/sageapollo/potential-math.git
cd potential-math
pip install -e .
```

---

🚀 快速开始

基础使用

```python
import torch
import torch.nn as nn
from potential_math import PotentialOptimizer

# 创建模型
model = nn.Linear(10, 1)

# 只需替换优化器！
optimizer = PotentialOptimizer(
    model.parameters(),
    lr=0.001,
    curvature_sensitivity=1.0,  # 曲率敏感度
    saddle_escape=True           # 启用鞍点逃逸
)

# 训练循环保持不变
for epoch in range(100):
    optimizer.zero_grad()
    loss = model(data).mean()
    loss.backward()
    optimizer.step()
    
    # 可选：获取曲率统计
    stats = optimizer.get_curvature_stats()
    print(f"Mean curvature: {stats['mean_curvature']:.4f}")
```

对比实验

```python
from potential_math import PotentialOptimizer

# Adam 优化器
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001)

# 势场优化器
optimizer_potential = PotentialOptimizer(model.parameters(), lr=0.001)

# 训练效果对比（详见 examples/simple_test.py）
```

---

🔬 地形分析

```python
from potential_math import LossLandscape

landscape = LossLandscape(model)

# 获取地形信息
info = landscape.get_landscape_info(inputs, targets)
print(f"地形类型: {info['terrain_type']}")  # steep, flat, saddle_or_plateau
print(f"曲率: {info['curvature']:.4f}")
print(f"是否鞍点: {info['is_saddle']}")

# 详细分析（包含 Hessian 迹估计）
info_detailed = landscape.get_landscape_info_detailed(inputs, targets)
```

---

🧩 势场叠加

```python
from potential_math import potential_superpose, stable_potential_superpose

# 定义多个势场
def gaussian1(x):
    return torch.exp(-x**2)

def gaussian2(x):
    return 0.5 * torch.exp(-(x-1)**2)

# 自动生成交叉项
x = torch.linspace(-3, 4, 100)
phi_total = potential_superpose([gaussian1, gaussian2], c=1.0, points=x)

# 数值稳定版本
phi1 = torch.exp(-x**2)
phi2 = 0.5 * torch.exp(-(x-1)**2)
phi_stable = stable_potential_superpose([phi1, phi2], c=1.0)
```

---

📊 参数说明

PotentialOptimizer 参数

参数 默认值 说明
lr 0.01 基础学习率
momentum 0.0 动量因子（0=不启用）
curvature_sensitivity 1.0 曲率敏感度，越大对曲率越敏感
saddle_escape True 是否启用鞍点逃逸
saddle_threshold 0.01 鞍点检测阈值
noise_scale 0.01 鞍点逃逸噪声尺度
grad_clip None 梯度裁剪阈值

调参建议

· 收敛慢：增大 lr 或减小 curvature_sensitivity
· 震荡：减小 lr 或增大 curvature_sensitivity
· 陷入鞍点：增大 saddle_escape 或 noise_scale

---

📁 项目结构

potential-math/
├── potential_math/              # 核心代码包
│   ├── __init__.py             # 包初始化
│   ├── optimizer.py             # 势场优化器
│   ├── landscape.py             # 损失地形分析
│   ├── superpose.py             # 势场叠加
│   └── utils.py                 # 辅助工具
├── examples/                    # 示例代码
│   ├── simple_test.py           # 简单回归示例
│   └── deepmd_benchmark.py      # 分子动力学应用
├── setup.py                     # 安装配置
├── README.md                    # 项目文档
├── LICENSE                      # MIT 许可证
└── .gitignore                   # Git 忽略文件

---

🎓 理论基础

## 🎓 理论基础与学术引用

### 📖 论文

本项目基于以下原创理论：

**频率调制统一理论及其在优化算法中的应用**  
*你的名字*  
OSF Preprints, 2026  
🔗 https://osf.io/td5jx

### 🧠 理论核心

频率调制统一理论揭示了优化过程的本质：

> 优化算法是粒子在损失地形势场中的运动过程，自适应步长由局部曲率决定。

**数学表述**：
- 运动方程：F = -E ∇Φ
- 自适应步长：η_adaptive = η_0 / (1 + α · κ)
- 曲率估计：κ = ||∇L(θ_t) - ∇L(θ_{t-1})|| / ||θ_t - θ_{t-1}||

势场优化器（PotentialOptimizer）是该理论的直接实现，验证了理论的有效性。

### 📚 如何引用

如果你在研究中使用本库，请引用以下论文：

```bibtex
@unpublished{王江祁2026frequency,
  author = {王江祁},
  title = {王江祁-b116势场优化器：频率调制统一理论在AI训练中的实验验证},
  year = {2026},
  note = {OSF Preprint},
  url = {https://osf.io/td5jx}
}
```

⚖️ 版权声明

类型 版权说明
论文 © 2026 王江祁。保留所有权利。未经许可不得用于商业出版。
代码 MIT 开源许可证，可自由使用。
引用 学术使用请规范引用，商业使用请联系作者。

注意：本代码库是对论文理论的验证实现，若对你的研究有帮助，请引用论文支持学术工作。

理论详情请阅读论文原文。

---

📈 性能表现

任务 优化器 收敛步数 加速比
简单回归 Adam 80 1.0×
简单回归 势场优化器 45 1.78×
DeePMD 模拟 Adam 1000 1.0×
DeePMD 模拟 势场优化器 600 1.67×

详细对比见 examples/simple_test.py

---

🔧 辅助工具

```python
from potential_math import set_seed, compute_gradient_statistics, compute_param_norm

# 设置随机种子
set_seed(42)

# 梯度统计
stats = compute_gradient_statistics(model)
print(f"平均梯度范数: {stats['mean_grad_norm']:.4f}")

# 参数范数
norm = compute_param_norm(model)
print(f"参数范数: {norm:.4f}")
```

---

🌟 谁在用？

如果你在使用势场数学库，欢迎通过 Issue 告诉我们！

· 深度学习研究
· 分子动力学模拟（DeePMD）
· 科学计算优化问题

---

📝 待办事项

· 添加更多示例（图像分类、NLP）
· 支持混合精度训练
· 添加 TensorBoard 可视化
· 发布到 PyPI

---

🤝 贡献

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (git checkout -b feature/AmazingFeature)
3. 提交更改 (git commit -m 'Add some AmazingFeature')
4. 推送到分支 (git push origin feature/AmazingFeature)
5. 提交 Pull Request

---

📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

---

📬 联系与交流

遇到问题？

· 📝 提交 Issue
· 💬 使用 Issue 模板：Bug报告 / 使用问题 / 功能建议

想交流？

· 📧 邮件：jq.wang@126.com
· 🌐 项目主页

你在用这个项目？

太好了！请告诉我们你的使用场景，帮助我们改进。

---

⭐ Star History

如果这个项目对你有帮助，欢迎点个 Star ⭐ 支持一下！

---

Made with ❤️ by 王江祁
