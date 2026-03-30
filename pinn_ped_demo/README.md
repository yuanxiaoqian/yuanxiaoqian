# PINN + Pedestrian Dynamics MVP (3-day demo)

这是一个可直接在 PyCharm 打开的最小项目，用于快速演示：

1. **宏观 Field-PINN**：密度/速度场拟合 + 连续性方程残差。  
2. **微观 Traj-PINN**：轨迹拟合 + 简化 SFM-ODE 残差。  
3. **跨尺度占位接口**：KDE 粗粒化（用于后续多尺度一致性损失）。

## 项目结构

```text
pinn_ped_demo/
  requirements.txt
  configs/
  scripts/
  pinn_ped/
```

## 快速开始

```bash
pip install -r requirements.txt
python scripts/preprocess_madras.py --config configs/madras_smallroi.yaml
python scripts/train_field_pinn.py --config configs/madras_smallroi.yaml
python scripts/train_traj_pinn.py --config configs/madras_smallroi.yaml
python scripts/eval_and_plot.py --run_dir outputs/runs
```

## 数据说明

- `data/raw/`：放原始数据（本仓库不分发）。
- `data/processed/madras_roi.npz`：由预处理脚本生成的演示小样本。

## 当前实现边界

- 提供最小可跑框架与占位损失；不是完整论文实现。
- 微观 ODE 使用简化社会力项，方便三天演示快速收敛。
- 可扩展项：SA-PINN、MultiAdam、RoPINN、跨尺度一致性联合训练。
