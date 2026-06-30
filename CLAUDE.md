# CLAUDE.md

本文件为 Claude Code（claude.ai/code）在本仓库中工作时提供指引。

## 项目简介

**Lift-Splat-Shoot**（ECCV 2020）的 PyTorch 实现。该模型将自动驾驶车辆上多个相机的图像编码为鸟瞰图（BEV）特征图，并输出 BEV 语义分割结果（车辆掩码）。基于 nuScenes 数据集运行。

## 命令

所有入口都通过 [fire](https://github.com/google/python-fire) 在 `main.py` 中分发。第一个位置参数选择子命令；第二个是 nuScenes 数据划分（`mini` 或 `trainval`）。

```bash
# 安装依赖（没有 requirements.txt —— 除 torch/torchvision 外，仅需以下 Python 依赖）
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0

# 训练（日志写入 ./runs，用 `tensorboard --logdir=./runs --bind_all` 查看）
python main.py train mini --dataroot=NUSCENES_ROOT --logdir=./runs --gpuid=0

# 用 checkpoint 评估 IOU
python main.py eval_model_iou mini --modelf=MODEL.pt --dataroot=NUSCENES_ROOT

# 可视化预测结果（需要 nuScenes 地图扩展数据）
python main.py viz_model_preds mini --modelf=MODEL.pt --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT

# 通过将 LiDAR 投影到图像上，校验外参/内参是否正确
python main.py lidar_check mini --dataroot=NUSCENES_ROOT --viz_train=False

# 验证 QuickCumsum 的梯度与 autograd 参考实现是否一致
python main.py cumsum_check mini --dataroot=NUSCENES_ROOT
```

仓库没有配置测试套件、linter 或 formatter。`cumsum_check` 最接近一个单元测试 —— 它验证体素池化中使用的自定义 autograd 函数 `QuickCumsum` 与慢速参考实现 `cumsum_trick` 的梯度是否一致。

## 架构

整个流水线以三个阶段命名，全部实现于 `src/models.py::LiftSplatShoot`：

1. **Lift**（`CamEncode` + `get_cam_feats`）：N 个相机图像（nuScenes 通常是 5–6 个）分别送入 EfficientNet-B0。一个 1×1 的 `depthnet` 卷积头输出 `D+C` 个通道 —— 前 `D` 个通道是基于 `dbound` 离散深度区间做了 softmax 的深度分布，后 `C` 个通道是外观特征。两者外积得到形状为 `(B, N, D, H', W', C)` 的深度加权特征体。

2. **Splat**（`get_geometry` + `voxel_pooling`）：预计算的 `frustum` 张量 `(D, H', W', 3)` 保存了每个特征图像素的 (u, v, depth)。`get_geometry` 先撤销数据增强的后处理变换，然后提升到相机坐标系，再用每个相机的外参/内参把每个点变换到车体（ego）坐标系。`voxel_pooling` 将这些 3D 位置离散化到由 `xbound/ybound/zbound` 定义的 BEV 网格中，按体素索引排序，并通过 `QuickCumsum`（位于 `src/tools.py` 中的自定义 autograd 算子）一次性完成每个体素内特征的求和池化。Z 维度随后被折叠进通道维，得到 2D BEV 特征图。

3. **Shoot**（`BevEncode`）：基于 ResNet-18 的编码器/解码器，带有 U-Net 风格的跳跃连接，最终输出形状为 `(B, outC, H_bev, W_bev)` 的 BEV 分割 logits。

### 坐标系与配置

两个配置字典贯穿整个代码：

- `grid_conf = {xbound, ybound, zbound, dbound}` —— 每个均为 `[min, max, step]`。`dbound` 控制 Lift 阶段的深度离散化；`xbound/ybound/zbound` 定义 Splat 阶段的 BEV 体素网格。`src/tools.py` 中的 `gen_dx_bx` 据此推导出步长、原点、网格数等张量。
- `data_aug_conf = {resize_lim, final_dim, rot_lim, H, W, rand_flip, bot_pct_lim, cams, Ncams}` —— 控制每个相机图像的数据增强。同一组增强参数（缩放/裁剪/旋转/翻转）会按样本写入 `post_rots` 和 `post_trans`，以便 `get_geometry` 在推理时能撤销这些变换。否则投影后的几何位置会是错的。

模型的批量 forward 签名为：
`forward(imgs, rots, trans, intrins, post_rots, post_trans)`，其中前三项描述外参，`intrins` 是相机内参，`post_*` 是数据增强变换。数据加载器（`src/data.py` 中的 `SegmentationData`）恰好输出这个元组，外加 `binimgs`（BEV 分割目标）。

### 文件

- `main.py` —— fire CLI 路由。新入口在这里注册。
- `src/models.py` —— `LiftSplatShoot`、`CamEncode`、`BevEncode`、`Up`。
- `src/data.py` —— `NuscData` 基类，`VizData` 和 `SegmentationData` 子类，以及 `compile_data`。
- `src/tools.py` —— 几何辅助函数、`QuickCumsum`、`SimpleLoss`（带 `pos_weight≈2.13` 的 BCE，用于抵消 BEV 车辆类别的不平衡）、IOU 指标、nuScenes 地图绘制。
- `src/explore.py` —— 非训练入口（`lidar_check`、`cumsum_check`、`eval_model_iou`、`viz_model_preds`）。
- `src/train.py` —— 单一训练循环。每 `val_step` 次迭代保存一次 checkpoint（`mini` 为 1000，`trainval` 为 10000）。

## 编辑须知

- 仓库内许多注释是中文 —— 前任维护者在 `models.py` 中详细标注了形状和数学推导。编辑时请保留，除非被明确要求清理。
- `LiftSplatShoot.use_quickcumsum` 在快速自定义 autograd 路径和慢速参考实现之间切换；调试体素池化的梯度问题时把它关掉。
- 模型没有任何与相机相关的可学习参数 —— 外参/内参只通过 `get_geometry` 进入。增减相机是配置变更（`data_aug_conf['cams']` + `Ncams`），不是架构变更。
- nuScenes 特有的路径问题由 `NuscData.fix_nuscenes_formatting` 处理：当数据布局为 `trainval/1`、`trainval/2`、… 而不是单一扁平目录时，它会静默重写文件名。
