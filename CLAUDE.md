# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 概述

本仓库是 *Lift, Splat, Shoot*（ECCV 2020, NVIDIA）的参考 PyTorch 实现：将多相机图像转换为鸟瞰图（BEV）特征网格，在 nuScenes 上端到端训练以完成 BEV 车辆分割。"lift" 步骤预测每个相机每像素的深度分布并构造特征视锥；"splat" 步骤把所有视锥栅格化到同一个自车坐标系下的 BEV 网格中。

## 依赖与安装

仓库中没有 `requirements.txt`、`setup.py`，也没有测试套件。需要手动安装：

```
pip install torch torchvision nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0 fire pyquaternion opencv-python matplotlib tqdm
```

`efficientnet_pytorch` 必须锁定 `0.7.0`，因为 `CamEncode.get_eff_depth` 直接访问了 EfficientNet 的私有内部成员（`_blocks`、`_swish`、`_bn0`、`_conv_stem`、`_global_params`）。

nuScenes 数据不在仓库内。数据必须位于 `{dataroot}/{version}`——`compile_data` 会把 `os.path.join(dataroot, version)` 作为 nuScenes 根目录传入，因此 `--dataroot=/data/nuscenes --version=mini` 实际期望的路径是 `/data/nuscenes/mini/v1.0-mini`。

## 命令

所有入口都由 `main.py` 通过 `fire` 分发：

```
python main.py train mini|trainval --dataroot=/data/nuscenes --logdir=./runs --gpuid=0
python main.py eval_model_iou mini|trainval --modelf=MODEL_LOCATION --dataroot=/data/nuscenes
python main.py viz_model_preds mini|trainval --modelf=MODEL --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT
python main.py lidar_check mini|trainval --dataroot=NUSCENES_ROOT [--viz_train=True]
python main.py cumsum_check mini|trainval --dataroot=NUSCENES_ROOT
tensorboard --logdir=./runs --bind_all
```

- 目标函数的任意参数都可以在命令行覆盖（fire 的约定），例如 `--bsz=8`、`--final_dim="(128,352)"`、`--dbound="[4.0,45.0,1.0]"`。
- `--gpuid` 默认是 **1**；`gpuid < 0` 表示使用 CPU。但该选项只生效了一半：`train` 和 `eval_model_iou` 会无条件地对 loss 调用 `.cuda(gpuid)`（见"陷阱"一节）。
- `cumsum_check` 是最接近测试的东西：它遍历**整个训练集**（中途不会停止），对每个 batch 用同一个随机初始化的模型，分别走 autograd 路径（`cumsum_trick`）和自定义 `QuickCumsum` 各做一次前向和反向，打印 `out.mean()` 和 `depthnet.weight.grad.mean()`，两组数应当一致。两条路径的前向代码相同，所以 `out.mean()` 完全相等；梯度均值有 float32 舍入级别的微小差异，属于正常现象。修改 `voxel_pooling` 或 `QuickCumsum` 之后请运行它。
- 预训练模型的下载链接在 `README.md` 中。车辆 IoU：论文报告 32.07，本仓库的预训练模型为 33.03（见 README 中的表格），二者都是在完整 nuScenes 验证集上测得的；在 mini 上测出的数字与之不可比。
- 可视化命令会把 `.jpg` 文件写到当前工作目录（已被 `.gitignore` 忽略）。`viz_model_preds` 还需要 nuScenes 地图扩展数据，通过 `--map_folder` 指定。

## 架构

数据流全部位于 `src/` 内：

**`data.py` —— 数据集。** `NuscData` 是基类；有两个具体解析器，由 `compile_data` 的 `parser_name` 选择：
- `SegmentationData`（被 `train`、`eval_model_iou`、`viz_model_preds`、`cumsum_check` 使用）返回 7 元组：`imgs, rots, trans, intrins, post_rots, post_trans, binimgs`。
- `VizData`（被 `lidar_check` 使用）在上述 7 项之外再追加 `lidar_data`（原始点云）作为第 8 项。元组长度弄错会导致整条流水线崩溃。

每个 loader 的 batch 都必须严格按这个顺序解包；`train`、`get_val_info` 以及 explore 中的各函数都硬编码了该解包顺序。

数据集的关键细节：
- 当 nuScenes 以 `trainval/1`、`trainval/2`……分片布局存储（而非标准布局）时，`fix_nuscenes_formatting` 会修正 `filename` 字段。
- `sample_augmentation()` 是在 `get_image_data` 循环内部**对每个相机各调用一次**，因此同一个样本的各相机会获得各自独立的随机 resize/crop/flip/rotate。由于每个相机都对应计算了 `post_rots`/`post_trans`，几何正确性不受影响，但不要假设各相机共享同一套增强。
- `rots`/`trans`/`intrins` 来自 `calibrated_sensor`；`post_rots`/`post_trans` 只编码图像空间的增强。`get_geometry` 会先抵消后者，再应用前者。
- `binimgs` 是 `vehicle.*` 标注框在 BEV 上的栅格化结果，通过在自车坐标系下用 `cv2.fillPoly` 投影框的底面角点得到。
- 训练时 `Ncams=5`，从 6 个 `cams` 中随机选取。

**`models.py` —— 网络**（`LiftSplatShoot` = lift → splat → BEV head）：
1. `CamEncode`（EfficientNet-B0 主干，手动抽取特征金字塔各层输出，`up1` 融合 `reduction_5` + `reduction_4`）在 1/16 分辨率下输出 `D + C` 个通道。它被拆分为深度 logit 图（`D` 个由 `dbound` 决定的 bin）和 `C=64` 维上下文向量；lift 后的特征是二者的外积 `softmax(depth) ⊗ context` → `B×N×D×H/16×W/16×C`。
2. `get_geometry` 把预计算的 `self.frustum`（在 `create_frustum` 中构建的 `D×H/16×W/16×3` 射线网格）先经 `post_rots`/`post_trans` 的逆变换，再经 `rots @ intrins.inverse()` 和 `trans` 映射到自车坐标系，并显式执行透视除法。
3. `voxel_pooling` 把自车坐标点离散化到 BEV 网格上，丢弃落在 `xbound/ybound/zbound` 之外的点，按打包后的线性 rank 排序使同一体素的点连续排列，然后用 `QuickCumsum` 做求和池化（自定义 autograd `Function`；`tools.py` 中的 `cumsum_trick` 是可微的参考/慢速实现，通过 `model.use_quickcumsum` 切换）。结果形状为 `B×C×Z×X×Y`，随后把 Z 折叠进通道维。
4. `BevEncode` 是截断版 `resnet18`（layers 1–3），配合 U-Net 式的上采样恢复到完整 BEV 分辨率，输出 `outC=1` 个 logit。

固定常量：`downsample=16`、`camC=64`、`outC=1`（仅做车辆分割）。

**`tools.py` —— 共享数学/IO。** `gen_dx_bx` 把 `[min, max, step]` 边界转换为格子尺寸 `dx`、首个格子中心 `bx` 和格子数 `nx`；`models.py` 和 `data.py` 中所有涉及 BEV 网格的地方都走这个函数。此外还包含 lidar 加载、`ego_to_cam`/`cam_to_ego`、`img_transform`（返回与之匹配的 `post_rot`/`post_trans`）、归一化、`SimpleLoss`（带 `pos_weight` 的 `BCEWithLogitsLoss`）、`get_batch_iou`（输入是**未过 sigmoid** 的 logits），以及 nuScenes 地图绘制辅助函数。

**`explore.py` —— 评估与可视化。** 每个公开函数都各自重复声明了一整套 `grid_conf`/`data_aug_conf` 默认值，与 `train.py` 形成重复。修改默认值时必须同时改这四处（`train`、`lidar_check`、`cumsum_check`、`eval_model_iou`、`viz_model_preds`），否则评估/可视化构建出的模型会与 checkpoint 不匹配。

### 网格与增强配置

`grid_conf` 使用 `[min, max, step]` 三元组。默认值：`xbound`/`ybound` 为 `[-50, 50, 0.5]` → 200×200 个 0.5 m 的 BEV 格子；`zbound [-10, 10, 20]` → 只有 1 个 Z bin；`dbound [4, 45, 1]` → 41 个深度 bin。边界必须能被步长整除，因为 `nx` 是用浮点除法计算后再转成 `LongTensor` 的。

`data_aug_conf`：`H`/`W` 为原始图像 900/1600，`final_dim` 为网络输入 (128, 352)，`resize_lim`/`bot_pct_lim`/`rot_lim`（单位为度）在训练时对每个相机采样，`rand_flip`，`cams`（nuScenes 的 6 个相机），`Ncams`。

## 陷阱

- **CPU 模式在两处是坏的。** `train` 中写的是 `SimpleLoss(pos_weight).cuda(gpuid)`，`eval_model_iou` 中写的是 `SimpleLoss(1.0).cuda(gpuid)`；尽管它们上方已经做了 CPU 的设备选择，`gpuid < 0` 时依然会报错。同样地，`eval_model_iou` 把 `pos_weight` 硬编码为 `1.0`，而训练用的是 `2.13`，因此评估打印出的 loss 并不是训练目标。
- **Checkpoint 与配置强耦合。** `torch.load` 恢复的是裸 `state_dict`，所以 `grid_conf`、`final_dim`、`dbound`、`Ncams` 必须与训练时一致，否则形状/几何会静默地发生偏差。
- **BEV 网格的轴序在多处被转置。** `voxel_pooling` 填充的是 `(Z, X, Y)`，其中 X 是第一个空间轴；绘图代码用 `[..., Y, X]` 索引并调用 `.T`，而 `binimgs` 通过交换角点（`pts[:, [1, 0]] = pts[:, [0, 1]]`）来对齐。新增 BEV 相关代码时请先确认约定。
- 当前工作树不是一个 git 仓库；`*.jpg`、`*.png`、`*.pt`、`runs/` 都已被 `.gitignore` 忽略。
