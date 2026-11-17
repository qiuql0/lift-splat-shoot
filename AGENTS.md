# 仓库指南

## 项目结构与模块组织
本仓库是 Lift-Splat-Shoot（ECCV 2020, NVIDIA）的 PyTorch 参考实现：把多相机图像转换成鸟瞰图（BEV）特征网格，在 nuScenes 上端到端训练做 BEV 车辆分割。"lift" 步骤预测每个相机每像素的深度分布并构造特征视锥；"splat" 步骤把所有视锥栅格化到同一个自车坐标系下的 BEV 网格中。

所有代码都位于 `src/`：

- `main.py`：命令行入口，通过 `fire.Fire` 暴露各个子命令。
- `src/data.py`：nuScenes 数据集类（`NuscData`、`SegmentationData`、`VizData`）以及 `compile_data`。
- `src/models.py`：网络定义（`CamEncode`、`BevEncode`、`LiftSplatShoot`）以及 `compile_model`。
- `src/tools.py`：几何变换、cumsum 技巧、损失函数，以及 IoU/绘图辅助函数。
- `src/train.py`：训练循环。`src/explore.py`：评估、可视化与健全性检查。
- `src/z_test.py`：几何一致性测试 `geometry_check`，把 `get_geometry` 的输出投影回图像，检查能否还原视锥模板。
- `imgs/`：README 中使用的 GIF 动图。除 `z_test.py` 外本项目没有测试目录，也未附带测试套件。

## 依赖与安装环境
仓库未提交 `requirements.txt` / `setup.py`。本机已验证的完整版本组合见 `requirements-win.txt`（numpy / matplotlib / opencv 三项有硬性版本约束）：

```
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```

`efficientnet_pytorch` 必须锁定 `0.7.0`，因为 `CamEncode.get_eff_depth` 直接访问了 EfficientNet 的私有内部成员（`_blocks`、`_swish`、`_bn0`、`_conv_stem`、`_global_params`）。

本机环境：conda 环境 `lss`（`/home/qianli/miniconda3/envs/lss`，Python 3.11），数据在 `/home/qianli/data/nuscenes`，只有一张 RTX 3080，**所有命令都要加 `--gpuid=0`**（默认值是 1）。

## 构建、测试与开发命令
常用命令（`DATA=/home/qianli/data/nuscenes`，`MAPS=/home/qianli/data/nuscenes/mini`）：

```
python main.py train mini --dataroot=$DATA --logdir=./runs --gpuid=0
python main.py eval_model_iou mini --modelf=MODEL.pt --dataroot=$DATA --gpuid=0
python main.py viz_model_preds mini --modelf=MODEL.pt --dataroot=$DATA --map_folder=$MAPS --gpuid=0
python main.py lidar_check mini --dataroot=$DATA
python main.py cumsum_check mini --dataroot=$DATA --gpuid=0
python -m src.z_test geometry_check mini --dataroot=$DATA
tensorboard --logdir=./runs --bind_all
```

- 目标函数的任意参数都可以在命令行覆盖（fire 的约定），例如 `--bsz=8`、`--final_dim="(128,352)"`、`--dbound="[4.0,45.0,1.0]"`；`--ncams=6` 可以关掉训练时的随机丢相机。
- `--gpuid` 默认是 **1**；`gpuid < 0` 表示使用 CPU。但该选项只生效了一半：`train` 和 `eval_model_iou` 会无条件地对 loss 调用 `.cuda(gpuid)`（见"陷阱"）。
- `lidar_check` 默认用验证集，mini 上生成 **81 张图**，写到 `lss_viz/lidar_check/`。
- `cumsum_check` 是最接近测试的东西：它遍历**整个训练集**（中途不会停止），对每个 batch 用同一个随机初始化的模型，分别走 autograd 路径（`cumsum_trick`）和自定义 `QuickCumsum` 各做一次前向和反向，打印 `out.mean()` 和 `depthnet.weight.grad.mean()`。两条路径前向代码相同，所以 `out.mean()` 完全相等；梯度均值有 float32 舍入级别的微小差异，属于正常现象。
- `viz_model_preds` 还需要 nuScenes 地图扩展数据，通过 `--map_folder` 指定；输出 `.jpg` 写到当前工作目录（已被 `.gitignore` 忽略）。
- 预训练模型的下载链接在 `README.md` 中。车辆 IoU：论文报告 32.07，本仓库的预训练模型为 33.03（见 README 表格），二者都是在完整 nuScenes 验证集上测得；mini 上的数字与之不可比。

**数据集目录结构（易踩坑）**：`compile_data` 内部执行 `os.path.join(dataroot, version)`，因此 `--dataroot` 要指向包含 `<version>` 子目录的**父目录**，而不是数据集本身所在的目录：

```
/home/qianli/data/nuscenes/     <- --dataroot 传这一层
  └─ mini/                      <- version 参数（mini / trainval）
       ├─ v1.0-mini/*.json
       ├─ samples/  sweeps/
       └─ maps/
```

若报 `AssertionError: Database version not found: .../mini/v1.0-mini`，就是这个层级放错了。`viz_model_preds` 还会读取 `<dataroot>/<version>/maps/` 下的矢量地图 JSON，需要额外下载 nuScenes 地图扩展包。

## 架构（数据流）
**`data.py` —— 数据集。** `NuscData` 是基类；两个具体解析器由 `compile_data` 的 `parser_name` 选择：

- `SegmentationData`（`train`、`eval_model_iou`、`viz_model_preds`、`cumsum_check` 使用）返回 7 元组：`imgs, rots, trans, intrins, post_rots, post_trans, binimgs`。
- `VizData`（`lidar_check` 使用）再追加 `lidar_data`（原始点云）作为第 8 项。

每个 loader 的 batch 必须严格按这个顺序解包；`train`、`get_val_info` 以及 explore 中的各函数都硬编码了该解包顺序。元组长度弄错会导致整条流水线崩溃。

关键细节：

- 当 nuScenes 以 `trainval/1`、`trainval/2`……分片布局存储时，`fix_nuscenes_formatting` 会修正 `filename` 字段。
- `sample_augmentation()` 在 `get_image_data` 循环内部**对每个相机各调用一次**，因此同一个样本的各相机会获得各自独立的随机 resize/crop/flip/rotate。每个相机都对应计算了 `post_rots`/`post_trans`，几何正确性不受影响，但不要假设各相机共享同一套增强。
- `rots`/`trans`/`intrins` 来自 `calibrated_sensor`；`post_rots`/`post_trans` 只编码图像空间的增强。`get_geometry` 会先抵消后者，再应用前者。
- `binimgs` 是 `vehicle.*` 标注框在 BEV 上的栅格化结果（`cv2.fillPoly` 投影框的底面角点得到），包括自行车、摩托车，且不按可见度过滤。
- 训练时 `Ncams=5`，从 6 个相机中随机选取，用于提升对相机缺失的鲁棒性（`choose_cams`）。
- 图像在增强之后统一做 ImageNet 归一化（`normalize_img`）。**归一化参数与预训练权重绑定**：实测把未归一化的像素喂给模型，IoU 会从 0.357 掉到 0.006。

**`models.py` —— 网络**（`LiftSplatShoot` = lift → splat → BEV head）：

1. `CamEncode`（EfficientNet-B0 主干，手动抽取特征金字塔各层输出，`up1` 融合 `reduction_5` + `reduction_4`）在 1/16 分辨率下输出 `D + C` 个通道。拆成深度 logit 图（`D` 个由 `dbound` 决定的 bin）和 `C=64` 维上下文向量；lift 后的特征是二者的外积 `softmax(depth) ⊗ context` → `B×N×D×H/16×W/16×C`。
2. `get_geometry` 把预计算的 `self.frustum`（`create_frustum` 中构建的 `D×H/16×W/16×3` 射线网格，元素为 `(u', v', d)`）先经 `post_rots`/`post_trans` 的逆变换，再经 `rots @ intrins.inverse()` 和 `trans` 映射到自车坐标系。视锥里的 `d` 是相机坐标系下的 z 深度，不是直线距离。
3. `get_cam_feats` 把 `(B, N, C, D, h, w)` 重排成 `(B, N, D, h, w, C)`，使前五维与 `get_geometry` 的输出完全对齐——`voxel_pooling` 摊平后必须逐点对应，顺序错了会静默错位。
4. `voxel_pooling` 把自车坐标点离散化到 BEV 网格，丢弃落在 `xbound/ybound/zbound` 之外的点，按打包的线性 rank 排序使同一体素的点连续排列，然后用 `QuickCumsum` 做求和池化（自定义 autograd `Function`；`tools.py` 的 `cumsum_trick` 是可微的参考/慢速实现，通过 `model.use_quickcumsum` 切换）。结果 `B×C×Z×X×Y`，随后把 Z 折叠进通道维。
5. `BevEncode` 是截断版 `resnet18`（layers 1–3），配合 U-Net 式上采样恢复到完整 BEV 分辨率，输出 `outC=1` 个 logit（**没有 sigmoid**）。

固定常量：`downsample=16`、`camC=64`、`outC=1`（仅做车辆分割）。

**`tools.py` —— 共享数学/IO。** `gen_dx_bx` 把 `[min, max, step]` 边界转换为格子尺寸 `dx`、首个格子中心 `bx` 和格子数 `nx`；`models.py` 和 `data.py` 中所有涉及 BEV 网格的地方都走这个函数。此外还有 lidar 加载、`ego_to_cam`/`cam_to_ego`、`img_transform`（返回与之匹配的 `post_rot`/`post_trans`）、归一化、`SimpleLoss`（带 `pos_weight` 的 `BCEWithLogitsLoss`）、`get_batch_iou`（输入是**未过 sigmoid** 的 logits，`logits > 0` 等价于 `sigmoid > 0.5`），以及 nuScenes 地图绘制辅助函数。

**`explore.py` —— 评估与可视化。** 每个公开函数都各自重复声明了一整套 `grid_conf`/`data_aug_conf` 默认值，与 `train.py` 形成重复。修改默认值时必须同时改这五处（`train`、`lidar_check`、`cumsum_check`、`eval_model_iou`、`viz_model_preds`），否则评估/可视化构建出的模型会与 checkpoint 不匹配。

### 网格与增强配置
`grid_conf` 使用 `[min, max, step]` 三元组。默认值：`xbound`/`ybound` 为 `[-50, 50, 0.5]` → 200×200 个 0.5 m 的 BEV 格子；`zbound [-10, 10, 20]` → 只有 1 个 Z bin；`dbound [4, 45, 1]` → 41 个深度 bin。边界必须能被步长整除，因为 `nx` 是用浮点除法计算后再转成 `LongTensor` 的。

`data_aug_conf`：`H`/`W` 为原始图像 900/1600，`final_dim` 为网络输入 (128, 352)，`resize_lim`/`bot_pct_lim`/`rot_lim`（单位为度）在训练时对每个相机采样，`rand_flip`，`cams`（nuScenes 的 6 个相机），`Ncams`。

## 陷阱
- **CPU 模式在两处是坏的。** `train` 中写的是 `SimpleLoss(pos_weight).cuda(gpuid)`，`eval_model_iou` 中写的是 `SimpleLoss(1.0).cuda(gpuid)`；尽管它们上方已经做了 CPU 的设备选择，`gpuid < 0` 时依然会报错。正确写法是 `.to(device)`。
- `eval_model_iou` 把 `pos_weight` 硬编码为 `1.0`，而训练用的是 `2.13`，因此评估打印出的 loss 并不是训练目标，不能和训练日志里的 loss 对比。评估只看 `iou`。
- **Checkpoint 与配置强耦合。** `torch.load` 恢复的是裸 `state_dict`，所以 `grid_conf`、`final_dim`、`dbound`、`Ncams` 必须与训练时一致，否则形状/几何会静默地发生偏差。
- **BEV 网格的轴序在多处被转置。** `voxel_pooling` 填充的是 `(Z, X, Y)`，其中 X 是第一个空间轴；`viz_model_preds` 画图用 `[..., X, Y]` 并设 `xlim((200,0))`、`ylim((0,200))`（车头朝上、左侧在左），`lidar_check` 则用 `.T` + `origin='lower'`（车头朝右、左侧朝上）；`binimgs` 通过交换角点（`pts[:, [1, 0]] = pts[:, [0, 1]]`）来对齐。新增 BEV 相关代码时请先确认约定。
- **形状相同不等于数据正确。** 例如把 `(D*H*W, 3)` 的点写成 `.view(3, -1)` 而不是 `.view(-1, 3).T`，形状没变但把不同点的 x/y/z 混在了一起；`(B,1) - (B,)` 会广播成 `(B,B)`。这类错误不报错，只能靠 `geometry_check` 这类数值对照发现。
- 当前工作树是 git 仓库（分支 `qiuql/dev`），`*.jpg`、`*.png`、`*.pt`、`runs/` 已被 `.gitignore` 忽略。

## 代码风格与命名约定
使用 4 空格缩进，并保持现有布局。遵循 PEP 8 命名规范：函数与变量用 `snake_case`，类名用 `PascalCase`，仅常量使用 `UPPER`。优先使用单引号。模块名使用 `lower_snake_case`。栅格相关的物理常量使用 `xbound`、`ybound`、`dbound` 等命名。`src/*.py` 与 `main.py` 每个文件都保留 NVIDIA 版权 docstring 头，请勿删除。仓库未配置格式化或 lint 工具，因此请与既有代码保持一致。

新增注释一律用中文；上游原有的英文注释保留不动；标识符（变量、函数、类名）保持英文。

## 测试指南
本项目没有自动化测试框架。请使用内置检查来验证改动：

- `lidar_check`：确认相机/LiDAR 几何关系、`get_geometry` 和标签栅格化是否正确。
- `cumsum_check`：验证 `voxel_pooling`/`QuickCumsum` 的自定义反向传播。修改这两处之后务必重跑。
- `python -m src.z_test geometry_check mini --dataroot=$DATA`：验证 `get_geometry` 与 `ego_to_cam` 是否互为逆运算。训练集和验证集都会测（训练集包含随机翻转与旋转），最大误差应在 1e-4 像素量级。

在报告训练或模型改动前，请先在 `mini` 划分上确认评估 IoU。

## 提交与拉取请求指南
Git 历史使用简短、首字母大写、命令式/句子式的主题（例如 `fix requirements`、`clarify commands in readme`）。请保持提交聚焦并说明意图。PR 应说明问题、概述改动、列出用于验证的命令，并关联相关 issue。对于 BEV 输出相关的改动，请附上改动前后的可视化结果。

## 安全与配置提示
请勿提交数据集、模型权重（`*.pt`）、地图或运行输出；`.gitignore` 已排除 `runs/`、`*.pt`、`*.json` 和 `*.csv`。请通过命令行参数配置数据集和日志路径，而不是在代码中硬编码本地路径。
