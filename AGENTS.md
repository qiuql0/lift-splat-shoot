# 仓库指南

## 项目结构与模块组织
本仓库是 Lift-Splat-Shoot（ECCV 2020）的 PyTorch 参考实现。所有模型代码都位于 `src/`：

- `main.py`：命令行入口，通过 `fire.Fire` 暴露各个子命令。
- `src/data.py`：nuScenes 数据集类（`NuscData`、`SegmentationData`、`VizData`）以及 `compile_data`。
- `src/models.py`：网络定义（`CamEncode`、`BevEncode`、`LiftSplatShoot`）以及 `compile_model`。
- `src/tools.py`：几何变换、cumsum 技巧、损失函数，以及 IoU/绘图辅助函数。
- `src/train.py`：训练循环。`src/explore.py`：评估、可视化与健全性检查。
- `imgs/`：README 中使用的 GIF 动图。本项目没有测试目录，也未附带测试套件。

## 构建、测试与开发命令
先安装依赖（仓库未提交 `requirements.txt`；本机已验证的完整版本组合见 `requirements-win.txt`，其中 numpy / matplotlib / opencv 三项有硬性版本约束）：

```
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```

常用命令（将 `NUSCENES_ROOT` 设为你的数据集路径）：

```
python main.py lidar_check mini --dataroot=$NUSCENES_ROOT   # 校验内参/外参
python main.py eval_model_iou mini --modelf=MODEL.pt --dataroot=$NUSCENES_ROOT
python main.py viz_model_preds trainval --modelf=MODEL.pt --dataroot=$NUSCENES_ROOT --map_folder=MAPS
python main.py train mini --dataroot=$NUSCENES_ROOT --logdir=./runs --gpuid=0
```

**数据集目录结构（易踩坑）**：`compile_data` 内部执行 `os.path.join(dataroot, version)`，因此 `--dataroot` 要指向包含 `<version>` 子目录的**父目录**，而不是数据集本身所在的目录：

```
D:\data\                 <- --dataroot 传这一层
  └─ mini\               <- version 参数（mini / trainval）
       ├─ v1.0-mini\*.json
       ├─ samples\  sweeps\
       └─ maps\
```

若报 `AssertionError: Database version not found: ...\mini\v1.0-mini`，就是这个层级放错了。

另外，`viz_model_preds` 还会读取 `<dataroot>\<version>\maps\` 下的矢量地图 JSON，需要额外下载 nuScenes 地图扩展包。

## 代码风格与命名约定
使用 4 空格缩进，并保持现有布局。遵循 PEP 8 命名规范：函数与变量用 `snake_case`，类名用 `PascalCase`，仅常量使用 `UPPER`。优先使用单引号。模块名使用 `lower_snake_case`。栅格相关的物理常量使用 `xbound`、`ybound`、`dbound` 等命名。`src/*.py` 与 `main.py` 每个文件都保留 NVIDIA 版权 docstring 头，请勿删除。仓库未配置格式化或 lint 工具，因此请与既有代码保持一致。

## 测试指南
本项目没有自动化测试框架。请使用内置检查来验证改动：运行 `lidar_check` 确认相机/LiDAR 几何关系，运行 `cumsum_check` 验证 cumsum 技巧。在报告训练或模型改动前，请先在 `mini` 划分上确认评估 IoU。

## 提交与拉取请求指南
Git 历史使用简短、首字母大写、命令式/句子式的主题（例如 `fix requirements`、`clarify commands in readme`）。请保持提交聚焦并说明意图。PR 应说明问题、概述改动、列出用于验证的命令，并关联相关 issue。对于 BEV 输出相关的改动，请附上改动前后的可视化结果。

## 安全与配置提示
请勿提交数据集、模型权重（`*.pt`）、地图或运行输出；`.gitignore` 已排除 `runs/`、`*.pt`、`*.json` 和 `*.csv`。请通过命令行参数配置数据集和日志路径，而不是在代码中硬编码本地路径。
