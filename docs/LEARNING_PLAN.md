# Lift-Splat-Shoot 学习计划（求职向）

> **起点**：已掌握 PyTorch，理解相机内参和外参。
> **目标**：求职 BEV 感知或自动驾驶相关岗位。
> **周期**：8 周，每周约 15–20 小时。时间更紧的话，可以把第 5–6 周的两个项目压缩成一个。

## 学完要达到什么程度

1. **能讲清楚**：不看代码也能画出 LSS 的完整数据流，说出每一步的张量形状，手推出像素到自车坐标的变换公式。
2. **能改代码**：在这个仓库上做出至少一个有量化结果的改进，作为简历上的项目。
3. **知道上下游**：能说出 LSS 和 BEVDet、BEVDepth、BEVFusion、BEVFormer 等后续方法的关系与优劣。
4. **能过面试**：答得出附录 A 的高频问题，能手写附录 B 的代码题。

## 求职方向分叉

前 4 周所有人都一样，第 5–6 周按目标岗位二选一（时间够的话两条都做）：

| 方向 | 适合的岗位 | 第 5–6 周做什么 |
|---|---|---|
| **A. 感知算法** | BEV 感知、3D 检测、Occupancy 算法工程师 | 深度监督、多任务分割、时序融合 |
| **B. 部署工程** | 模型部署、推理加速、自动驾驶系统工程师 | BEV Pooling 加速、ONNX/TensorRT 部署 |

---

## 本机环境速查

```bash
conda activate lss
cd ~/repos/lift-splat-shoot-master

# 常用路径
DATA=/home/qianli/data/nuscenes           # --dataroot
MAPS=/home/qianli/data/nuscenes/mini      # --map_folder（里面有 maps/expansion）
```

- 本机只有一张 RTX 3080（10 GB），所有命令都要加 **`--gpuid=0`**。默认值是 1，不加会报错。
- 目前只有 mini 数据集（10 个场景，404 个样本）。trainval 的下载方法见第 4 周。
- 修改 `grid_conf` 或 `data_aug_conf` 的默认值时，`train.py` 和 `explore.py` 里的 5 个函数都要同步改，否则模型和 checkpoint 对不上。详见仓库根目录的 `CLAUDE.md`。
- `docs/lift-splat-shoot_architecture_report.html` 是仓库的架构图解，可以配合阅读。

---

## 8 周总览

| 周 | 主题 | 产出 |
|---|---|---|
| 1 | 论文 + 跑通 | 用自己的话写一页论文笔记；跑通 4 个入口 |
| 2 | 数据流与几何 | 手推坐标变换公式；写一个几何一致性测试 |
| 3 | 模型精读 | 完整的张量形状表；自己实现一版 voxel pooling |
| 4 | 训练、评估与消融 | 一张消融实验表；修复 CPU 模式的 bug |
| 5–6 | 简历项目（A 或 B） | 一个带量化结果的改进项目 |
| 7 | 横向扩展 | 主流 BEV 方法对比表；读一个工业级代码库 |
| 8 | 面试冲刺 | 面试题过关；能流畅讲 3 分钟项目 |

---

## 第 1 周：论文 + 跑通

**目标**：知道 LSS 在解决什么问题，亲眼看到它的输入和输出。

- [ ] 看作者的 [10 分钟讲解视频](https://youtu.be/oL5ISk6BnDE)
- [x] 精读 [论文](https://arxiv.org/abs/2008.05711)，重点读第 3 节（方法）和第 5 节（实验），可以跳过 Shoot（规划）部分的细节
- [x] 跑 `lidar_check`，打开生成的 jpg，确认激光点能和图像里的物体对上：
  ```bash
  python main.py lidar_check mini --dataroot=$DATA
  ```
- [x] 下载 README 里的预训练模型（Google Drive 打不开的话，先跳过这一项）
- [x] 跑评估和可视化：
  ```bash
  python main.py eval_model_iou mini --modelf=模型.pt --dataroot=$DATA --gpuid=0
  python main.py viz_model_preds mini --modelf=模型.pt --dataroot=$DATA --map_folder=$MAPS --gpuid=0
  ```
- [x] 跑 `cumsum_check`，确认两行输出一致：
  ```bash
  python main.py cumsum_check mini --dataroot=$DATA --gpuid=0
  ```

**本周产出**：一页笔记，回答以下问题。

- 为什么自动驾驶要用 BEV 表示？和直接在图像上做检测相比，好处是什么？
自动驾驶要做的决策都发生在地面上，BEV（鸟瞰图）就是"地面坐标系下的地图"。用 BEV 表示，可以把多个相机、多种传感器和下游规划统一到同一个坐标系里。
- "Lift"、"Splat"、"Shoot" 分别是什么意思？
- 论文和哪些 baseline 做了对比？结论是什么？
![alt text](image.png)
---

## 第 2 周：数据流与几何（`src/data.py`、`src/tools.py`）

**目标**：搞清楚一个 batch 里每个张量的含义，能手推坐标变换。

### 阅读顺序

1. `src/data.py:244` `compile_data`：如何划分训练集和验证集、如何选择解析器
2. `src/data.py:230` `SegmentationData.__getitem__`：返回的 7 元组
3. `src/data.py:195` `choose_cams`：训练时从 6 个相机中随机选 5 个，想一想为什么这么做
答复：正则化，相当于相机级别的 dropout（主要原因）;实际部署时要能扛住相机失效;顺带省了显存和计算
4. `src/data.py:121` `get_image_data`：读图，取 `rots`、`trans`、`intrins`
5. `src/data.py:96` `sample_augmentation` + `src/tools.py:141` `img_transform`：图像增强，以及 `post_rot`、`post_tran` 是怎么累积出来的
6. `src/data.py:171` `get_binimg`：车辆框如何栅格化成 200×200 的标签。注意 `pts[:, [1, 0]] = pts[:, [0, 1]]` 这一行交换了坐标轴
7. `src/tools.py:199` `gen_dx_bx`：`dx`、`bx`、`nx` 分别是什么

### 必须手推的公式

视锥中的一个点 `(u', v', d)` 是增强后图像上的像素坐标加上深度，它变换到自车坐标系的步骤如下：

```
1. 撤销图像增强：       [u, v, d]ᵀ = post_rot⁻¹ · ([u', v', d]ᵀ − post_tran)
2. 反投影到相机坐标系： p_cam = K⁻¹ · [u·d, v·d, d]ᵀ
3. 相机坐标系 → 自车：   p_ego = R · p_cam + t
```

对照 `src/models.py:185` 的 `get_geometry` 逐行核对。想清楚以下两个问题：

- 为什么要先乘以 d 再乘 K⁻¹？
- `post_rot` 是 3×3 矩阵，但只有左上 2×2 是有意义的，为什么？
  答：因为数据增强只是在图像平面上搬动像素，不会改变任何点的深度

### 动手练习

- [x] 在 Python 中取一个 batch，打印 7 个张量的 shape，并用 matplotlib 画出 `imgs`（需要反归一化，可以用 `src/tools.py` 中的 `denormalize_img`）和 `binimgs`
- [x] **写一个几何一致性测试**：
  1. 取 `get_geometry` 输出的某个自车坐标点
  2. 用 `src/tools.py:80` 的 `ego_to_cam` 投影回原始图像
  3. 再施加 `post_rot` 和 `post_tran`
  4. 应该得到原来的 `(u', v')` 和深度 `d`

  这个测试同时验证了你的理解和代码的正确性，面试时也可以拿来讲。

---

## 第 3 周：模型精读（`src/models.py`）

**目标**：能脱离代码画出整个网络，说出每一步的形状。

### 张量形状表

默认配置：B=4，N=5，输入 128×352，下采样 16 倍，D=41，C=64。

| 步骤 | 代码位置 | 形状 | 说明 |
|---|---|---|---|
| 输入 | — | 4×5×3×128×352 | |
| 视锥模板 | `create_frustum` `:153` | 41×8×22×3 | 每个元素是 `(u', v', d)` |
| 骨干网络 | `CamEncode.get_eff_depth` `:61` | 20×512×8×22 | EfficientNet-B0，融合 1/32 层（320 通道）与 1/16 层（112 通道） |
| 深度和特征头 | `depthnet` | 20×105×8×22 | 105 = D + C = 41 + 64 |
| **Lift** 外积 | `get_depth_feat` `:51` | 20×64×41×8×22 | `softmax(depth) ⊗ context` |
| 重排维度 | `get_cam_feats` `:229` | 4×5×41×8×22×64 | |
| **Lift** 几何 | `get_geometry` `:185` | 4×5×41×8×22×3 | 每个特征点的自车坐标 |
| **Splat** | `voxel_pooling` `:241` | 4×64×200×200 | Z 轴只有 1 个格子，直接合并进通道维 |
| BEV 编码 | `BevEncode` `:90` | 4×1×200×200 | ResNet18 前 3 层加 U-Net 式上采样 |

### `voxel_pooling` 逐步拆解（面试重点）

1. **量化**：用 `(p − (bx − dx/2)) / dx` 把连续坐标转成格子下标
2. **加 batch 下标**：不同样本的点不能混进同一个格子
3. **过滤**：丢掉落在网格范围外的点
4. **排序**：用 rank 编码 `(x, y, z, b)`，`argsort` 之后同一个格子的点连续排列
5. **cumsum 技巧**：先对整个序列做前缀和，然后只保留每段的最后一个元素，再与前一段相减，就得到每段的和。时间复杂度 O(N)，不需要循环
6. **写回网格**：把每段的和填进 `B×C×Z×X×Y` 的网格

### `QuickCumsum` 的反向传播（`src/tools.py:218`）

求和池化的梯度很简单：同一个格子内每个点的梯度，都等于该格子输出的梯度。`backward` 中的 `back = cumsum(kept) − 1`（只对每段末尾的点减 1）就是在给每个点找到它所属段的下标。值得自己推一遍，体会为什么自定义 backward 比让 autograd 自动求导更快、更省显存。

### 思考题

<details><summary>1. 每个 batch 有多少个视锥点？显存大头在哪里？</summary>

4×5×41×8×22 = 144,320 个点，每个点有 64 维特征。深度 bin 数 D 和特征图分辨率会直接按乘法放大这个数字，这也是后续工作热衷于优化 BEV Pooling 的原因。

</details>

<details><summary>2. voxel_pooling 量化时用了 <code>.long()</code>，有什么隐患？</summary>

`.long()` 是向零取整，不是向下取整。下标落在 (−1, 0) 之间的点会被取整成 0，于是能通过 `>= 0` 的过滤条件，被错误地计入边缘格子。以 x 轴为例，[−50.5, −50) 米之间的点会被算进第 0 格。改成先 `floor()` 再 `.long()` 就能修复。影响虽然很小，但面试中可以作为"读源码发现的细节问题"来讲。

</details>

<details><summary>3. 为什么网络只用 BEV 分割做监督，却能学会深度？</summary>

深度分布决定了特征落在 BEV 的哪个格子。如果深度预测错了，特征就会落到错误的位置，导致分割损失变大，梯度会把深度往正确的方向推。这就是"隐式深度监督"。它的缺点是信号弱、学得慢，BEVDepth 正是针对这一点做的改进。

</details>

### 动手练习

- [ ] 用随机输入构造模型（不需要数据），在每一步打印 shape，核对上表
- [ ] **自己实现一版 voxel pooling**：用 `torch.Tensor.index_add_` 或 `scatter_add_` 代替排序加 cumsum。与原实现对比数值是否一致、哪个更快。面试手撕代码时，这个写法最容易写对
- [ ] 打开 `use_quickcumsum = False`，对比两种实现的显存和速度

---

## 第 4 周：训练、评估与消融（`src/train.py`）

**目标**：完整走通训练流程，产出第一张实验表。

### 读 `src/train.py`（共 120 行）

- 优化器：Adam，lr=1e-3，weight_decay=1e-7，梯度裁剪阈值 5.0
- 损失：带 `pos_weight=2.13` 的 BCE。想一想为什么需要 `pos_weight`（提示：BEV 上车辆像素占比很小）
- 指标：`get_batch_iou` 以 `logits > 0` 作为阈值，等价于 sigmoid 后大于 0.5。验证集 IoU 的计算方式是：先把整个数据集的交集和并集各自累加，最后相除。这和逐张算 IoU 再取平均是不同的
- 注意：`eval_model_iou` 里把 `pos_weight` 写死成了 1.0，所以评估打印的 loss 和训练时的 loss 不可直接比较

### 任务

- [ ] 在 mini 上训练，用 TensorBoard 观察曲线：
  ```bash
  python main.py train mini --dataroot=$DATA --logdir=./runs/baseline --gpuid=0
  tensorboard --logdir=./runs --bind_all
  ```
  mini 每 1000 步验证并保存一次模型；`nepochs` 默认是 10000，看够了就按 Ctrl+C 停止。你会看到训练 IoU 很高、验证 IoU 很低，这是过拟合，属于正常现象。
- [ ] 从 TensorBoard 的 `train/step_time` 估算：在 trainval 上训练到收敛需要多长时间
- [ ] **修复 CPU 模式的 bug**：`train.py:72` 和 `eval_model_iou` 里把 `.cuda(gpuid)` 写死了，改成 `.to(device)`。这是一个很好的"读懂代码再改"的小练习
- [ ] **（建议本周开始，后台进行）下载 trainval 关键帧**：只需要 `v1.0-trainval01~10_keyframes.tgz`（约 42 GiB）和 `v1.0-trainval_meta.tgz`，不需要约 285 GiB 的 blobs。本机剩余空间约 250 GB，放得下。mini 只能用来调试，简历上要写的数字最好来自 trainval

### 消融实验（产出一张表）

先在 mini 上调通流程，有了 trainval 后再出正式数字：

| 实验 | 修改方式 | 要回答的问题 |
|---|---|---|
| 深度 bin 数 | `--dbound="[4.0,45.0,2.0]"` | 深度分辨率对 IoU 的影响有多大？ |
| 训练相机数 | `--ncams=6` 与 `--ncams=5` 对比 | 随机丢相机是否起到了正则化作用？ |
| 数据增强 | 把 `resize_lim`、`rot_lim` 设为固定值，`--rand_flip=False` | 增强有多重要？ |
| 输入分辨率 | `--final_dim="(256,704)"` | 分辨率、显存和精度之间如何取舍？ |

---

## 第 5–6 周：简历项目（选一个方向）

> 原则：**一个做深的项目胜过三个浅尝辄止的项目**。每个项目都要有基线对比、量化结果和可视化。

### 方向 A：感知算法

**A1. 激光雷达深度监督（推荐，这正是 BEVDepth 的核心思想）**

- 动机：LSS 的深度只靠 BEV 分割损失间接学习，信号弱
- 做法：
  1. 在 `SegmentationData` 中额外返回激光雷达点。只用关键帧（`nsweeps=1`），trainval 的 keyframes 分片已经包含这部分数据
  2. 对每个相机，用 `ego_to_cam` 加 `post_rot`/`post_trans` 把点投影到 8×22 的特征图上，生成每个像素的 one-hot 深度 bin 标签
  3. 修改 `CamEncode.forward`，让它同时返回 `depth`（目前在 `:84` 被丢弃了）
  4. 增加一个深度分类损失（BCE 或 CE，只在有激光点的像素上计算），与分割损失加权求和
- 产出：IoU 对比、深度图可视化对比、损失权重的消融
- 面试亮点：能讲清楚"隐式深度 vs 显式深度监督"，以及为什么激光点稀疏仍然有效

**A2. 多类别与地图分割**

- 把 `outC=1` 改成多通道：车辆、行人，以及从 map expansion 栅格化得到的可行驶区域（`src/tools.py:347` 的 `get_local_map` 可以参考）
- 难点：各类别的不平衡（逐类设置 `pos_weight` 或改用 focal loss）、多任务损失的权重
- 产出：逐类 IoU 表

**A3. 时序融合（较难，BEVDet4D 的思路）**

- 用 ego pose 把上一帧的 BEV 特征对齐到当前帧（`grid_sample`），与当前帧特征拼接后再送入 `BevEncode`
- 产出：IoU 提升，以及在遮挡场景下的可视化对比

### 方向 B：部署工程

**B1. BEV Pooling 加速**

- 观察：相机内外参固定时，`get_geometry` 的结果、量化下标、排序结果都与输入图像无关，可以**预计算一次**（BEVPoolv2 的思路）
- 做法：先实现"预计算下标 + `index_add_`"的版本，再尝试写一个 CUDA kernel（用 `torch.utils.cpp_extension`）
- 产出：原版、预计算版、CUDA 版三者在不同 D 和分辨率下的耗时对比表

**B2. 导出 ONNX 与 TensorRT 部署**

- 难点：`argsort`、`cumsum`、`torch.inverse`、布尔索引对导出都不友好
- 思路：结合 B1，把几何部分外置为预计算的常量下标，网络中只剩 CNN 和 gather/scatter；必要时为 BEV Pooling 写一个 TensorRT 插件
- 产出：PyTorch、TRT FP32、TRT FP16 三者的延迟和 IoU 对比

**B3. 训练工程（可以作为 B1 或 B2 的补充）**

- 混合精度训练（`torch.autocast`）、DataLoader 性能分析、`torch.compile`
- 产出：训练吞吐对比

### 简历写法模板

> **基于 Lift-Splat-Shoot 的多相机 BEV 感知**
> - 精读并复现 LSS（nuScenes 车辆 BEV 分割，IoU xx.x），梳理视锥构建、坐标变换和 Voxel Pooling 全流程
> - 引入激光雷达深度监督，将车辆 IoU 从 xx.x 提升到 xx.x；消融了深度损失权重和深度 bin 数
> - （方向 B）将 BEV Pooling 改为预计算下标加自定义 CUDA kernel，耗时从 x ms 降到 x ms；导出 TensorRT FP16，端到端延迟 x ms

每个数字都要能讲出怎么来的，面试官一定会追问。

---

## 第 7 周：横向扩展到主流 BEV 方法

**目标**：把 LSS 放进整个技术谱系里理解。面试官很少只问 LSS 本身。

| 方法 | 和 LSS 的关系 | 重点关注 |
|---|---|---|
| **BEVDet** | LSS 加 BEV 空间数据增强加 3D 检测头 | BEV 增强为什么要同步变换标签 |
| **BEVDet4D** | 加入时序 | 帧间对齐方式、速度估计 |
| **BEVDepth** | 加入激光雷达深度监督和相机参数感知 | 对应项目 A1 |
| **BEVFusion**（MIT） | 以 LSS 作为相机分支，与激光雷达在 BEV 上融合 | 高效 BEV Pooling（对应项目 B1） |
| **BEVFormer** | **反向投影**：BEV query 通过 deformable attention 去图像上采样 | 与 LSS 前向投影的优劣对比（必考） |
| **FB-BEV / FB-OCC** | 前向投影与反向投影结合 | 如何互补 |
| **Occupancy**（如 FlashOcc） | 从 BEV 平面扩展到 3D 占据栅格 | Z 轴不再压扁 |
| **UniAD / VAD** | 端到端感知、预测、规划 | 延续了 LSS 中 "Shoot" 的思路 |

- [ ] 每篇论文写一段话总结：解决了 LSS 的什么问题？代价是什么？
- [ ] **读一个工业风格的代码库**，推荐 BEVDet，或 mmdetection3d 中的 BEV 相关实现。重点看它的 `view_transformer` 和这个仓库有什么不同（访问 GitHub 时需要走 gh-proxy）

---

## 第 8 周：面试冲刺

- [ ] 附录 A 的每道题都能脱稿讲 1–2 分钟
- [ ] 附录 B 的代码题都能在白板或纸上写出来
- [ ] 准备一段 3 分钟的项目介绍：背景、做法、结果、遇到的坑，并录音自己听一遍
- [ ] 准备追问：为什么选这个改进？还能怎么改？上车部署会遇到什么问题？

---

## 附录 A：高频面试题（附答题要点）

<details><summary>1. 为什么要做 BEV 感知？</summary>

- 多相机、多传感器可以统一到自车坐标系下融合，避免透视图逐相机后处理和跨相机拼接
- 下游的预测、规划本身就在 BEV（俯视平面）上进行
- BEV 上物体尺度不随距离变化，没有透视图中近大远小的问题

</details>

<details><summary>2. Lift 是怎么做的？为什么预测深度分布而不是单一深度值？</summary>

- 对特征图的每个像素，预测 D 个深度 bin 的 softmax 概率 α 和 C 维上下文特征 c，lift 后的特征是外积 α ⊗ c
- 单一深度值一旦预测错，特征就落到错误的位置，无法挽回；分布可以表达不确定性，拿不准时把特征分摊到多个深度上
- 整个过程可微，只用 BEV 标签就能端到端训练

</details>

<details><summary>3. 推导视锥点到自车坐标的变换</summary>

见第 2 周的三步公式。关键点：要先撤销图像增强；反投影前要先乘深度 d，即齐次坐标；`rots` 和 `trans` 来自 `calibrated_sensor`，表示相机到自车的变换。

</details>

<details><summary>4. Splat 和 voxel pooling 是怎么实现的？怎么加速？</summary>

- 原版：量化 → 过滤 → 按 rank 排序 → cumsum 技巧做分段求和 → 写回网格；自定义 backward 把每个格子的梯度直接复制给格内所有点
- 加速：BEVPoolv2 在相机参数固定时预计算下标和区间，去掉每次都要做的排序；BEVFusion 用 CUDA kernel 让每个线程处理一个区间（interval reduction），避免前缀和带来的数值问题和额外显存

</details>

<details><summary>5. LSS 有哪些缺点？后续工作如何改进？</summary>

| 缺点 | 改进 |
|---|---|
| 深度没有显式监督，估得不准 | BEVDepth：激光雷达深度监督 |
| 视锥点多，Pooling 慢、占显存 | BEVPoolv2、BEVFusion 的高效 Pooling |
| 远处视锥发散，BEV 特征稀疏 | 反向投影（BEVFormer），或前后向结合（FB-BEV） |
| Z 轴被压扁，丢失高度信息 | Occupancy 方法 |
| 没有时序信息 | BEVDet4D、BEVFormer 的时序模块 |

</details>

<details><summary>6. 前向投影（LSS）和反向投影（BEVFormer）各有什么优劣？</summary>

- **前向投影**：深度显式可解释；配合 BEVPool 计算高效、容易部署。缺点是依赖深度质量，远处 BEV 稀疏
- **反向投影**：每个 BEV 格子都主动采样，特征稠密。缺点是同一条射线上的多个 3D 点会采到同一个像素特征，存在深度歧义；deformable attention 部署难度更高

</details>

<details><summary>7. 做图像数据增强时，几何上要注意什么？</summary>

- 图像空间的缩放、裁剪、翻转、旋转都会改变像素与 3D 射线的对应关系，因此必须记录 `post_rot` 和 `post_trans`，并在 `get_geometry` 中撤销
- BEV 空间增强（BEVDet）要同时变换 BEV 特征和 GT
- 本仓库对每个相机独立采样增强参数

</details>

<details><summary>8. 前景和背景不平衡怎么处理？IoU 怎么算？</summary>

- 本仓库用的是 BCE 加 `pos_weight=2.13`；其他做法有 focal loss、dice loss，或二者组合
- IoU 是在整个数据集上累加交集和并集后再相除，阈值是 logits 大于 0

</details>

<details><summary>9. 如何把 LSS 部署到车端？</summary>

- 相机参数固定时预计算几何下标，去掉 `argsort`、`inverse` 等不利于导出的算子
- BEV Pooling 写成 TensorRT 插件
- 使用 FP16 或 INT8 量化（注意深度 softmax 和 BEV Pooling 的精度）
- 在深度 bin 数、特征通道数和输入分辨率之间做精度与延迟的取舍

</details>

<details><summary>10. 论文中对相机标定误差和相机缺失做了哪些鲁棒性分析？</summary>

训练时随机丢弃相机（本仓库 `Ncams=5`）起到了正则化作用，并提升了测试时缺少相机情况下的表现；论文还分析了外参噪声对性能的影响。细节以论文中关于鲁棒性的章节为准，面试前务必回顾原文。

</details>

## 附录 B：手撕代码题

1. 给定 K、R、t、post_rot、post_trans，写出视锥点到自车坐标的变换（参考 `get_geometry`）
2. 给定 N 个带特征的 3D 点，写出 BEV 求和池化（用 `index_add_` 的版本最稳妥）
3. 写出 cumsum 分段求和技巧（参考 `src/tools.py:207` 的 `cumsum_trick`）
4. 把一个 3D 框的底面栅格化到 BEV 网格上（参考 `get_binimg`）
5. 计算二值分割的 IoU（参考 `get_batch_iou`）

## 附录 C：资源

- 论文：https://arxiv.org/abs/2008.05711
- 项目主页：https://nv-tlabs.github.io/lift-splat-shoot/
- nuScenes devkit 教程：`nuscenes-devkit` 包中的 `tutorials/nuscenes_tutorial.ipynb`，可以帮助理解 sample、sample_data、calibrated_sensor 之间的关系
- 本仓库的注意事项与已知陷阱：根目录的 `CLAUDE.md`
