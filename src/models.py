"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True) # bilinear **只支持 4 维张量 (B,C,H,W) 将H和W放大2倍

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1) # 除了`dim=1`这个通道维度以外，**其余维度 (B,H,W) 必须完全相等* dim=1是channel层
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D # 41
        self.C = C # 64

        # 加载ImageNet 预训练的 EfficientNet-B0 模型。原生设计目标是图像分类任务
        # 会加载权重。from_name()就不会加载权重。
        # EfficientNet 是 Google 2019 年提出的轻量级卷积神经网络，核心思想是**用系统化的复合缩放方法，同时平衡网络深度、宽度和输入分辨率，在精度和计算效率之间取得最优平衡*
        # 复合缩放策略（Compound Scaling）传统网络要么只加深（ResNet）、要么只加宽（WideResNet）、要么只改输入尺寸。EfficientNet 用一个统一的系数 φ 同时缩放三个维度
        # 整个系列从 B0 到 B8 共 9 个标准型。**B0**：基线模型，~5.3M 参数，输入 224×224，速度最快。B1~B7：逐步放大，输入尺寸从 240 到 600 递增。B8：标准系列最大，输入 672×672.
        # 网络具体结构，见最下面。
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0") 

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0) # 这里代码**没有写 bias=False**，默认`bias=True

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x) # x=[B,512,8,22]
        # Depth
        x = self.depthnet(x) # x=[B,105,8,22]

        depth = self.get_depth_dist(x[:, :self.D]) # depth=[B,41,8,22】
        # depth.unsqueeze(1)=[B,1,41,8,22]
        # x[:, self.D:(self.D + self.C)].unsqueeze(2)=[B,64,1,8,22]
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2) # new_x=[B,64,41,8,22]

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        # Swish 是一种平滑的非线性激活函数
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4']) # 320+112-》512
        return x

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)
    
    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape #D=41
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC) # outC=1

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim'] # ogfH=128,ogfW=352, final_dim=(128, 352),
        fH, fW = ogfH // self.downsample, ogfW // self.downsample #  self.downsample=16 fH=8 fW=22

        # dbound=[4.0, 45.0, 1.0], `expand` **不会分配新内存**，只是视图。如果后续 in-place 修改，所有广播出来的位置会一起
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) # ds=[41,8,22] 第一维是等间隔序列，后两维是扩展

        D, _, _ = ds.shape # D=41
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW) # xs=[41,8,22] 第三维是等间隔序列，前两维是扩展
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1) #frustum=[41,8,22,3]
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B*N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)

# ==============================================================
# EfficientNet-B0 完整结构说明（基线模型）
# 输入尺寸：224×224 RGB图像；总参数量：~5.3M
# 整体架构：Stem输入层 → 7个Stage（MBConv堆叠） → Head输出头
# 核心组件：MBConv倒置残差瓶颈 + SE通道注意力，激活函数为Swish
# ==============================================================

# -------------------- 0. Stem 输入层 --------------------
# 操作：3×3卷积(stride=2) + BatchNorm + Swish
# 输入：[B, 3, 224, 224] 原始RGB图像
# 输出：[B, 32, 112, 112]，完成第一次2倍下采样，提取底层边缘纹理特征

# -------------------- 1. Stage 1 --------------------
# 配置：MBConv1，卷积核3×3，带SE注意力，无下采样
# 模块数量：1个
# 输入通道：32 → 输出通道：16
# 扩展倍率：1（MBConv1 = 不做通道升维）
# 输出分辨率：112×112（对应2倍下采样层级）

# -------------------- 2. Stage 2 --------------------
# 配置：MBConv6，卷积核3×3，带SE注意力，stride=2下采样
# 模块数量：2个（仅第1个块下采样，第2个保持分辨率）
# 输入通道：16 → 输出通道：24
# 扩展倍率：6（MBConv6 = 通道先放大6倍再卷积）
# 输出分辨率：56×56（对应4倍下采样层级）

# -------------------- 3. Stage 3 --------------------
# 配置：MBConv6，卷积核5×5，带SE注意力，stride=2下采样
# 模块数量：2个（仅第1个块下采样）
# 输入通道：24 → 输出通道：40
# 扩展倍率：6
# 输出分辨率：28×28（对应8倍下采样层级）

# -------------------- 4. Stage 4 --------------------
# 配置：MBConv6，卷积核3×3，带SE注意力，stride=2下采样
# 模块数量：3个（仅第1个块下采样）
# 输入通道：40 → 输出通道：80
# 扩展倍率：6
# 输出分辨率：14×14（对应16倍下采样层级）

# -------------------- 5. Stage 5 --------------------
# 配置：MBConv6，卷积核5×5，带SE注意力，无下采样
# 模块数量：3个
# 输入通道：80 → 输出通道：112
# 扩展倍率：6
# 输出分辨率：14×14（对应16倍下采样层级）

# -------------------- 6. Stage 6 --------------------
# 配置：MBConv6，卷积核5×5，带SE注意力，stride=2下采样
# 模块数量：4个（仅第1个块下采样）
# 输入通道：112 → 输出通道：192
# 扩展倍率：6
# 输出分辨率：7×7（对应32倍下采样层级）

# -------------------- 7. Stage 7 --------------------
# 配置：MBConv6，卷积核3×3，带SE注意力，无下采样
# 模块数量：1个
# 输入通道：192 → 输出通道：320
# 扩展倍率：6
# 输出分辨率：7×7（对应32倍下采样层级）

# -------------------- Head 输出头（原生分类任务） --------------------
# 1. 1×1卷积：通道320 → 1280，配合BatchNorm + Swish
# 2. 全局平均池化：[B, 1280, 7, 7] → [B, 1280, 1, 1]
# 3. 全连接层：1280维 → 1000维（ImageNet分类类别数）
# 注：做分割/深度估计等下游任务时，通常丢弃全连接层，仅用卷积主干提取多尺度特征

# -------------------- 核心模块补充说明 --------------------
# MBConv：倒置残差瓶颈卷积，完整流程：1×1卷积升维 → 深度可分离卷积 → SE注意力 → 1×1卷积降维 → 残差连接
# MBConv1/6：后缀数字代表通道扩展倍率，1=不升维，6=卷积前先将通道数放大6倍
# SE（Squeeze-and-Excitation）：通道注意力机制，全局池化后学习各通道权重，自动强化有效特征、抑制无效特征
# 下采样规则：每个Stage仅第一个Block可使用stride=2完成2倍下采样，后续Block保持分辨率不变