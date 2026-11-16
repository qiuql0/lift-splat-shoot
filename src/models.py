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

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

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
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320+112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
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
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

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
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        """构造视锥模板：特征图每个像素位置 × D 个候选深度，组成 (u', v', d) 网格。

        只在 __init__ 中调用一次，与样本无关；数据增强和相机标定的差异由 get_geometry 处理。
        u', v': 增强后网络输入图（final_dim，默认 352x128）上的像素坐标，不是特征图下标
        d:      沿光轴方向的深度，即相机坐标系的 z（米），不是到相机的直线距离；
                因此同一个 d 的点组成一个垂直于光轴的平面

        Returns D x H/downsample x W/downsample x 3，默认 41 x 8 x 22 x 3
        """
        # make grid in image plane
        # 网络输入图尺寸 (128, 352)
        ogfH, ogfW = self.data_aug_conf['final_dim']
        # 特征图尺寸 (8, 22)，与 CamEncode 输出的 1/16 分辨率一致
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        # 深度：arange(4, 45, 1) -> 4, 5, ..., 44，共 D=41 个（不含 45，取的是每个 bin 的左端点）
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        # D 同时决定了 CamEncode.depthnet 的输出通道数 D + C，改 dbound 会改变网络结构
        D, _, _ = ds.shape
        # u：在 0 ~ 351 上均匀取 22 个点，步长 351/21 ≈ 16.71。首尾贴着图像边缘，
        #    并不精确落在每个 16x16 块的中心（16j + 7.5），越靠边偏差越大，最多约 7.5 px
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        # v：在 0 ~ 127 上均匀取 8 个点，情况同上
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        # 最后一维是 (u', v', d)
        frustum = torch.stack((xs, ys, ds), -1)
        # 用 Parameter 是为了随 model.to(device) 移到 GPU；它不参与训练，
        # 但会被存进 state_dict，加载 checkpoint 时会被覆盖成训练时的视锥
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.

        N 为相机数。所有入参都来自 NuscData.get_image_data（src/data.py）。
        rots:       B x N x 3 x 3  相机坐标系 -> 自车坐标系的旋转（来自 calibrated_sensor）
        trans:      B x N x 3      相机坐标系 -> 自车坐标系的平移，即相机光心在自车坐标系中的位置（米）
        intrins:    B x N x 3 x 3  原始图像（未做数据增强）的内参 K
        post_rots:  B x N x 3 x 3  原图像素 -> 增强后像素的线性变换（缩放/翻转/旋转）；
                                   只有左上角 2x2 有实际内容，深度不受影响
        post_trans: B x N x 3      同一变换的平移部分（裁剪偏移等）；
                                   第 3 个分量（深度方向）恒为 0

        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        # 第 1 步：撤销数据增强。增强是 p' = A·p + b（A = post_rots，b = post_trans），
        # 反过来就是 p = A⁻¹·(p' − b)：先减平移，再乘逆矩阵，顺序不能反。
        # 视锥 (D x H x W x 3) 与 view 后的 (B x N x 1 x 1 x 1 x 3) 广播，
        # 每个相机的所有视锥点都减去该相机自己的平移量
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        # unsqueeze(-1) 把每个点变成 3x1 列向量，以便做矩阵乘法；结果为 B x N x D x H x W x 3 x 1
        # matmul 的广播只作用于前面的维度，最后两维按矩阵乘法规则计算
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        # 第 2 步：反投影。针孔模型是 d·(u, v, 1)ᵀ = K·p_cam，
        # 所以先把 (u, v, d) 变成 (u·d, v·d, d)，再乘 K⁻¹ 就得到 p_cam，其 z 分量恰好等于 d
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        # 第 3 步：相机坐标系 -> 自车坐标系，p_ego = R·p_cam + t = R·K⁻¹·(u·d, v·d, d)ᵀ + t。
        # 先把 R 和 K⁻¹ 合并成一个矩阵（每个相机只算一次），这样每个点只需做一次 3x3 矩阵乘法
        combine = rots.matmul(torch.inverse(intrins))
        # squeeze(-1) 去掉列向量的最后一维，变回 B x N x D x H x W x 3
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # 加上相机光心在自车坐标系中的位置
        points += trans.view(B, N, 1, 1, 1, 3)

        # 每个视锥点在自车坐标系下的 (x, y, z)，单位米
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
