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
    """
    上采样融合模块（U-Net解码器风格）。
    
    本模块采用经典的U-Net解码器设计模式，实现上采样与跳跃连接的特征融合。
    虽然Lift-Splat-Shoot整体架构不是标准U-Net，但本模块借鉴了U-Net的核心思想：
    - 通过上采样恢复特征图分辨率
    - 通过跳跃连接融合编码器的高分辨率细节特征
    
    核心功能：
    1. 将低分辨率特征图上采样到高分辨率
    2. 与编码器对应的高分辨率特征进行通道拼接（跳跃连接）
    3. 通过卷积序列处理融合后的特征
    
    Args:
        in_channels (int): 输入通道数（融合后的总通道数，如320+112=432）
        out_channels (int): 输出通道数（如512）
        scale_factor (int): 上采样倍数（默认2，将H和W放大2倍）
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        # 双线性上采样层：将特征图尺寸放大scale_factor倍
        # mode='bilinear'：使用双线性插值，保持图像平滑
        # align_corners=True：对齐输入输出的角落像素，保持坐标一致性
        # mode='bilinear'输入要求：4D张量 (B, C, H, W)，最后两维为高度和宽度
        
        # PyTorch 的 nn.Upsample 对不同 mode 有明确的维度限制：
        # mode	支持维度	对应场景
        # 'bilinear'	4D (N, C, H, W)	2D 图像上采样
        # 'trilinear'	5D (N, C, D, H, W)	3D 体数据上采样
        # 'nearest'	3D/4D/5D	通用最近邻插值
        # 'bicubic'	4D	2D 双三次插值
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        # 卷积序列：处理融合后的特征
        # 结构：Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
        self.conv = nn.Sequential(
            # 第一层卷积：将输入通道映射到输出通道
            # kernel_size=3, padding=1 保持H和W不变（same padding）
            # bias=False：BatchNorm会添加偏移，无需额外偏置
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            # 批归一化：加速训练，提高稳定性
            nn.BatchNorm2d(out_channels),
            # 激活函数：引入非线性
            nn.ReLU(inplace=True),
            # 第二层卷积：进一步提取特征
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        """
        前向传播：实现上采样和特征融合。
        
        Args:
            x1 (torch.Tensor): 来自解码器的低分辨率特征图，形状为 (B, C1, H/2, W/2)
            x2 (torch.Tensor): 来自编码器的高分辨率特征图，形状为 (B, C2, H, W)
            
        Returns:
            torch.Tensor: 融合后的特征图，形状为 (B, out_channels, H, W)
            
        流程：
        1. 上采样：将x1放大到与x2相同的空间尺寸
        2. 拼接：在通道维度上融合x2（高分辨率细节）和x1（上采样特征）
        3. 卷积：通过卷积序列处理融合后的特征
        """
        # 步骤1：上采样x1（低分辨率特征图）到与x2相同的空间尺寸
        # 输入x1形状：(B, C1, H/2, W/2) → 输出形状：(B, C1, H, W)
        x1 = self.up(x1)
        
        # 步骤2：在通道维度（dim=1）拼接x2和上采样后的x1
        # x2形状：(B, C2, H, W)，x1形状：(B, C1, H, W)
        # 拼接后形状：(B, C1+C2, H, W)，即 (B, in_channels, H, W)
        x1 = torch.cat([x2, x1], dim=1)
        
        # 步骤3：通过卷积序列处理融合后的特征
        # 输入形状：(B, in_channels, H, W) → 输出形状：(B, out_channels, H, W)
        return self.conv(x1)


class CamEncode(nn.Module):
    """
    相机编码器类，负责将2D相机图像转换为3D深度感知特征。
    
    Lift-Splat-Shoot方法的核心组件之一，实现从多视角图像到3D空间特征的映射。
    
    Args:
        D (int): 深度离散化步数（如41，表示将深度范围分为41个区间）
        C (int): 3D特征通道数（如64，用于BEV特征表示）
        downsample (int): 特征图下采样因子（如16）
    """
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        
        # 深度相关参数
        self.D = D  # 深度步数，例如41（对应深度范围4m-45m，步长1m）
        self.C = C  # 3D特征通道数，例如64

        # 预训练主干网络：使用EfficientNet-B0作为图像特征提取器
        # 从ImageNet预训练权重初始化，提供丰富的视觉特征表示
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        # 上采样融合模块：融合EfficientNet不同层级的特征
        # 输入通道320+112：EfficientNet两个不同层级的特征图通道数
        # 输出通道512：将融合后的特征映射到512维
        self.up1 = Up(320+112, 512)
        
        # 深度估计网络：通过1x1卷积生成深度分布和3D特征
        # 输入：512维特征图（来自上采样模块）
        # 输出：D+C维特征（前D维用于深度分布，后C维用于3D特征表示）
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)  # 输出通道数：41+64=105

    def get_depth_dist(self, x, eps=1e-20):
        """
        将深度对数概率转换为深度概率分布。
        
        该方法通过softmax函数对深度维度进行归一化，使得每个空间位置的
        所有深度层概率之和为1，从而得到深度的概率分布。
        
        Args:
            x (torch.Tensor): 深度对数概率张量，形状为 (B, D, H, W)
                - B: batch size
                - D: 深度步数（如41）
                - H, W: 特征图高度和宽度
            eps (float): 数值稳定性参数（默认1e-20），防止softmax计算时出现数值问题
            
        Returns:
            torch.Tensor: 深度概率分布，形状与输入相同 (B, D, H, W)
                每个空间位置 (h, w) 的深度概率满足 sum_{d=1 to D} p(d|h,w) = 1
        """
        # 在深度维度（dim=1）上应用softmax，将对数概率转换为概率分布
        # softmax公式：p_i = exp(x_i) / sum_j exp(x_j)
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        """
        从图像特征中提取深度分布和深度感知的3D特征。
        
        该方法是Lift-Splat-Shoot中"Lift"阶段的核心，实现从2D图像到3D特征的转换：
        1. 使用EfficientNet提取多尺度特征
        2. 通过depthnet生成深度分布和特征表示
        3. 将特征与深度分布加权，得到深度感知的3D特征
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, 3, H, W)
            
        Returns:
            tuple: 包含两个元素
                - depth (torch.Tensor): 深度概率分布，形状为 (B, D, H/16, W/16)
                - new_x (torch.Tensor): 深度感知的3D特征，形状为 (B, C, D, H/16, W/16)
            
        符号说明：
            - B: batch size
            - D: 深度步数（如41，对应深度范围4m-45m）
            - C: 特征通道数（如64）
            - H, W: 原始图像高度和宽度
        """
        # 步骤1：通过EfficientNet提取多尺度特征并融合
        # 输入：(B, 3, H, W) → 输出：(B, 512, H/16, W/16)
        x = self.get_eff_depth(x)
        
        # 步骤2：通过depthnet生成深度分布和特征表示
        # depthnet是1x1卷积，输入512通道，输出D+C通道
        # 输入：(B, 512, H/16, W/16) → 输出：(B, D+C, H/16, W/16)
        x = self.depthnet(x)

        # 步骤3：提取深度分布（前D通道）并通过softmax归一化
        # x[:, :self.D] 取前D通道，形状为 (B, D, H/16, W/16)
        # get_depth_dist内部调用softmax(dim=1)，使每个空间位置的深度概率和为1
        depth = self.get_depth_dist(x[:, :self.D])  # 形状：(B, D, H/16, W/16)
        
        # 步骤4：生成深度感知的3D特征
        # x[:, self.D:(self.D + self.C)] 取后C通道作为特征，形状为 (B, C, H/16, W/16)
        # unsqueeze(2) 在深度维度扩展，形状变为 (B, C, 1, H/16, W/16)
        
        # depth.unsqueeze(1) 在通道维度扩展，形状变为 (B, 1, D, H/16, W/16)
        # 逐元素相乘后，每个深度层的特征被对应的深度概率加权
        
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        # 输出形状：(B, C, D, H/16, W/16)，即 (B, 通道数, 深度步数, 高度, 宽度)

        return depth, new_x

    def get_eff_depth(self, x):
        """
        从EfficientNet提取多尺度特征并进行上采样融合。
        
        该方法是对EfficientNet官方实现的适配，用于提取不同层级的特征图，
        并通过上采样模块融合高分辨率细节和高语义信息。
        
        Args:
            x (torch.Tensor): 输入图像张量，形状为 (B, 3, H, W)
            
        Returns:
            torch.Tensor: 融合后的特征图，形状为 (B, 512, H/16, W/16)
            
        特征提取流程：
        1. Stem层：初始卷积+BN+激活
        2. Blocks：遍历所有MBConv块，记录下采样时的特征图
        3. Head：将reduction_5上采样后与reduction_4融合
        
        EfficientNet-B0的特征层级（reduction指下采样倍数）：
        - reduction_1: 1/2 分辨率，通道数 16
        - reduction_2: 1/4 分辨率，通道数 24
        - reduction_3: 1/8 分辨率，通道数 40
        - reduction_4: 1/16 分辨率，通道数 112  ← 高分辨率细节特征
        - reduction_5: 1/32 分辨率，通道数 320  ← 高语义特征
        """
        # 代码改编自：https://github.com/lukemelas/EfficientNet-PyTorch
        endpoints = dict()  # 存储不同层级的特征图

        # Stem层：初始卷积层，将输入通道从3映射到EfficientNet的初始通道数
        # 结构：Conv2d → BatchNorm → Swish激活
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x  # 记录前一层特征，用于检测下采样

        # Blocks：遍历所有MBConv块（Mobile Inverted Residual Bottleneck）
        for idx, block in enumerate(self.trunk._blocks):
            # Drop Connect正则化：按块索引线性调整丢弃率（早期块丢弃率低，晚期块丢弃率高）
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)
            
            # 前向传播通过当前块
            x = block(x, drop_connect_rate=drop_connect_rate)
            
            # 检测是否发生下采样（特征图高度/宽度减半）
            # 当下采样发生时，记录前一层的特征图
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            
            prev_x = x  # 更新前一层特征

        # Head：记录最后一个块的输出（最深层特征）
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        
        # 特征融合：将reduction_5（低分辨率高语义）上采样后与reduction_4（高分辨率细节）拼接
        # 输入：reduction_5 (B, 320, H/32, W/32), reduction_4 (B, 112, H/16, W/16)
        # 输出：融合后特征 (B, 512, H/16, W/16)
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    """
    BEV（鸟瞰图）编码器类，负责将BEV特征图转换为最终的语义分割输出。
    
    这是Lift-Splat-Shoot架构中的"Shoot"阶段，实现从BEV特征到分割掩码的转换。
    采用编码器-解码器结构，基于ResNet-18的前三层提取特征，然后通过两次上采样恢复分辨率。
    
    Args:
        inC (int): 输入通道数（如64，来自CamEncode的深度感知特征投影后的通道数）
        outC (int): 输出通道数（如分割任务中的类别数）
    """
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        # 初始化ResNet-18作为主干网络（不加载预训练权重，残差连接初始化为零）
        # zero_init_residual=True：将残差块的最后BN层初始化为零，有助于训练初期的稳定性
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        
        # 第一层卷积：将输入通道映射到64通道
        # kernel_size=7, stride=2, padding=3：使特征图尺寸减半（same padding的变种）
        # bias=False：后续有BN层，无需额外偏置
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1    # 从ResNet-18获取预初始化的BN层
        self.relu = trunk.relu  # ReLU激活函数

        # 从ResNet-18获取前三个残差块
        # layer1: 输出通道64，步长1，尺寸不变
        # layer2: 输出通道128，步长2，尺寸减半
        # layer3: 输出通道256，步长2，尺寸减半
        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        # 第一次上采样融合模块（U-Net风格跳跃连接）
        # 输入：layer3输出(256通道) 和 layer1输出(64通道)，共64+256=320通道
        # 输出：256通道，尺寸放大4倍（通过scale_factor=4）
        # 作用：融合深层语义特征和浅层细节特征
        self.up1 = Up(64+256, 256, scale_factor=4)
        
        # 第二次上采样序列：进一步恢复分辨率并生成最终输出
        self.up2 = nn.Sequential(
            # 双线性上采样，尺寸放大2倍
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 3x3卷积，将256通道映射到128通道（same padding，尺寸不变）
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 1x1卷积，将128通道映射到输出通道数（如分割类别数）
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        """
        BEV编码器的前向传播，将BEV特征图转换为语义分割输出。
        
        执行流程：
        1. 初始卷积+BN+ReLU
        2. 通过ResNet-18的三个残差块提取特征
        3. 第一次上采样融合（跳跃连接）
        4. 第二次上采样生成最终输出
        
        Args:
            x (torch.Tensor): BEV特征图，形状为 (B, inC, H_bev, W_bev)
                - B: batch size
                - inC: 输入通道数（如64）
                - H_bev, W_bev: BEV网格的高度和宽度
            
        Returns:
            torch.Tensor: 语义分割输出，形状为 (B, outC, H_out, W_out)
                - outC: 输出通道数（如分割类别数）
                - H_out, W_out: 输出特征图的高度和宽度（通常是输入的8倍）
        """
        # 阶段1：初始卷积层（下采样2倍）
        # 输入：(B, inC, H_bev, W_bev) → 输出：(B, 64, H_bev/2, W_bev/2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 阶段2：ResNet-18残差块（共下采样4倍）
        # layer1: (B, 64, H_bev/2, W_bev/2) → (B, 64, H_bev/2, W_bev/2) （尺寸不变）
        x1 = self.layer1(x)  # 保存layer1输出用于跳跃连接
        # layer2: (B, 64, H_bev/2, W_bev/2) → (B, 128, H_bev/4, W_bev/4) （下采样2倍）
        x = self.layer2(x1)
        # layer3: (B, 128, H_bev/4, W_bev/4) → (B, 256, H_bev/8, W_bev/8) （下采样2倍）
        x = self.layer3(x)

        # 阶段3：第一次上采样融合（U-Net跳跃连接）
        # 输入：layer3输出(256通道) 和 layer1输出(64通道)
        # 输出：(B, 256, H_bev/2, W_bev/2) （尺寸放大4倍）
        x = self.up1(x, x1)
        
        # 阶段4：第二次上采样（生成最终输出）
        # 输入：(B, 256, H_bev/2, W_bev/2)
        # 输出：(B, outC, H_bev, W_bev) （尺寸放大2倍，通道数变为outC）
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    """
    Lift-Splat-Shoot 核心模型类，实现从多相机图像到鸟瞰图(BEV)的端到端转换。
    
    核心流程：
    1. Lift：将2D图像特征提升到3D空间
    2. Splat：将3D特征投影到BEV网格
    3. Shoot：通过BEV编码器生成最终输出
    
    Args:
        grid_conf (dict): BEV网格配置，包含xbound, ybound, zbound, dbound
        data_aug_conf (dict): 数据增强配置，包含图像尺寸、旋转、翻转等参数
        outC (int): 输出通道数（如分割任务中为类别数）
    """
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        
        # 保存配置参数
        self.grid_conf = grid_conf          # BEV网格配置
        self.data_aug_conf = data_aug_conf  # 数据增强配置

        # 生成BEV网格参数：dx(步长), bx(偏移), nx(网格数量)
        # xbound/ybound/zbound格式: [min, max, step]
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)  # 网格步长 [x_step, y_step, z_step]
        self.bx = nn.Parameter(bx, requires_grad=False)  # 网格偏移 [x_offset, y_offset, z_offset]
        self.nx = nn.Parameter(nx, requires_grad=False)  # 网格数量 [x_num, y_num, z_num]

        # 网络架构参数
        self.downsample = 16  # 特征图下采样因子（原图1600x900 -> 100x56）
        self.camC = 64        # 相机特征通道数
        
        # 创建相机视锥体（用于将图像坐标映射到3D空间）
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape  # D: 深度步数（如41，对应深度范围4m-45m）
        
        # 初始化相机编码器：将2D图像转换为深度感知的3D特征
        # 为每个像素位置和深度生成一个64维的特征向量
        self.camencode = CamEncode(self.D, self.camC, self.downsample)  # (深度步数, 特征通道数, 下采样因子)
        
        # 初始化BEV编码器：将BEV特征映射到最终输出（如分割掩码）
        self.bevencode = BevEncode(inC=self.camC, outC=outC)  # (输入通道数, 输出通道数)（64, 1）

        # 快速累加开关：使用QuickCumsum替代PyTorch原生autograd，加速训练
        self.use_quickcumsum = True
    
    def create_frustum(self):
        """
        创建相机视锥体（Frustum）坐标网格，用于将2D图像坐标映射到3D空间。
        
        视锥体是相机视野内的3D空间区域，这里通过离散化深度范围和图像平面坐标，
        构建一个规则的3D坐标网格。这个网格将用于后续的特征提升（Lift）操作，
        将2D图像特征映射到3D空间中的对应位置。
        
        Returns:
            torch.nn.Parameter: 视锥体坐标网格，形状为 (D, fH, fW, 3)
                - D: 深度步数（如41，对应深度范围4m-45m）
                - fH: 特征图高度（原图高度/下采样因子，如128/16=8）
                - fW: 特征图宽度（原图宽度/下采样因子，如352/16=22）
                - 最后一维3: (x, y, z) 坐标
        """
        # 步骤1：获取图像尺寸参数
        # original feature Height/Width：数据增强后图像的最终尺寸
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 通常为 (128, 352)
        # feature Height/Width：经过下采样后的特征图尺寸
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 通常为 (8, 22)
        
        # 步骤2：生成深度坐标（z轴）
        # dbound格式: [min_depth, max_depth, depth_step]，如 [4.0, 45.0, 1.0]
        # torch.arange生成从min_depth到max_depth的序列，步长为depth_step
        # view(-1, 1, 1) 将1D序列转换为3D张量 (D, 1, 1)
        # expand(-1, fH, fW) 在H和W维度扩展，得到 (D, fH, fW)
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape  # D为深度步数，如41
        
        # 步骤3：生成宽度坐标（x轴）
        # torch.linspace生成从0到ogfW-1的fW个均匀分布的点
        # view(1, 1, fW) 转换为3D张量 (1, 1, fW)
        # expand(D, fH, fW) 在D和H维度扩展，得到 (D, fH, fW)
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        
        # 步骤4：生成高度坐标（y轴）
        # torch.linspace生成从0到ogfH-1的fH个均匀分布的点
        # view(1, fH, 1) 转换为3D张量 (1, fH, 1)
        # expand(D, fH, fW) 在D和W维度扩展，得到 (D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # 步骤5：堆叠坐标形成完整的3D网格
        # torch.stack在最后一维(-1)堆叠xs, ys, ds，得到 (D, fH, fW, 3)
        # 每个位置的坐标为 (x, y, z)，其中z为深度值
        frustum = torch.stack((xs, ys, ds), -1)
        
        # 将视锥体坐标包装为nn.Parameter，requires_grad=False表示不参与梯度更新
        # 因为视锥体坐标是固定的，不需要训练
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """
        计算相机视锥体中每个像素在 ego 坐标系（车辆坐标系）中的 3D 坐标。
        
        该方法实现了从图像像素坐标到 3D 世界坐标的完整转换，包含以下步骤：
        1. 撤销数据增强的后处理变换（post_rots/post_trans）
        2. 将归一化图像坐标转换为相机坐标系下的 3D 坐标
        3. 将相机坐标系转换为 ego 坐标系（车辆坐标系）
        
        Args:
            rots (torch.Tensor): 相机外参旋转矩阵，形状为 (B, N, 3, 3)
                                描述相机相对于车辆坐标系的旋转
            trans (torch.Tensor): 相机外参平移向量，形状为 (B, N, 3)
                                 描述相机相对于车辆坐标系的位置
            intrins (torch.Tensor): 相机内参矩阵，形状为 (B, N, 3, 3)
                                   包含焦距(fx, fy)、主点(cx, cy)等参数
            post_rots (torch.Tensor): 图像后处理旋转矩阵，形状为 (B, N, 3, 3)
                                     数据增强时应用的旋转变换
            post_trans (torch.Tensor): 图像后处理平移向量，形状为 (B, N, 3)
                                      数据增强时应用的平移变换
            
        Returns:
            torch.Tensor: 每个像素在 ego 坐标系中的 3D 坐标，形状为 (B, N, D, H', W', 3)
                          B: 批次大小，N: 相机数量，D: 深度步数
                          H': 下采样后的图像高度，W': 下采样后的图像宽度
                          最后一维3表示 (x, y, z) 坐标
        
        坐标变换流程（从图像到 ego 坐标系）：
        
        1. 初始视锥体坐标（归一化图像平面）
           self.frustum 预计算了 D 个深度层上每个像素的 (u, v, d) 坐标
           u, v ∈ [-1, 1]（归一化图像坐标），d ∈ [dbound[0], dbound[1]]（深度值）
        
        2. 撤销后处理变换（undo post-transformation）
           将经过数据增强变换的坐标还原回原始相机坐标系
           points = inv(post_rots) × (frustum - post_trans)
        
        3. 图像坐标 → 相机坐标系（cam_to_ego 第一步）
           将归一化图像坐标转换为相机坐标系下的 3D 坐标
           [x_cam, y_cam, z_cam] = [u×d, v×d, d]
        
        4. 相机坐标系 → ego 坐标系（cam_to_ego 第二步）
           使用外参将相机坐标转换为车辆坐标
           points_ego = rots × inv(intrins) × points_cam + trans
        
        关键数学公式：
        - 图像坐标到相机坐标: x = u * d, y = v * d, z = d
        - 相机坐标到 ego 坐标: P_ego = R * K^{-1} * P_cam + T
          其中 R 是旋转矩阵，K 是内参矩阵，T 是平移向量
        """
        B, N, _ = trans.shape

        # 步骤1：撤销数据增强的后处理变换
        # 将经过 post_rots/post_trans 变换的坐标还原
        # self.frustum 形状: (D, H', W', 3)，需要广播到 (B, N, D, H', W', 3)
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # 步骤2：图像坐标 → 相机坐标系
        # 将归一化图像坐标 (u, v) 转换为相机坐标系下的 3D 坐标
        # [x, y, z] = [u*d, v*d, d]，其中 d 是深度值
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        
        # 步骤3：相机坐标系 → ego 坐标系
        # combine = rots × inv(intrins): 组合旋转和内参逆矩阵
        # points_ego = combine × points_cam + trans
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
        """
        Lift-Splat 阶段的核心方法，将多视角相机图像转换为 BEV 体素特征。
        
        该方法实现了 Lift-Splat-Shoot 中的 "Lift" 和 "Splat" 两个步骤：
        1. Lift：将 2D 图像特征提升到 3D 空间（增加深度维度）
        2. Splat：将 3D 特征投影并聚合到 BEV 体素网格上
        
        Args:
            x (torch.Tensor): 输入的多视角图像张量，形状为 (B, N, 3, H, W)
                             B: 批次大小，N: 相机数量，3: RGB通道，H: 图像高度，W: 图像宽度
            rots (torch.Tensor): 相机外参旋转矩阵，形状为 (B, N, 3, 3)
                                描述相机相对于车辆坐标系的旋转
            trans (torch.Tensor): 相机外参平移向量，形状为 (B, N, 3)
                                 描述相机相对于车辆坐标系的位置
            intrins (torch.Tensor): 相机内参矩阵，形状为 (B, N, 3, 3)
                                   包含焦距、主点等参数
            post_rots (torch.Tensor): 图像后处理旋转矩阵，形状为 (B, N, 3, 3)
                                     用于数据增强的旋转变换
            post_trans (torch.Tensor): 图像后处理平移向量，形状为 (B, N, 3)
                                      用于数据增强的平移变换
            
        Returns:
            torch.Tensor: BEV 体素特征图，形状为 (B, C, H_bev, W_bev)
                          C: 特征通道数（包含深度维度展开）
                          H_bev, W_bev: BEV 网格的高度和宽度
        
        执行流程：
        1. get_geometry: 计算每个像素对应的 3D 空间坐标（X, Y, Z）
           - 输入：相机参数
           - 输出：几何特征张量，形状为 (B, N, D, H, W, 3)
        
        2. get_cam_feats: 提取相机图像特征并扩展到深度维度
           - 通过 CamEncode 编码器提取 2D 图像特征
           - 将特征扩展到 D 个深度层，生成深度感知特征
           - 输出形状：(B, N, D, H', W', C)
        
        3. voxel_pooling: 将 3D 特征投影并聚合到 BEV 网格
           - 根据几何坐标将特征分配到对应的体素
           - 使用累积求和技巧聚合同一体素内的特征
           - 输出形状：(B, C, H_bev, W_bev)
        """
        # 步骤1：计算每个像素的 3D 几何位置
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        
        # 步骤2：提取相机特征并扩展到深度维度（Lift 步骤）
        x = self.get_cam_feats(x)

        # 步骤3：将特征投影并聚合到 BEV 体素网格（Splat 步骤）
        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        """
        Lift-Splat-Shoot 模型的核心前向传播方法，实现从多视角图像到 BEV 特征的完整转换。
        
        该方法是整个模型的入口，包含两个关键阶段：
        1. Lift-Splat：将多视角相机图像提升（Lift）到 3D 空间并投影（Splat）到 BEV 体素网格
        2. Shoot：通过 BEV 编码器处理体素特征，输出最终预测结果
        
        Args:
            x (torch.Tensor): 输入的多视角图像张量，形状为 (B, N, 3, H, W)
                             B: 批次大小，N: 相机数量，3: RGB通道，H: 图像高度，W: 图像宽度
            rots (torch.Tensor): 相机外参旋转矩阵，形状为 (B, N, 3, 3)
            trans (torch.Tensor): 相机外参平移向量，形状为 (B, N, 3)
            intrins (torch.Tensor): 相机内参矩阵，形状为 (B, N, 3, 3)
            post_rots (torch.Tensor): 图像后处理旋转矩阵，形状为 (B, N, 3, 3)
            post_trans (torch.Tensor): 图像后处理平移向量，形状为 (B, N, 3)
            
        Returns:
            torch.Tensor: BEV 特征图或分割预测结果，形状为 (B, outC, H_bev, W_bev)
                          outC: 输出通道数（如分割任务中为类别数）
                          H_bev, W_bev: BEV 特征图的高度和宽度
        
        执行流程：
        1. get_voxels: 将多视角图像转换为 BEV 体素特征
           - get_geometry: 计算每个像素的 3D 几何位置
           - get_cam_feats: 提取相机特征并扩展到深度维度
           - voxel_pooling: 将特征投影并聚合到 BEV 网格
        2. bevencode: 通过卷积编码器处理 BEV 特征，输出最终预测
        """
        # 阶段1：Lift-Splat - 将多视角图像转换为 BEV 体素特征
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        
        # 阶段2：Shoot - 通过 BEV 编码器生成最终预测
        x = self.bevencode(x)
        
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)