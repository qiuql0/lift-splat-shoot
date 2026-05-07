"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    """
    nuScenes数据集的PyTorch Dataset封装类。
    
    负责加载和预处理nuScenes数据集，为Lift-Splat-Shoot模型提供训练/验证数据。
    主要功能包括：场景划分、数据预处理、BEV网格参数计算、数据格式修复等。
    """
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        """
        初始化nuScenes数据集加载器。
        
        Args:
            nusc (NuScenes): nuScenes数据集对象，包含完整的数据集元信息
            is_train (bool): 是否为训练模式，决定数据增强策略和数据集划分
            data_aug_conf (dict): 数据增强配置，包含图像尺寸、旋转、翻转等参数
            grid_conf (dict): BEV网格配置，包含xbound、ybound、zbound、dbound
                - xbound: [min_x, max_x, step_x]，x轴范围和步长
                - ybound: [min_y, max_y, step_y]，y轴范围和步长
                - zbound: [min_z, max_z, step_z]，z轴范围和步长
                - dbound: [min_depth, max_depth, step_depth]，深度范围和步长
        """
        # 保存基础配置参数
        self.nusc = nusc                # nuScenes数据集对象
        self.is_train = is_train        # 训练/验证模式标志
        self.data_aug_conf = data_aug_conf  # 数据增强配置
        self.grid_conf = grid_conf      # BEV网格配置

        # 步骤1：获取场景列表（根据训练/验证模式划分）
        self.scenes = self.get_scenes()
        
        # 步骤2：预处理数据，生成样本索引列表
        # self.ixes是一个列表，每个元素是一个样本的元数据（如scene_token, sample_token等）
        self.ixes = self.prepro()

        # 步骤3：计算BEV网格参数
        # gen_dx_bx函数生成：
        # - dx: 网格步长 [x_step, y_step, z_step]
        # - bx: 网格偏移（中心位置）[x_offset, y_offset, z_offset]， 最小值加上半个偏移量，也就是第一个网格的中心位置
        # - nx: 网格数量 [x_num, y_num, z_num]
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        # 步骤4：修复nuScenes数据格式（处理trainval/1, trainval/2等目录结构）
        self.fix_nuscenes_formatting()

        # 步骤5：打印数据集信息（样本数量、配置参数等）
        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):
        """
        采样数据增强参数。
        
        根据训练/验证模式，生成一组图像变换参数，包括：缩放比例、裁剪区域、
        是否水平翻转、旋转角度。这些参数将用于后续的图像变换操作。
        
        训练模式下：参数是随机采样的，增加数据多样性
        验证/测试模式下：参数是固定的，保证结果的可重复性
        
        Returns:
            tuple: 包含5个元素的元组
                - resize (float): 缩放比例
                - resize_dims (tuple): 缩放后的图像尺寸 (width, height)
                - crop (tuple): 裁剪区域 (x_min, y_min, x_max, y_max)
                - flip (bool): 是否进行水平翻转
                - rotate (float): 旋转角度（度）
        """
        # 获取配置参数
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']  # 原始图像高度和宽度(900, 1600)
        fH, fW = self.data_aug_conf['final_dim']  # 最终输出图像尺寸 (128, 352)
        
        if self.is_train:
            # ========== 训练模式：随机增强 ==========
            
            # 1. 随机缩放：在resize_lim范围内均匀采样缩放比例
            # resize_lim格式: [min_resize, max_resize] # [0.193, 0.225]
            # 缩放后的宽度 309~360 略大于目标宽度 352
            # 缩放后的高度 174~203 略大于目标高度 128！！！！！，这里是大于128的   
            resize = np.random.uniform(*self.data_aug_conf['resize_lim']) # [0.193, 0.225]
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            
            # 2. 随机裁剪：去除图像底部的一定比例，从上方剩余区域中裁剪（保留道路信息）
            # bot_pct_lim格式: [min_bot_pct, max_bot_pct]，如[0.0, 0.22]
            # 先去除底部bot_pct比例的区域，然后从上方(1-bot_pct)的区域中裁剪fH x fW
            # 缩放后最低高度174，即使裁剪笔记0.22，也剩余135高度，大于目标高度128
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) * newH) - fH # h的起始位置
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))  # w的起始位置，因为newW(如：209)可能小于fW(352)，所以这里要用max
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)  # (x_min, y_min, x_max, y_max)
            
            # 3. 随机水平翻转：根据rand_flip配置决定是否启用
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            
            # 4. 随机旋转：在rot_lim范围内均匀采样旋转角度
            # rot_lim格式: [min_rot, max_rot]，如[-5.4, 5.4]（度）
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim']) # rot_lim=(-5.4, 5.4), 旋转角度范围 -5.4~5.4
            
        else:
            # ========== 验证/测试模式：固定增强 ==========
            
            # 1. 固定缩放：取刚好能包含final_dim的最小缩放比例
            resize = max(fH / H, fW / W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            
            # 2. 固定裁剪：居中裁剪，使用bot_pct_lim的均值
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)  # 水平方向居中
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # 3. 不翻转
            flip = False
            
            # 4. 不旋转
            rotate = 0
        
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        """
        获取多视角相机图像数据和对应的相机参数。
        
        该方法遍历指定的相机列表，加载每个相机的图像，并获取相机的内外参数。
        同时应用数据增强（缩放、裁剪、翻转、旋转），并返回处理后的图像和参数。
        
        Args:
            rec (dict): 样本记录字典，包含样本的元数据（如scene_token、sample_token等）
            cams (list/np.ndarray): 要处理的相机名称列表（如['CAM_FRONT', 'CAM_FRONT_LEFT', ...]）
            
        Returns:
            tuple: 包含6个张量的元组
                - imgs (torch.Tensor): 处理后的图像张量，形状为 (Ncams, 3, H, W)
                - rots (torch.Tensor): 相机外参旋转矩阵，形状为 (Ncams, 3, 3)
                - trans (torch.Tensor): 相机外参平移向量，形状为 (Ncams, 3)
                - intrins (torch.Tensor): 相机内参矩阵，形状为 (Ncams, 3, 3)
                - post_rots (torch.Tensor): 后处理旋转矩阵，形状为 (Ncams, 3, 3)
                - post_trans (torch.Tensor): 后处理平移向量，形状为 (Ncams, 3)
        """
        # 初始化存储列表
        imgs = []           # 存储处理后的图像
        rots = []           # 存储相机外参旋转矩阵
        trans = []          # 存储相机外参平移向量
        intrins = []        # 存储相机内参矩阵
        post_rots = []      # 存储后处理旋转矩阵（用于数据增强后的坐标变换）
        post_trans = []     # 存储后处理平移向量（用于数据增强后的坐标变换）
        
        # 遍历每个相机
        for cam in cams:
            # 步骤1：获取相机数据记录
            # rec['data'][cam] 是该相机的sample_data token
            samp = self.nusc.get('sample_data', rec['data'][cam])
            
            # 步骤2：构建图像路径并加载图像
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            img = Image.open(imgname)
            
            # 步骤3：初始化后处理变换矩阵（2D，后续扩展为3D）
            post_rot = torch.eye(2)   # 初始化为单位矩阵(2,2)
            post_tran = torch.zeros(2) # 初始化为零向量(2,)

            # 步骤4：获取相机标定参数
            # sens 包含相机的内参、外参、畸变参数等
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            intrin = torch.Tensor(sens['camera_intrinsic'])  # 3x3内参矩阵
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)  # 3x3旋转矩阵
            tran = torch.Tensor(sens['translation'])  # 3D平移向量

            # 步骤5：应用数据增强
            # sample_augmentation() 返回增强参数：缩放、裁剪、翻转、旋转
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()
            # img_transform() 对图像进行变换，并更新后处理变换矩阵
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # 步骤6：将后处理变换矩阵扩展为3x3（方便后续3D计算）
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2  # 将2D平移向量放入前两维
            post_rot[:2, :2] = post_rot2  # 将2D旋转矩阵放入左上角

            # 步骤7：归一化图像并收集数据
            imgs.append(normalize_img(img))  # 图像归一化到image net规范
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        # 将列表转换为张量并返回
        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        """
        生成BEV（鸟瞰图）语义分割标签（二进制掩码）。
        
        该方法将nuScenes数据集中的3D车辆标注转换为BEV网格上的二进制掩码。
        具体流程：
        1. 获取当前帧的ego车辆姿态
        2. 将3D边界框从全局坐标系转换到ego车辆坐标系
        3. 将边界框的底部角点投影到BEV网格
        4. 使用多边形填充生成二进制掩码（1表示车辆，0表示背景）
        
        Args:
            rec (dict): 样本记录字典，包含样本的元数据（如scene_token、anns等）
            
        Returns:
            torch.Tensor: BEV语义分割标签，形状为 (1, H_bev, W_bev)
                - H_bev: BEV网格高度（self.nx[0]，即x方向网格数量）
                - W_bev: BEV网格宽度（self.nx[1]，即y方向网格数量）
                - 值为1表示车辆区域，0表示背景
        """
        # 步骤1：获取当前帧的ego车辆姿态
        # 通过LIDAR_TOP传感器数据获取ego_pose token
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        
        # 步骤2：计算从全局坐标系到ego车辆坐标系的变换
        # trans: 平移向量（取反，因为要将点从全局坐标系转换到ego坐标系）
        # rot: 旋转四元数的逆（因为要将点从全局坐标系旋转到ego坐标系）
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        
        # 步骤3：初始化BEV网格（全0背景）
        # self.nx[0]: x方向网格数量（BEV高度）
        # self.nx[1]: y方向网格数量（BEV宽度）
        img = np.zeros((self.nx[0], self.nx[1])) # （200, 200）
        
        # 步骤4：遍历所有标注实例
        for tok in rec['anns']:
            # 获取实例标注信息
            inst = self.nusc.get('sample_annotation', tok)
            
            # 过滤非车辆类别（只保留vehicle类别）
            # category_name格式如：'vehicle.car'，split('.')[0]提取第一部分
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            
            # 创建3D边界框（使用nuScenes的Box类）
            # 参数：中心坐标、尺寸、旋转四元数
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            
            # 将边界框从全局坐标系转换到ego车辆坐标系
            box.translate(trans)  # 平移
            box.rotate(rot)       # 旋转

            # [
                # [x0, x1, x2, x3],  # 第0行：所有角点的x坐标
                # [y0, y1, y2, y3],  # 第1行：所有角点的y坐标
                # [z0, z1, z2, z3],  # 第2行：所有角点的z坐标（底部角点的z值相同）
                # [ 1,  1,  1,  1],  # 第3行：齐次坐标的固定项
            # ]
            # 步骤5：提取边界框底部的四个角点（用于BEV投影）
            # bottom_corners()返回4x4矩阵，[:2]取前两行（x, y坐标），.T转置为4x2矩阵
            pts = box.bottom_corners()[:2].T
            
            # 步骤6：将连续坐标转换为BEV网格的离散坐标
            # 计算公式：(坐标 - 网格偏移 + 步长/2) / 步长
            # 四舍五入后转换为整数索引
            # 等价于pts - (self.bx[:2] - self.dx[:2]/2.)，这里减self.dx[:2]/2.是因为self.bx[:2]已经加过半个偏移量了
            # pts - (self.bx[:2] - self.dx[:2]/2.)的含义是距离（-50, -50）的距离，除以self.dx[:2]，表示多少个格子。0.5米算一个格子
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            
            # 步骤7：交换x和y坐标（因为BEV网格的坐标系与图像坐标系不同）
            # BEV网格中：x对应高度，y对应宽度
            pts[:, [1, 0]] = pts[:, [0, 1]] # 相当于以（-50, 50）为原点了
            
            # 步骤8：使用OpenCV填充多边形，生成车辆区域的掩码
            cv2.fillPoly(img, [pts], 1.0)

        # 步骤9：转换为PyTorch张量，并添加通道维度（从(H, W)变为(1, H, W)）
        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        """
        选择要使用的相机视角。
        
        在训练模式下，如果配置的相机数量(Ncams)小于可用相机总数，则随机选择Ncams个相机；
        在验证/测试模式下，或当Ncams等于可用相机总数时，使用所有相机。
        
        这种设计可以增加训练数据的多样性，防止模型过度依赖特定相机视角。
        
        Returns:
            list/np.ndarray: 选中的相机名称列表
                - 训练模式（随机选择）: numpy数组，包含Ncams个随机选择的相机名称
                - 其他模式: 原始相机列表，包含所有可用相机
        """
        # 训练模式下，且需要的相机数量小于可用相机总数
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            # 无放回地随机选择Ncams个相机
            # replace=False表示每个相机只能被选中一次
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            # 验证/测试模式，或使用所有相机
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        rec = self.ixes[index]
        
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        """
        获取单个训练样本（实现PyTorch Dataset的核心方法）。
        
        该方法返回单个样本的完整数据，包括多视角相机图像、相机姿态参数、
        内参矩阵以及对应的语义分割标签（鸟瞰图二进制掩码）。
        
        Args:
            index (int): 样本索引，范围为 [0, len(self.ixes)-1]
            
        Returns:
            tuple: 包含7个元素的元组
                - imgs (torch.Tensor): 多视角相机图像张量，形状为 (Ncams, 3, H, W)
                    - Ncams: 相机数量（如5或6）
                    - 3: RGB通道
                    - H, W: 图像高度和宽度
                - rots (torch.Tensor): 相机外参旋转矩阵，形状为 (Ncams, 3, 3)
                    表示相机坐标系到ego车辆坐标系的旋转
                - trans (torch.Tensor): 相机外参平移向量，形状为 (Ncams, 3)
                    表示相机坐标系到ego车辆坐标系的平移
                - intrins (torch.Tensor): 相机内参矩阵，形状为 (Ncams, 3, 3)
                    包含焦距、主点等参数
                - post_rots (torch.Tensor): 后处理旋转矩阵，形状为 (Ncams, 3, 3)
                    用于图像裁剪/缩放后的坐标变换
                - post_trans (torch.Tensor): 后处理平移向量，形状为 (Ncams, 3)
                    用于图像裁剪/缩放后的坐标变换
                - binimg (torch.Tensor): BEV语义分割标签，形状为 (1, H_bev, W_bev)
                    二进制掩码，1表示目标类别（如车辆），0表示背景
        """
        # 获取样本记录（包含该样本的元数据信息）
        rec = self.ixes[index]

        # 选择要使用的相机视角（随机选择ncams个相机）
        cams = self.choose_cams()
        
        # 获取图像数据和相机参数
        # 包括：原始图像、外参（旋转+平移）、内参、后处理变换
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        
        # 获取BEV语义分割标签（二进制掩码）
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader