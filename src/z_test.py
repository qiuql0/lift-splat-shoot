"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch

from .data import compile_data
from .tools import ego_to_cam
from .models import compile_model


def geometry_check(version,
                   dataroot='/data/nuscenes',
                   nbatches=None,
                   tol=1e-2,

                   H=900, W=1600,
                   resize_lim=(0.193, 0.225),
                   final_dim=(128, 352),
                   bot_pct_lim=(0.0, 0.22),
                   rot_lim=(-5.4, 5.4),
                   rand_flip=True,
                   ncams=5,

                   xbound=[-50.0, 50.0, 0.5],
                   ybound=[-50.0, 50.0, 0.5],
                   zbound=[-10.0, 10.0, 20.0],
                   dbound=[4.0, 45.0, 1.0],

                   bsz=4,
                   nworkers=10,
                   ):
    """几何一致性测试

    get_geometry 把视锥模板上的每个点 (u', v', d) 变换到自车坐标系；
    再用 ego_to_cam 投影回原图、施加图像增强，应该回到原来的 (u', v', d)。
    训练集和验证集都测：训练集有随机翻转和旋转，验证集只有固定的缩放和裁剪。

    nbatches: 每个划分最多测多少个 batch，None 表示全部
    tol:      允许的最大误差（u'、v' 的单位是像素，d 的单位是米）

    运行（在仓库根目录）：
        python -m src.z_test geometry_check mini --dataroot=/home/qianli/data/nuscenes
    """
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
                    'resize_lim': resize_lim,
                    'final_dim': final_dim,
                    'rot_lim': rot_lim,
                    'H': H, 'W': W,
                    'rand_flip': rand_flip,
                    'bot_pct_lim': bot_pct_lim,
                    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                    'Ncams': ncams,
                }
    # 不需要激光点云，用 segmentationdata 即可，返回 7 元组
    trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata')

    # get_geometry 只依赖视锥模板和相机参数，与网络权重无关，
    # 所以不需要加载 checkpoint，在 CPU 上算即可
    model = compile_model(grid_conf, data_aug_conf, outC=1)
    # 视锥模板 (D, H/16, W/16, 3) -> 3 x (D*H*W)，每列一个点 (u', v', d)
    target = model.frustum.view(-1, 3).T

    for split, loader in (('train', trainloader), ('val', valloader)):
        max_err = torch.zeros(3)  # u'、v'、d 各自的最大误差
        ncam_total = nflip = nrot = 0
        with torch.no_grad():
            for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(loader):
                if nbatches is not None and batchi >= nbatches:
                    break
                # imgs: (B, N, 3, 128, 352)，binimgs: (B, 1, 200, 200)

                img_pts = model.get_geometry(rots, trans, intrins, post_rots, post_trans)
                # img_pts: (B, N, D, H/16, W/16, 3)，下标在视锥空间，值是自车坐标系下的 (x, y, z)

                for si in range(imgs.shape[0]):
                    for imgi in range(imgs.shape[1]):
                        # (D, H, W, 3) -> (D*H*W, 3) -> 3 x (D*H*W)：ego_to_cam 要求每列一个点。
                        # 不能写成 view(3, -1)：形状虽然一样，但会按内存顺序把不同点的 x、y、z 混在一起
                        ego_pts = img_pts[si, imgi].view(-1, 3).T
                        # 自车坐标 -> 原图上的 (u, v, d)
                        cam_pts = ego_to_cam(ego_pts, rots[si, imgi], trans[si, imgi], intrins[si, imgi])
                        # 施加图像增强 -> 增强后图像上的 (u', v', d)
                        final_pts = post_rots[si, imgi].matmul(cam_pts) + post_trans[si, imgi].unsqueeze(1)

                        # 每一行（u'、v'、d）分别取最大误差
                        err = (final_pts - target).abs().max(dim=1).values
                        max_err = torch.maximum(max_err, err)

                        # 统计覆盖了多少翻转、旋转过的相机，确认测试真的走到了这些增强
                        A = post_rots[si, imgi, :2, :2]
                        ncam_total += 1
                        nflip += int(torch.det(A) < 0)          # 翻转时行列式为负
                        nrot += int(A[0, 1].abs() > 1e-6)       # 有旋转时非对角元不为 0

        print(f"[{split}] 测了 {ncam_total} 个相机图像，其中翻转 {nflip} 个、旋转 {nrot} 个；"
              f"最大误差 u'={max_err[0]:.2e} px，v'={max_err[1]:.2e} px，d={max_err[2]:.2e} m")
        assert max_err.max() < tol, f'{split} 上几何不一致：最大误差 {max_err.max():.3g} 超过 {tol}'

    print('几何一致性测试通过')


if __name__ == '__main__':
    from fire import Fire
    Fire({
        'geometry_check': geometry_check,
    })
