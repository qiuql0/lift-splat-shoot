# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is a PyTorch implementation of Lift-Splat-Shoot (ECCV 2020), which encodes multi-camera images into a bird's-eye-view (BEV) feature map for vehicle segmentation on the nuScenes dataset. This fork has been heavily annotated with Chinese comments explaining the architecture and tensor shapes at each step — when editing, preserve the explanatory comments where they remain accurate.

## Commands

All entry points go through `main.py`, which uses `fire` to dispatch to functions in `src/`. The local mini dataset lives at `data/nuscenes/mini`, so `--dataroot=./data/nuscenes` resolves correctly (`compile_data` joins `dataroot` + `version`).

```bash
# Train (mini split, CPU). gpuid=-1 forces CPU; gpuid>=0 selects cuda:<id>
python main.py train mini --dataroot=./data/nuscenes --logdir=./runs --gpuid=0
tensorboard --logdir=./runs --bind_all

# Evaluate IOU on the validation set
python main.py eval_model_iou mini --modelf=./model_pretrain/model525000.pt --dataroot=./data/nuscenes

# Visualize BEV predictions (writes eval*.jpg to cwd; needs map data)
python main.py viz_model_preds mini --modelf=MODEL.pt --dataroot=./data/nuscenes --map_folder=./data/nuscenes/mini

# Sanity-check extrinsics/intrinsics parsing (writes lcheck*.jpg to cwd)
python main.py lidar_check mini --dataroot=./data/nuscenes --viz_train=False

# Verify QuickCumsum matches autograd cumsum (forward + grad must agree)
python main.py cumsum_check mini --dataroot=./data/nuscenes
```

Swap `mini` for `trainval` to run on the full dataset (`v1.0-{version}` is constructed in `compile_data`). VS Code launch config `lss train` (in `.vscode/launch.json`) runs CPU mini training. There is no test suite, linter, or build step.

Dependencies (not pinned in a requirements file):
```bash
pip install nuscenes-devkit tensorboardX efficientnet_pytorch==0.7.0
```

## Architecture

The data flow is **multi-camera images → per-camera frustum features (Lift) → BEV voxel grid (Splat) → segmentation head (Shoot)**. The model is config-driven by two dicts threaded through every entry point:

- **`grid_conf`**: `xbound`/`ybound`/`zbound`/`dbound`, each `[min, max, step]`. `xyz` bounds define the BEV voxel grid (default 200×200×1 at 0.5m); `dbound` defines depth-plane discretization (default 4–45m, 1m steps → D=41).
- **`data_aug_conf`**: image size (`H`,`W`), `final_dim`, resize/crop/flip/rotate ranges, camera list, and `Ncams`.

`gen_dx_bx` (in `src/tools.py`) turns bounds into `dx` (voxel size), `bx` (first voxel center), `nx` (grid dims). These three appear in both the model and the dataset and must stay consistent.

### Module map

- **`src/models.py`** — the network (`compile_model` → `LiftSplatShoot`):
  - `create_frustum` builds a fixed `(D, fH, fW, 3)` grid of (u, v, depth) image-plane points. `fH,fW = final_dim // downsample` (downsample=16).
  - `get_geometry` maps frustum points into the ego frame using extrinsics (`rots`,`trans`), intrinsics, and the augmentation transforms (`post_rots`,`post_trans`). Returns `(B, N, D, fH, fW, 3)`. **The `post_rots`/`post_trans` "undo" the data augmentation applied in the dataloader** — geometry must match the augmented pixels.
  - `CamEncode` (Lift): EfficientNet-B0 backbone → `depthnet` 1×1 conv outputs `D+C` channels. First D channels become a softmax depth distribution; last C channels are features. Outer product gives `(B, C, D, fH, fW)` depth-weighted features.
  - `voxel_pooling` (Splat): discretizes ego coords to voxel indices, filters out-of-range points, sorts by a packed rank, and sums features falling in the same voxel via cumsum. `QuickCumsum` (custom autograd Function in `tools.py`) is the fast path; `cumsum_trick` is the reference. Output `(B, C, X, Y)` after collapsing Z.
  - `BevEncode` (Shoot): ResNet-18 first layers + two upsampling stages → `(B, outC, X, Y)`. `outC=1` for vehicle segmentation.
- **`src/data.py`** — nuScenes loading. `NuscData` is the base; `SegmentationData` yields `(imgs, rots, trans, intrins, post_rots, post_trans, binimg)` for training; `VizData` additionally yields lidar. `get_binimg` rasterizes vehicle 3D boxes into the BEV label via `cv2.fillPoly`. `sample_augmentation` produces random (train) or fixed (val) resize/crop/flip/rotate params; `fix_nuscenes_formatting` patches file paths for `trainval/1,2,...` split storage.
- **`src/train.py`** — training loop. Loss is `SimpleLoss` (BCE-with-logits, `pos_weight≈2.13` to counter class imbalance). Periodically logs train loss/IOU and validation metrics to TensorBoard and checkpoints to `logdir/model<counter>.pt`.
- **`src/explore.py`** — eval/visualization entry points (`eval_model_iou`, `viz_model_preds`, `lidar_check`, `cumsum_check`).
- **`src/tools.py`** — shared helpers: coordinate transforms (`ego_to_cam`, `cam_to_ego`), `img_transform` (applies augmentation + tracks the post-transform matrices), `gen_dx_bx`, `QuickCumsum`/`cumsum_trick`, `SimpleLoss`, `get_batch_iou`, `get_val_info`, nuScenes map plotting.

### Tensor-shape conventions

Batches carry a camera dimension: most tensors are `(B, N, ...)` where N is the number of cameras. The frustum/voxel pipeline expands to `(B, N, D, fH, fW, ...)`. Final BEV outputs are `(B, outC, 200, 200)` with the default grid. When modifying the geometry or pooling code, the inline comments document the exact shape at each line — keep them in sync with changes.

## Notes

- `model_pretrain/model525000.pt` is a checkpoint; `runs/` holds TensorBoard logs. `data/v1.0-mini.tgz` is the raw archive (extracted into `data/nuscenes/mini`).
- `script.py` is a scratch file for experimenting with the cumsum rank trick, not part of the pipeline.
- The original loss uses `.cuda(gpuid)`; in this fork `train.py` runs it on CPU-friendly device (the `.cuda()` call is commented out) so training works with `gpuid=-1`.
