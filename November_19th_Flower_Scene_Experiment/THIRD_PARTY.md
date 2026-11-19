# Source and attribution

- **DirectFisheye-GS:** [upstream repository](https://github.com/Metaverse-AI-Lab-THU/DirectFisheye-GS), commit
  `db9b3a25bff79a59f02ce90f5bd235cb59c03aaa`. The full patch and file hashes are under `provenance/`.
  The original research-use license is retained in `licenses/DirectFisheye_Gaussian_Splatting_LICENSE.md`.
  Bootstrap downloads the upstream CUDA sources and their original license notices; it does not replace those terms.
- **PCA implementation:** Sofia's [TRDP_fisheyeGS repository](https://github.com/sofiacostap/TRDP_fisheyeGS), commit
  `39c4af1ee031507a79c7b0426c94f1f0e3d9ce6e`, file `scripts/create_uniform_pose_manifest.py`.
  It is downloaded unchanged from that pinned source and verified by SHA256, rather than republished as our own implementation.
- **Angular FPS inspiration:** Qi et al., [PointNet++](https://arxiv.org/abs/1706.02413), 2017.
  We adapted the sampling idea to 3D camera direction angles with a widest-pair start. No PointNet++ network code is used.
- **Coverage inspiration:** Chen et al., [Coverage Optimization for Camera View Selection](https://arxiv.org/abs/2604.05259), 2026.
  Our static COLMAP observation-and-direction objective is custom code, not the published COVER implementation or a reproduction of its reported results.
- **Dataset:** original [FishEyeNeRF archive](https://drive.google.com/file/d/1VhnpMUIKEak4TBpKY4H8vMJJe--FM2k5/view).
  Photographs are obtained from their original source and are not republished in this repository.
  The privately shared Flowers geometry bundle is the exact reconstruction and caches used by this study; it is not a trained model and is not publicly distributed here.

There is no blanket license here that overrides upstream software or dataset terms.
Retain the original notices and citations when reusing those components.
