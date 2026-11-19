# Sparse view native fisheye Gaussian splatting

Reproducible training-view selection experiments using DirectFisheye-GS.

## November 19 Flowers study

Start in [November 19th Flower Scene Experiment](November_19th_Flower_Scene_Experiment/README.md).

- Three methods: PCA angle, 3D angular farthest point sampling, and our Coverage adaptation.
- Four training counts: 12, 24, 48 and 96; the same 19 held-out test photos throughout.
- Includes runnable selection, training, evaluation and reporting code, exact split manifests, pinned source patches and measured results.
- For Chairs or Globe, follow the [teammate handoff](November_19th_Flower_Scene_Experiment/TEAM_HANDOFF.md).

The [results summary](November_19th_Flower_Scene_Experiment/results/README.md) contains all 12 test results.
The exact Flowers COLMAP input bundle is shared privately by the experiment owner;
the [setup instructions](November_19th_Flower_Scene_Experiment/README.md#obtain-the-flowers-inputs) explain how to install and verify it.
