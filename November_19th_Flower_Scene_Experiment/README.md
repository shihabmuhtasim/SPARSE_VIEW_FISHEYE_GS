# November 19th Flower Scene Experiment

This folder contains the code and frozen inputs for our coordinated Flowers study.
We trained DirectFisheye-GS using **12, 24, 48 and 96 photos**, selected in three ways:
**PCA angle**, **3D angular FPS** and **Coverage**. All 12 runs evaluated the same 19 test photos.
The queue started on 18 November and finished on 19 November 2026.

**Main result:** Coverage had the best mean test PSNR and SSIM, and the lowest LPIPS, at each training count.
These are results for one scene and seed 0, not a claim that one method wins on every image or scene.

## Read first

- [Measured results and plots](results/README.md)
- [How the three selectors work](METHODS.md)
- [How to repeat the study on Chairs or Globe](TEAM_HANDOFF.md)
- [What was validated before publication](VALIDATION.md)

## Fixed experiment settings

| Item | Setting |
|---|---|
| Images | Original FishEyeNeRF Flowers: 149 photos |
| Candidate pool | 130 training candidates |
| Tests | Same frozen 19 photos in every run |
| Training counts | 12, 24, 48, 96 |
| Methods | PCA angle, 3D angular FPS, Coverage adaptation |
| Training | 30,000 iterations, batch size 2, seed 0, CVO `recip` |
| Final model | Default EMA, iterations 20,001–30,000, decay 0.999 |
| Images in training | Native fisheye; no undistortion; default working resolution about 1600 × 899 |
| Geometry | Same full COLMAP model and 54,865 initialization points in every run |
| Coverage points | 51,695 points observed by at least two eligible training cameras |
| Evaluation | Final PSNR, SSIM and LPIPS-VGG on every test photo; final train metrics also retained |
| Hardware used | NVIDIA RTX 4060 8 GB; Python 3.12.11; PyTorch 2.8.0+cu126; NumPy 2.1.2 |

**Geometry limitation:** the COLMAP reconstruction was made using all 149 images, including tests.
Test photos are excluded from training updates, and their observation tracks are excluded from Coverage selection.
This compares image selection with known geometry; it is not a training-only sparse COLMAP experiment.
Do not describe these results as a completely test-independent reconstruction pipeline.

## Set up the training engine

Use a Linux computer with an NVIDIA GPU and a compatible CUDA compiler. The recorded setup used CUDA 12.6.
The PyTorch package supplies CUDA runtime libraries, but does not supply the `nvcc` compiler needed to build the extensions.
Check `nvidia-smi` and `nvcc --version` before installing. Existing working DFGS environments can be reused.

```bash
git clone https://github.com/shihabmuhtasim/SPARSE_VIEW_FISHEYE_GS.git
cd SPARSE_VIEW_FISHEYE_GS/November_19th_Flower_Scene_Experiment

# Use your lab/scratch storage, not a small home quota. Replace these example paths.
export DFGS_ENGINE=/path/to/large_storage/DirectFisheye-GS-Nov19
export FLOWERS_DATA=/path/to/large_storage/scene_flowers
export FLOWERS_OUTPUT=/path/to/local_disk/flowers_nov19_outputs
export FLOWERS_ARCHIVE=/path/to/persistent_storage/flowers_nov19_archive

conda env create -f environment.yml
conda activate DFGS-Nov19
python code/bootstrap.py --destination "$DFGS_ENGINE"
python -m pip install --no-build-isolation -e "$DFGS_ENGINE/submodules/diff-gaussian-rasterization"
python -m pip install --no-build-isolation -e "$DFGS_ENGINE/submodules/simple-knn"
```

Bootstrap checks out DirectFisheye commit `db9b3a25bff79a59f02ce90f5bd235cb59c03aaa`, applies the full training patch,
and verifies all 33 recorded training files. It also downloads Sofia's unchanged, hash-verified PCA script
at commit `39c4af1ee031507a79c7b0426c94f1f0e3d9ce6e`. No PointNet++ neural network or COVER package is installed.
Keep `provenance/environment_packages.txt` as the full inventory of the environment that produced the results.

## Obtain the Flowers inputs

1. Obtain the [original FishEyeNeRF archive](https://drive.google.com/file/d/1VhnpMUIKEak4TBpKY4H8vMJJe--FM2k5/view).
2. Extract the **149 original JPGs from `FishEyeNeRF/flowers/images/`** into `$FLOWERS_DATA/images/`.
   Do not use the smaller `flowers/train/rgb/` copies. Do not rename, resize or undistort the files.
3. Obtain `flowers_frozen_colmap_v1.tar.gz` privately from Shihab, then install the exact reconstruction used here:

```bash
python code/fetch_flowers_geometry.py --scene-dir "$FLOWERS_DATA" \
  --archive /path/to/shared/flowers_frozen_colmap_v1.tar.gz
```

The separate geometry bundle is about 16.4 MiB. It contains the three COLMAP binaries, the cached initialization PLY
and the camera-association table. It contains no photographs or trained model weights.
The helper verifies the archive and each file, and refuses to overwrite different existing geometry.
The raw geometry bundle is not publicly hosted with the code. You need this privately shared input to reproduce the exact Flowers geometry;
rerunning COLMAP on the source images can produce different geometry and different measured results.

Expected layout:

```text
scene_flowers/
  images/                  # 149 original JPGs
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin
    points3D.ply
    camera_associations_overlap_skip9.txt
```

Use `frozen/flowers/` directly for exact Flowers reproduction. It already contains the 12 selections,
the 130-candidate/19-test split, geometry for plotting and selection histories.
The image and COLMAP checksums prevent accidental use of different input data.

## Validate without starting training

```bash
python -m unittest discover -s tests -v
python code/run_experiments.py \
  --source "$DFGS_ENGINE" --scene-dir "$FLOWERS_DATA" \
  --manifests frozen/flowers --output "$FLOWERS_OUTPUT" \
  --archive "$FLOWERS_ARCHIVE" --validate-only
```

This checks source hashes, input images and all 12 splits, and writes `run_plan.json` with the exact commands.
It performs no GPU training. A full loader/CUDA/LPIPS check runs automatically before training begins.
The first LPIPS check may download the VGG16 and LPIPS weights into the output cache; allow about 0.6 GB for these.

## Run all 12 experiments in order

Use the same paths as validation. `--prune-temporary` removes only optimizer snapshots and training PNGs,
after the final model, all test renders, metrics and logs have a verified second copy.
If interrupted, `--restart-incomplete` preserves the unfinished attempt and restarts only that run at seed 0.
Finished runs are checked and skipped. This does not claim exact mid-iteration recovery.

```bash
python code/run_experiments.py \
  --source "$DFGS_ENGINE" --scene-dir "$FLOWERS_DATA" \
  --manifests frozen/flowers --output "$FLOWERS_OUTPUT" \
  --archive "$FLOWERS_ARCHIVE" --restart-incomplete --prune-temporary
```

Order: PCA 12→24→48→96, angular FPS 12→24→48→96, Coverage 12→24→48→96.
Training and evaluation run sequentially on GPU 0; use `--gpu` for another GPU index.
The launcher waits for other compute jobs and never terminates someone else's task.
Successful training, rendering and evaluation took **18.48 hours in total** for the recorded study, excluding interrupted attempts.

### Keep it running after SSH disconnects

On a Linux host with systemd user services, use this instead of starting the foreground command above:

```bash
# Follow your university's policy; this may require administrator permission.
loginctl enable-linger "$USER"
python code/start_background.py --name flowers-nov19 -- \
  --source "$DFGS_ENGINE" --scene-dir "$FLOWERS_DATA" \
  --manifests "$(pwd)/frozen/flowers" --output "$FLOWERS_OUTPUT" \
  --archive "$FLOWERS_ARCHIVE" --restart-incomplete --prune-temporary
```

The persistent user unit retries failures after 60 seconds and starts with the user's service manager.
It does not override laboratory shutdown or account policies. A powered-off machine cannot train.
Keep the code, environment and dataset paths available; a wiped scratch disk requires restoring those inputs.

### Storage limits and monitoring

The launcher requires 20 GiB free on the output disk and 3 GiB on the archive disk.
For a quota-limited persistent area, also pass `--quota-root /path/to/your/workspace --quota-gib 45`.
This keeps 3 GiB of quota headroom. Filesystem free space alone does not measure a personal quota.
The 45 GiB value is an example from our account, not a universal university allowance.

```bash
watch -n 10 cat "$FLOWERS_OUTPUT/status.json"
tail -F "$FLOWERS_OUTPUT/runs/pca_angle_12/train_eval.log"
journalctl --user -u flowers-nov19.service -f
```

Completion is `state: complete` with `ALL 12 EXPERIMENTS HAVE BEEN COMPLETED`.
For failures, read `status.json`, `preflight.log` and the current run's log before retrying.
To stop the background queue: `systemctl --user stop flowers-nov19.service`.

## Outputs

Each run retains its command, timing, final EMA model, final train/test metric JSON,
per-test-image CSV, all test GT/predictions, filename mapping, TensorBoard events and training scalar CSVs.
`summary/` contains overall tables and matched image panels.
For each training count and each method's PSNR ranking, panels show best, median and worst photos
with **the same filename across GT, PCA, angular FPS and Coverage**.
For 19 tests the median is the 10th sorted image. These examples are for presentation, not tuning.

Training loss is recorded during optimization. Checkpoint PSNR is measured at 7,000 and 30,000;
checkpoint train PSNR uses sampled train views. Final train metrics use every selected training photo.
SSIM and LPIPS are final evaluations, not measurements at every training iteration.
Metrics come from the rendering tensors before PNG compression; do not substitute scores recomputed from display images.

`results/` contains the measured November results, including all 228 test records.
Raw photographs, caches, optimizer checkpoints and the large Word report are kept outside Git history.
This package does not start another experiment merely by cloning it.

## Source and attribution

The executed controller snapshot is in `reference_execution/`; `code/` supplies the portable launcher.
See [THIRD_PARTY.md](THIRD_PARTY.md) for source pins, licenses and references.
The static Coverage adaptation is our implementation inspired by COVER, not its released training-time implementation.
Earlier 10/20-view experiments are not relabelled or included in these 12 measurements.
