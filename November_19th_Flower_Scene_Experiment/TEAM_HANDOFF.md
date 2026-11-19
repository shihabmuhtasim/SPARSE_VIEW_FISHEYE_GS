# Repeating this study on Chairs or Globe

The main comparison is **three selectors × 12, 24, 48 and 96 training views**.
Choose the test split once for each scene, then keep it identical across all methods and counts.
Use the same pinned engine, patch, environment and settings as Flowers. Do not substitute the earlier fixed-XY selector.

## Prepare a scene

1. Put original native-fisheye photos in `images/` and that scene's complete existing COLMAP model in `sparse/0/`.
2. Keep the geometry fixed. Rebuilding COLMAP for each subset would introduce another experimental variable.
3. Create or verify the frozen full-view manifest before selecting sparse training sets:

```bash
export PARTNER_SCENE=chairs
export PARTNER_DATA=/path/to/scene_chairs
export PARTNER_MANIFESTS=/path/to/large_storage/chairs_manifests

python code/create_scene_manifest.py \
  --scene "$PARTNER_SCENE" --scene-dir "$PARTNER_DATA" \
  --output "$PARTNER_MANIFESTS/${PARTNER_SCENE}_fullview_split_v1.json"

# Downloads only the unchanged pinned PCA script, if not already present.
python code/bootstrap.py --partner-only
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python code/prepare_study.py \
  --scene-dir "$PARTNER_DATA" \
  --baseline-manifest "$PARTNER_MANIFESTS/${PARTNER_SCENE}_fullview_split_v1.json" \
  --output "$PARTNER_MANIFESTS"
```

If an agreed baseline already exists, add `--import-baseline /path/to/agreed.json` to `create_scene_manifest.py`.
It checks both image lists and COLMAP hashes. A mismatch stops the process; coordinate the inputs instead of silently changing the split.
The helper preserves the repository rule: sort original filenames lexicographically and hold out zero-based indices 0, 8, 16, ... .
It accepts an imported manifest containing an additional `project.ini` hash while preserving/checking the three binary model files needed here.

Scene-specific counts come from the reconstruction; do not force 19 tests or 130 candidates on another scene.
All three methods require at least 96 training candidates for this study. Use `globe` instead of `chairs` for the other scene.

## Run and collect

```bash
python code/run_experiments.py \
  --source "$DFGS_ENGINE" --scene-dir "$PARTNER_DATA" \
  --manifests "$PARTNER_MANIFESTS" --output /path/to/chairs_outputs \
  --archive /path/to/chairs_archive --validate-only

python code/start_background.py --name chairs-selection-study -- \
  --source "$DFGS_ENGINE" --scene-dir "$PARTNER_DATA" \
  --manifests "$PARTNER_MANIFESTS" --output /path/to/chairs_outputs \
  --archive /path/to/chairs_archive --restart-incomplete --prune-temporary
```

Check `Linger=yes` for disconnect survival, storage quotas and the GPU before leaving the queue.
Run the validation and launch commands from the same activated environment.
For a quota-limited archive add `--quota-root` and the account's actual soft limit as explained in the main README.

Return these materials for each scene:

- Frozen full-view and 12 sparse manifests; source, image and geometry hashes.
- Final train/test means, every test photo's PSNR/SSIM/LPIPS, all final test GT/renders and `view_order.json`.
- Runtime, settings, software inventory, training scalars and selection histories.
- Matched best/median/worst panels with the same test filenames across all three methods.

Compare mean **test** scores at the same training count. Train means at different counts average different image sets.
Different hardware or numerical libraries may produce small numerical differences despite the same seed.
This package's portability checks do not claim that a new Chairs/Globe training run has already been completed.
