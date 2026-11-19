# Publication checks

Verified on 19 November 2026 before publishing this package:

- Fresh pinned upstream checkout plus the full patch reproduced all **33** recorded training files by SHA256.
- The unchanged partner PCA script matched its pinned SHA256.
- Regenerating selections on the actual Flowers COLMAP data in NumPy 2.1.2 reproduced **all 12 complete manifest records**, not only their image counts.
- The portable launcher's **12 training argument lists** matched the executed commands, apart from input/output locations.
- All original image hashes, geometry hashes, candidate/test lists and sparse-subset exclusions passed.
- The actual training loader passed **all 12 splits**, including native fisheye cameras, 54,865 points and the same 130-camera normalization radius.
- CUDA/simple-knn and LPIPS-VGG checks passed on the idle lab GPU. No new optimization run was started.
- Portable postprocessing ran against the completed PCA-12, FPS-12 and Coverage-12 outputs in an isolated folder.
  All three resulting per-test-image CSVs matched the original CSVs, and nine matched comparison panels were generated.
- Five CPU tests passed: full-3D angular selection, direction-aware Coverage behavior, frozen split exclusions,
  fixed command settings, and preservation/integrity checks for interrupted and completed outputs.

The machine-readable record is `provenance/publication_validation.json`.
The exact executed controllers are retained under `reference_execution/` for inspection.
The portable launcher supplies configurable paths, generic scene/test counts and updated matched-image reporting.
The training engine and selection algorithm used for the reported measurements are unchanged.

These checks do **not** constitute another 12-run GPU training study or completed Chairs/Globe experiments.
The measured results in `results/` come from the completed Flowers runs, not synthetic tests.
Matching the environment and frozen selections improves reproducibility; bit-for-bit GPU outputs are not guaranteed on different hardware.
