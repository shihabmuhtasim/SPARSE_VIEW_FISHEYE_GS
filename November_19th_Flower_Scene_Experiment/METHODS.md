# How training photos are selected

All methods start from the same eligible training cameras. Tests are set aside first.
COLMAP supplies camera poses and sparse 3D scene points: estimated positions of recognizable image features,
with records of which photos observed each feature. These points are not a complete surface model.

## PCA angle baseline

1. Find the average position of the candidate cameras.
2. Fit a flat plane to their positions using PCA, which finds the two directions along which the positions vary most.
3. Project the positions onto that plane and measure their angles around the average position.
4. Place equally spaced angle targets around that point: 30° apart for 12 targets, 15° for 24, and so on.
5. In target order, choose the closest unused camera angle.

This invokes Sofia's unchanged `create_uniform_pose_manifest.py`. The exact commit and SHA256 are recorded.
Each training count is selected separately, so the PCA subsets need not be nested.
This baseline uses a fitted plane, not our earlier fixed world-XY projection.
The **targets** are evenly spaced in that plane; the available cameras are only approximate matches.
It does not enforce equal physical distances or complete 360° scene coverage.

Camera centres are calculated as `C = -R.T @ t`. PCA uses SVD of `C - mean(C)`;
the first two right-singular vectors define the plane. Position angles use `atan2` of the two projected coordinates.
Targets are the centres of equal circular bins from −π to π. Ties use the original lexicographic filename order.

## 3D angular farthest point sampling

1. Take the middle X, middle Y and middle Z values of all frozen reconstructed points as one fixed reference.
2. Draw a direction from this reference to every candidate camera, using all three coordinates.
3. Start with the pair of cameras whose directions have the largest angle between them.
4. For each remaining camera, find its smallest angle to any selected camera.
5. Add the camera with the largest such angle, and repeat until 96 are selected.

Use the first 12, 24, 48 or 96 cameras in that sequence. This spreads **3D position directions from the reference**;
it does not enforce physical camera spacing or directly compare where the lenses point.
Directions are normalized `(C - reference)` vectors. Their angle is `acos(clip(dot(u, v), -1, 1))`.
The first maximum in lexicographic camera order breaks ties.

The inspiration is the farthest point sampling idea used in [PointNet++](https://arxiv.org/abs/1706.02413).
We use angles between camera directions and a widest-pair start. We do not run its neural network
or claim this angular variant is PointNet++'s original sampling implementation.

## Coverage adaptation

We use records of visible scene details and their viewing directions, rather than a single scene centre.
For Flowers, 51,695 of the 54,865 points are eligible: each is observed by at least two candidate training cameras.
Test-image observations are removed before this check.

1. Start with no selected photos.
2. For each candidate, ask how much it would improve representation of the recorded scene details and directions.
3. Select the candidate with the largest improvement.
4. Keep, for each recorded observation, the best match supplied by any selected photo.
5. Repeat to 96 photos. Use prefixes for 12, 24, 48 and 96.

**What a direction means here:** for one scene point, draw a line from that point to each camera that observed it.
The algorithm compares these 3D lines. This differs from the FPS directions, which all start at one common reference.

### A small example

Suppose one flower detail was recorded by A and B from nearly the same direction, and by C from 60° away.
After selecting A, B mostly repeats its direction. C can better represent the direction still missing from the selection.
That can make C useful even though it adds **zero new point IDs**. The graph counting new points does not count that directional improvement.

### Exact score needed to reproduce it

For one point, compare a selected observation direction with every candidate observation direction of that same point.
Similarity is `exp(-0.5 * (angle_degrees / 30)^2)`; a camera that did not observe the point contributes zero.
Keep the largest similarity among selected cameras for each recorded direction, then average within the point.
Sum these point averages. Each eligible point therefore has equal total weight, regardless of how many photos observed it.
Choose the unused photo that increases this sum the most. Computation uses float32; the first maximum in filename order wins.
The 30° value is a smooth falloff scale, not a hard cutoff or a minimum spacing rule.

For the example with directions A=0°, B=0° and C=60°, selecting A represents that point with
`(1 + 1 + exp(-2)) / 3 ≈ 0.712`. Adding B changes nothing; adding C makes the score 1, an improvement of about 0.288.
For real selection, gains are summed over all eligible points; this one-point example does not determine the whole selection.

This is a **fixed, deterministic, offline COLMAP adaptation** inspired by
[Coverage Optimization for Camera View Selection](https://arxiv.org/abs/2604.05259).
Published COVER integrates active selection with reconstruction training. We did not use its code or claim to reproduce that algorithm.
Our photos are selected before DirectFisheye training starts. No agent or learned AI model decides the selection.

## Comparability and limits

- Only the training-photo selection changes. The model settings, geometry, candidate pool and test list stay fixed.
- PCA uses the camera mean; FPS uses the median reconstructed-point reference; Coverage uses each individual scene point.
- FPS and Coverage use nested prefixes. PCA repeats selection separately for each count.
- Point coverage percentages count sparse feature IDs, not the fraction of the actual scene surface reconstructed.
- The reconstruction and camera associations use full-scene geometry, including test-assisted geometry. This limitation applies to every method.
- The historical point-only 20-view experiment is separate. Every Coverage run in this coordinated 12/24/48/96 study uses the same directional rule.
