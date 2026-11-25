# Segmentation — Multi-Order RANSAC + DBSCAN

CLI-first Python package (`seg3d`) that automates the Open3D multi-order plane RANSAC and residual DBSCAN workflow previously used as a Spyder cell script.

Pipeline:

```text
PLY / PCD / Gaussian training PLY
  → center, statistical outlier filter, voxel downsample, normals
  → iterative RANSAC planes
  → DBSCAN on non-planar residuals
  → labeled PLY exports + summary.json
```

## Install

Python 3.10+ recommended (Open3D wheels).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e .
```

## Run

```bash
seg3d run --input path/to/cloud.ply --out output/run1
```

Gaussian (Graphdeco) training PLYs are detected when `opacity`, `scale_*`, `rot_*`, and `f_dc_*` properties exist. Geometry uses XYZ; extra attributes are preserved on export when you **skip voxel downsample** (downsample merges points and drops 1:1 attributes):

```bash
seg3d run --input scene_gaussians.ply --out output/gs --skip-downsample --voxel 0 \
  --plane-dist 0.03 --max-planes 6 --dbscan-eps 0.08
```

Useful knobs (defaults match the historical Spyder / script values where applicable):

| Flag | Default | Meaning |
|---|---|---|
| `--outlier-nn` / `--outlier-std` | 16 / 2.0 | Statistical outlier filter |
| `--voxel` | 0.1 | Voxel downsample size (scene units) |
| `--skip-downsample` | off | Keep full resolution (keeps Gaussian attrs) |
| `--max-planes` | 6 | Sequential RANSAC plane count |
| `--plane-dist` | 0.02 | RANSAC inlier distance |
| `--min-inliers` | 50 | Skip / stop if plane too small |
| `--dbscan-eps` | 0.05 | Residual clustering radius |
| `--dbscan-min-points` | 5 | DBSCAN density |

Distances are in **input coordinate units** (often meters). Tune to scene scale.

## Outputs

| Path | Contents |
|---|---|
| `segmented.ply` | All working points colored by label |
| `planes/plane_XX.ply` | Each RANSAC plane |
| `clusters/cluster_XX.ply` | Each DBSCAN residual cluster |
| `noise.ply` | Unclustered residual points (if any) |
| `labels.npy` | Per-point integer labels (`-1` = noise) |
| `summary.json` | Plane equations, counts, paths |

## Legacy script

`Segmentation_Multi_Order_RANSAC.py` is the original interactive Spyder-style notebook script (hardcoded paths + Open3D GUI). Prefer `seg3d run` for automation.

## Related local work (DCMG3D)

- Spyder R&D: `MG3D Dropbox/Daniel Caban/R&D/Spyder/3DGS_1.py` (load/viz exploration)
- SplatGraph (Gaussian voxel research CLI): Downloads `splatgraph_spyder_starter` — complementary; not required to run `seg3d`
