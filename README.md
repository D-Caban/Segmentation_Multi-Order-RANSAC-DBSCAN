# Segmentation — Multi-Order RANSAC + selectable objects

CLI-first Python package (`seg3d`) for point-cloud / Gaussian-splat segmentation.

Two entry points:

| Command | Purpose |
|---|---|
| `seg3d run` | Classic multi-order plane RANSAC + residual DBSCAN |
| `seg3d selectables` (`objects`) | Structure planes + affinity-graph residual objects → **per-object PLYs** for LichtFeld |

```text
Gaussian / PLY
  → preprocess
  → RANSAC planes (structure)
  → kNN affinity graph on residuals (or adaptive DBSCAN)
  → merge tiny / cap object count
  → shape hints + coarse semantic tags
  → objects/*.ply + manifest.json
```

## Install

Python 3.10+ recommended (Open3D wheels).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Selectable objects (recommended for LichtFeld)

```bash
seg3d selectables \
  --input scene_gaussians.ply \
  --out output/selectables \
  --skip-downsample \
  --plane-dist 0.03 \
  --max-planes 6 \
  --min-object-points 200 \
  --max-objects 80
```

Gaussian (Graphdeco) training PLYs are detected when `opacity`, `scale_*`, `rot_*`, and `f_dc_*` exist. Use `--skip-downsample` so those attributes survive on each object PLY.

### Outputs (`selectables`)

| Path | Contents |
|---|---|
| `objects/structure_XX.ply` | Large RANSAC planes (architecture) |
| `objects/object_XXX.ply` | Residual object clusters (furniture-like, etc.) |
| `manifest.json` | id, label, shape, aabb, centroid, path, counts |
| `overview.ply` | Full working cloud colored by object |
| `labels.npy` | Per-point labels (optional; LichtFeld does **not** use this as groups) |

### Import into LichtFeld Studio

1. Open the export folder (e.g. `C:\Users\DanielCaban\Downloads\seg3d-selectables-faber\`).
2. Import **each** PLY under `objects\` as its **own model / layer**.
3. Select / hide / group models in the LichtFeld UI. Do not expect `labels.npy` alone to create selectable groups.

## Classic `seg3d run`

```bash
seg3d run --input path/to/cloud.ply --out output/run1
```

Useful knobs (defaults match historical Spyder / script values where applicable):

| Flag | Default | Meaning |
|---|---|---|
| `--outlier-nn` / `--outlier-std` | 16 / 2.0 | Statistical outlier filter |
| `--voxel` | 0.1 | Voxel downsample size (scene units) |
| `--skip-downsample` | off | Keep full resolution (keeps Gaussian attrs) |
| `--max-planes` | 6 | Sequential RANSAC plane count |
| `--plane-dist` | 0.02 | RANSAC inlier distance |
| `--min-inliers` | 50 | Skip / stop if plane too small |
| `--dbscan-eps` | 0.05 | Residual clustering radius (`run` only) |
| `--min-object-points` | 200 | Drop/merge tiny objects (`selectables`) |
| `--max-objects` | 80 | Cap residual objects (`selectables`) |
| `--method` | `graph` | `graph` affinity CC or adaptive `dbscan` |

Distances are in **input coordinate units** (often meters). Tune to scene scale.

## Legacy script

`Segmentation_Multi_Order_RANSAC.py` is the original interactive Spyder-style notebook script. Prefer `seg3d`.

## Related local work (DCMG3D)

- Spyder R&D: `MG3D Dropbox/Daniel Caban/R&D/Spyder/3DGS_1.py`
- SplatGraph: Downloads `splatgraph_spyder_starter` — affinity / scene-graph ideas reused conceptually
- LichtFeld Studio: separate install; load per-object PLYs as models
