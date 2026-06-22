![Libor](./media/libor.png)

# Libor — Scene-agnostic ALS boresight self-calibration

**Libor** is a lightweight Python implementation of LiDAR boresight self-calibration from
**point-to-point correspondences**. It estimates the boresight misalignment angles
between an airborne laser scanner and its inertial navigation system using a
**Gauss–Helmert (GH) adjustment** that takes the post-processed INS/GNSS trajectory as
observations without requiring planar surfaces, special calibration flight patterns, nor
dedicated calibration site.

The correspondences are extracted with [**LiMatch**](https://doi.org/10.1016/j.isprsjprs.2025.08.011)
(Brun et al., 2025) directly from overlapping flight strips, which makes the approach
*scene-agnostic*: it works over buildings, mixed scenes, and natural terrain alike,
as long as the scene is not completely flat, monotone, or laser-absorbing.

This repository accompanies the paper:

> **Scene-agnostic ALS boresight self-calibration**
> A. Brun and J. Skaloud, Environmental Sensing and Observation Laboratory (ESO),
> EPFL. *Preprint, 2026.*
> 📄 arXiv: _[link to be added]_

Libor implements the **lightweight GH** formulation. The rigorous **Dynamic Network (DN)**
companion described in the paper, which estimates boresight jointly with the full
trajectory and inertial errors, is run through the [ODyN](https://odyn.epfl.ch)
factor-graph solver and is not part of this repository.

---

## Installation

Libor and its dependencies can be installed as follow:

```bash
git clone https://github.com/ESO-EPFL/libor.git
cd libor


pip install numpy scipy pyproj matplotlib reportlab cycler pyyaml
```

Tested with NumPy 1.x and 2.x.

---

## Running the example

A small example dataset (one ALS flight) ships in `data/`: one smoothed trajectory and three LiMatch correspondence files. It can be run as follow:

```bash
python libor.py -c configs/config.yml
```

Expected output (boresight recovered to a few thousands of a degree, mean residual
dropping from 12 m to a few centimetres):

```
| === Solution ===
| Reference boresight: [0.206 0.025 0.352] °
| Estimated boresight: [0.208 0.025 0.353] °
| Diff. from reference: [0.002 0.    0.001] °
| Mean residual = 0.181 m
| Time elapsed: 7.86 seconds

...
| a-posteriori sigma0 ≈ unity
| Condition number: 3.8e+01
```

The run writes a log, three SVG figures, and a **PDF calibration report** to the
output folder (`out/`).

To calibrate your own data, copy `configs/config.yml` and adapt the paths and parameters.

---

## Usage

```bash
python libor.py --cfg path/to/config.yml
```

Everything is driven by a single YAML configuration file. The pipeline:

1. loads the SBET trajectory and sets up a local ENU tangent plane;
2. loads and samples the point-to-point correspondences (`corrLoader`);
3. builds the GH `Model` and solves for `θ`;
4. marginalises outliers and re-solves;
5. computes posterior uncertainty and observability;
6. generates plots and a PDF report (`CalibrationStats`).

### Configuration reference

```yaml
# ============================================================================
# Libor — example configuration
# Runs out of the box on the bundled sample data.
#
#    python libor.py -c configs/config.yml
#
# ============================================================================

prj_name: 'example_Libor'

trj: "data/trajectory.out"    # Path to trajectory (SBET format)
p2p_folder: "data/Limatch/"                       # Path to Limatch correspondence folder

t_span: [289800.0, 291000.0]                  # Trajectory GPS seconds-of-week window (avoid loading all trajectory)

mount:
  leverArm: [[0.0], [0.0], [0.15]]    # Position of lidar sensor in body frame (m)
  R_s2b: [[0, 1, 0], [0, 0, 1], [1, 0, 0]]  # Rotation sensor to body
  initBor: [[0.0], [0.0], [0.0]]              # Initial boresight (deg, roll/pitch/yaw)

tp_latlon: [46., 7.]                        # Tangent-plane origin (decimal degrees), should be close to the correspondences

sampling:
  strategy: 'max'                             # 'max' | 'freq' | 'time_window' | (else: all)
  value: 5000

refBor: [[0.206], [0.025], [0.352]]          # reference calibration for comparison (deg)

sigmas:                                        # observation std-devs (meters and degrees)
  xy: 0.03
  z:  0.05
  rp: 0.003
  y:  0.005
  p2p: 0.15                                     # ~ 0.5 x GSD

output:
  folder: 'out/'
  fig_svg: true

info: "Example run on sample data."

```

### Sampling strategies

The correspondence set is usually far larger than needed; `corrLoader` thins it per file:

| `strategy`     | `value` means…                | extra keys      |
|----------------|-------------------------------|-----------------|
| `max`          | total correspondences kept (split evenly across files, random) | — |
| `freq`         | target sampling rate in **Hz** | — |
| `time_window`  | window length in **seconds** around each file's mean time | `max_per_file` |
| *anything else*| load **all** correspondences  | — |

---

## Input / output formats

**Trajectory (`trj`)** — Applanix **SBET** binary: records of 17 × `float64`
(col 0 = GPS seconds-of-week, cols 1–3 = lat/lon/alt in radians/m, cols 7–9 = roll/pitch,
heading = col 9 − col 10). Parsed by `loadSBET`.

**Correspondences (`p2p_folder/*`)** — comma-separated text, one correspondence per line:

```
t_i, t_j, vx_i, vy_i, vz_i, vx_j, vy_j, vz_j
```

where `t_i, t_j` are the two epochs (GPS sec-of-week) and `(vx, vy, vz)` are the laser
vectors in the **sensor frame** at each epoch. This is the standard
`LiDAR_p2p.txt` output of [LiMatch](https://doi.org/10.1016/j.isprsjprs.2025.08.011).

**Outputs** (in `output.folder`):

- `<prj_name>.log` — full run log
- `<prj_name>_calibration_report.pdf` — one-page summary (estimates, residuals, covariance, observability)
- `<prj_name>_residuals.svg`, `_bor_dif.svg`, `_corr_matrix.svg` — figures (when `fig_svg: true`)

---

## Repository structure

```
libor/
├── libor.py            # command-line entry point / pipeline orchestration
├── config.yml          # ready-to-run example (bundled data)
├── lib/
│   ├── map.py          # TangentPlane, Trajectory, Pose, loadSBET
│   ├── model.py        # corrLoader, Correspondence, Model (Gauss–Helmert solver)
│   ├── rotations.py    # rotation matrices, analytical derivatives, skew operators
│   └── stats.py        # CalibrationStats: plots + PDF report
├── data/               # bundled example (trajectory + correspondences)
└── media/libor.png
```

### Validation datasets

The paper validates Libor across **four operational ALS flights / five system
configurations** (the LAR flight is processed with both its navigation-grade and a
UAV-grade IMU):

| Flight / system | IMU grade |
|-----------------|-----------|
| Legacy 30 kHz heli, urban (LMS-Q240i) | tactical (LN200) |
| Riegl VQ-480, mixed scene (AIRINS) | navigation |
| same flight, APX-15 | UAV-grade |
| Riegl VQ-1560 II, mountain (IMU-57) | navigation |
| Leica TerrainMapper-3 (ISA-100C) | high-tactical |

The repository ships a single ready-to-run example, `configs/config.yml`, on the bundled
sample data. To reproduce the paper experiments, obtain the data from the portal below and
place your own configs under `configs/custom/`. Each experiment varies a few config keys:
the number of strips, a synthetic attitude offset on the trajectory, a coarse boresight
initialisation, or the correspondence `sampling` strategy.

---

## Data availability

A subset of the validation data (three of the four flights) is hosted at
[**addlidar.epfl.ch**](https://addlidar.epfl.ch). The correspondence-extraction pipeline
(LiMatch) and the Dynamic Network solver (ODyN) are available separately.

---

## Citing this work

If you use Libor, please cite the boresight self-calibration paper and the LiMatch
correspondence framework it builds on:

```bibtex
@article{brun2026libor,
  title   = {Scene-agnostic ALS boresight self-calibration},
  author  = {Brun, Aur{\'e}lien and Skaloud, Jan},
  year    = {2026},
  note    = {Preprint, arXiv:XXXX.XXXXX}
}

@article{brun2025limatch,
  title   = {Generalization of point-to-point matching for rigorous optimization
             in kinematic laser scanning},
  author  = {Brun, Aur{\'e}lien and Kolecki, Jakub and Xiao, Mingfei and Insolia, Luca
             and van der Zwan, Els and Guerrier, St{\'e}phane and Skaloud, Jan},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume  = {229},
  pages   = {107--121},
  year    = {2025},
  doi     = {10.1016/j.isprsjprs.2025.08.011}
}
```

The method also builds on the original rigorous boresight self-calibration of
Skaloud & Lichti (2006), *ISPRS J. Photogramm. Remote Sens.* **61**, 47–59,
[doi:10.1016/j.isprsjprs.2006.07.003](https://doi.org/10.1016/j.isprsjprs.2006.07.003).

---

## License

Released under the **MIT License** — see [`LICENSE`](./LICENSE). © 2025 ESO-EPFL.

---

## Acknowledgements

Developed at the Environmental Sensing and Observation Laboratory (ESO),
School of Architecture, Civil and Environmental Engineering (ENAC), EPFL.
