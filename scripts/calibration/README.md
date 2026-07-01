# Eye-in-Hand Calibration Toolkit

Calibrate the rigid transform between an Intel RealSense camera and a Pico XR
controller, when the camera is rigidly mounted on the controller ("eye in hand")
and observes a static chessboard target.

This is the setup used elsewhere in this repo when a controller-mounted camera must
share a frame with the controller pose (e.g. projecting camera observations into the
VR world frame). The toolkit is self-contained and only depends on packages the
project already ships (`numpy`, `opencv-python`, `pyrealsense2`, `xrobotoolkit_sdk`).

## The math, in one line

For every sample the following identity must hold (the target is fixed in the world):

```
X_T^B  =  X_G^B  ·  X_C^G  ·  X_T^C
```

| Symbol | Meaning | Source |
|--------|---------|--------|
| `X_G^B` | gripper (= Pico controller) pose in the world frame | `XrClient.get_pose_by_name` |
| `X_C^G` | **camera pose in the gripper frame (the unknown we solve for)** | `cv2.calibrateHandEye` |
| `X_T^C` | chessboard target pose in the camera frame | `cv2.solvePnP` |
| `X_T^B` | target pose in the world frame (must be constant across all samples) | derived, used for validation |

## Prerequisites

- **Hardware**
  - A printed chessboard of known square size, glued rigidly to a wall/table (it must
    not move for the entire session).
  - A RealSense camera solidly strapped/3D-printed to a Pico controller. Any slip
    between camera and controller invalidates the result.
- **Software** — the project conda env already provides everything:
  - `xrobotoolkit_sdk` (run the XRoboToolkit PC Service before launching)
  - `pyrealsense2`, `opencv-python`, `numpy`
- **Working directory** — run every script from the **repo root** so the
  `xrobotoolkit_teleop` package and the sibling `handeye_common` module import
  correctly:
  ```bash
  python scripts/calibration/handeye_capture.py ...
  ```

### Ubuntu setup with `uv`

On the Ubuntu machine, clone the calibration branch and run the calibration-focused
setup script from the repo root:

```bash
git clone -b feat/hand-eye-calibration https://github.com/linshenghou/XRoboToolkit-Teleop-Sample-Python.git
cd XRoboToolkit-Teleop-Sample-Python

# Install uv first if needed:
# curl -LsSf https://astral.sh/uv/install.sh | sh
# source "$HOME/.local/bin/env"

bash scripts/calibration/setup_uv_calibration_ubuntu.sh --system-packages
source .venv-calibration/bin/activate
```

The script installs the minimal calibration environment (`numpy`, `opencv-python`,
`pyrealsense2`, `xrobotoolkit_sdk`, and this repo editable with `--no-deps`). It does
not install the full simulation/hardware dependency set unless you run the main project
setup separately.

## Workflow

The four scripts map onto the standard hand-eye calibration SOP. `handeye_common.py`
holds shared utilities and is not run directly.

### Phase 1 — Physical setup

1. Fix the chessboard so it cannot move.
2. Rigidly mount the RealSense on the Pico controller.
3. Note the chessboard **inner-corner count** (`COLS ROWS`) and the **square edge
   length in metres**. Inner corners = squares − 1 per axis (e.g. a 12×9 square board
   → `--pattern 11 8`).

### Phase 2 — Axis sanity check (optional but recommended)

Discover how the Pico's native axes map to the physical world and produce an
`--axis-remap` matrix. Run it before the first capture:

```bash
python scripts/calibration/handeye_sanity_check.py --controller right_controller
```

- The live HUD shows `det(R)` — this should read `1.0000` at all times (a quaternion
  → matrix conversion is always a proper rotation). That confirms the Pico delivers
  valid right-handed rotations.
- Follow the on-screen protocol: hold at REST, then move ~15 cm along your chosen
  `+X`, `+Y`, `+Z` (press `SPACE` at each). Pick a **right-handed** triple (e.g.
  X=right, Y=up, Z=backward), otherwise the script rejects the result.
- On success it prints a ready-to-paste `--axis-remap ...` line and saves
  `calibration_data/axis_remap.json`.

> **Note:** `axis_remap` is never required for the solver to work — `calibrateHandEye`
> only needs internally consistent proper rotations. The remap just re-expresses
> `X_C^G` in a physically meaningful base frame. The one hard rule is `det(M) = +1`,
> which the script enforces.

### Phase 3 — Collect samples

```bash
python scripts/calibration/handeye_capture.py \
    --record-dir calibration_data/exp01 \
    --controller right_controller \
    --serial <realsense-serial> \
    --square-size 0.025 --pattern 11 8 \
    --pico-trigger auto
```

- A RealSense pipeline runs in a dedicated **warm-started child process** that opens
  the camera once, discards the first ~30 frames (auto-exposure settling), and then
  streams into shared memory. The main loop always sees the freshest frame with zero
  cold-start latency, which removes the camera (30 Hz) vs Pico (200 Hz) timestamp
  skew.
- **Hold still for ~0.6 s** before capturing (enforced); then press `SPACE` **or**
  pull the Pico **trigger**. The trigger fires only on the rising edge (hysteresis),
  so holding it does not auto-repeat.
- Aim for **20–30 samples** with large, varied rotations (roll/pitch/yaw all moved) —
  rotational excitation is what constrains the orientation of `X_C^G`.
- Keys: `SPACE`/trigger = capture, `d` = undo last, `q`/ESC = quit & save.

Outputs written to `--record-dir`:

```
calibration_data/exp01/
├── dataset.json          # intrinsics (factory), chessboard params, axis_remap
├── sample_000.png        # color image per sample
├── sample_000.json       # raw Pico pose [x,y,z,qx,qy,qz,qw] + timestamps
├── sample_001.png
└── ...
```

The raw Pico pose (pre-remap) is stored on purpose: you can change `axis_remap` in
`dataset.json` and re-run the solver without re-collecting data.

### Phase 4 — Solve for `X_C^G`

```bash
python scripts/calibration/handeye_solve.py --record-dir calibration_data/exp01 --method PARK
```

- Re-runs `solvePnP` per image to get every `X_T^C`, converts raw Pico poses to
  `X_G^B` using the dataset's `axis_remap`, then calls `cv2.calibrateHandEye`.
- Runs **all five methods** (TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS) and prints their
  spread so you can pick the most stable; `--method` selects which one is persisted.
- Warns when mean pairwise gripper rotation is below `--min-rot-deg` (default 30°) — a
  sign of poor rotational excitation.

Writes `handeye_result.json` (the chosen `X_C^G` + full diagnostics).

### Phase 5 — Cross-validate

```bash
python scripts/calibration/handeye_validate.py --record-dir calibration_data/exp01
```

Two complementary checks on the governing identity:

1. **Consistency** — compute `X_T^B_i = X_G^B_i · X_C^G · X_T^C_i` for every sample;
   they should all collapse to one transform. Reports translation RMS (mm) and
   rotation RMS (deg). Rule of thumb: RMS t < 5 mm and RMS R < 1° is good.
2. **Leave-one-out** — drop each sample, re-solve on the rest, measure how much
   `X_C^G` moves. Samples whose removal swings the result are flagged as outlier
   candidates (delete them and re-run Phase 4).

## Scripts at a glance

| Script | Purpose |
|--------|---------|
| [`handeye_common.py`](handeye_common.py) | Shared pose math (no scipy), chessboard detection, dataset I/O |
| [`handeye_sanity_check.py`](handeye_sanity_check.py) | Phase 2 — discover Pico axis mapping → `axis_remap` |
| [`handeye_capture.py`](handeye_capture.py) | Phase 3 — warm-start mp.Process capture loop |
| [`handeye_solve.py`](handeye_solve.py) | Phase 4 — solve `X_C^G` with 5 methods + diagnostics |
| [`handeye_validate.py`](handeye_validate.py) | Phase 5 — consistency + leave-one-out validation |

## Troubleshooting

- **Capture hangs at "Warming up camera…"** — wrong `--serial`, or another process
  holds the camera. Run `scripts/visualization/rs_cam_streaming.py` to confirm the
  device and its serial.
- **`XrClient()` raises on startup** — the XRoboToolkit PC Service is not running, or
  `xrobotoolkit_sdk` is not in the active env.
- **Chessboard never detected** — `--pattern` counts **inner corners**, not squares;
  also ensure good lighting and that the whole board is in frame and roughly flat to
  the camera.
- **`det(M) < 0` in the sanity check** — the physical `(X, Y, Z)` triple you moved
  along is left-handed; flip one axis and redo.
- **Validation shows large scatter** — collect again with bigger rotations and verify
  the camera is not slipping on the controller.

## Notation convention

Transforms follow the OpenCV "pose of B in A" reading: `X_C^G` (= `T_cam2gripper`)
is the **camera's pose expressed in the gripper (Pico) frame**. The same 4×4 also
maps camera-frame points into the gripper frame. Keep this convention when composing
`X_C^G` with other transforms downstream.
