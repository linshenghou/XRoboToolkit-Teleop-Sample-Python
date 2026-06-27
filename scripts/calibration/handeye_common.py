"""Shared utilities for Eye-in-Hand hand-eye calibration (RealSense + Pico controller).

Notation (matches the derivation in the design doc):

    X_G^B : gripper (= Pico controller) pose in the base / world frame.   [from XrClient]
    X_C^G : camera pose in the gripper frame.                             [UNKNOWN -- what we solve]
    X_T^C : target (chessboard) pose in the camera frame.                 [from cv2.solvePnP]

Identity that must hold for every sample once X_C^G is known (target is static):

    X_T^B = X_G^B @ X_C^G @ X_T^C

Only numpy + opencv are used here on purpose, so this module adds no new dependency
beyond what the project already ships (note: the repo's geometry.py uses the
(w, x, y, z) quaternion ordering, whereas XrClient returns [x, y, z, qx, qy, qz, qw]
i.e. scalar-last -- this module always states the ordering explicitly to avoid bugs).
"""

import json
from pathlib import Path

import cv2
import numpy as np

# =====================================================================================
# Pose conversions (pure numpy, no scipy)
# =====================================================================================


def quat_xyzw_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion [qx, qy, qz, qw] (scalar-last, XrClient convention) -> 3x3 rotation."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"Expected quaternion of shape (4,), got {q.shape}.")
    qx, qy, qz, qw = q
    # enforce unit norm (XrClient quaternions are unit in practice; guard anyway)
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> quaternion [qx, qy, qz, qw] (scalar-last, Shepperd's method)."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def pico_pose_to_matrix(pose_xyzq: np.ndarray, axis_remap: np.ndarray | None = None) -> np.ndarray:
    """Pico pose [x, y, z, qx, qy, qz, qw] (XrClient) -> 4x4 SE(3) = X_G^B.

    Args:
        pose_xyzq: length-7 array as returned by ``XrClient.get_pose_by_name``.
        axis_remap: optional constant 3x3 applied as ``R_base <- R_remap @ R_native``
            and ``t_base <- R_remap @ t_native``. Use it to fold in the result of the
            Phase-2 sanity check (Pico's native frame to the desired right-handed base
            frame). Leave None if the axes are already consistent -- the hand-eye solver
            only needs internal consistency, so this never affects solvability, only the
            physical interpretation of the returned X_C^G.

    Returns:
        4x4 homogeneous transform (float64).
    """
    pose = np.asarray(pose_xyzq, dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError(f"Expected Pico pose of shape (7,), got {pose.shape}.")
    t = pose[:3].copy()
    R = quat_xyzw_to_rotmat(pose[3:])  # pose[3:] == [qx, qy, qz, qw]
    if axis_remap is not None:
        M = np.asarray(axis_remap, dtype=np.float64)
        R = M @ R
        t = M @ t
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inverse_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 SE(3) transform (closed form, faster & more stable than np.linalg.inv)."""
    out = np.eye(4, dtype=np.float64)
    R = T[:3, :3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def rotmat_distance_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic rotation distance between two 3x3 matrices, in degrees."""
    R_diff = np.asarray(R1, dtype=np.float64).T @ np.asarray(R2, dtype=np.float64)
    cos_angle = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def is_right_handed(R: np.ndarray, tol: float = 1e-6) -> bool:
    """True if a 3x3 is a proper rotation (det == +1). Handy for the Phase-2 sanity check."""
    R = np.asarray(R, dtype=np.float64)
    return abs(np.linalg.det(R) - 1.0) < tol and np.allclose(R @ R.T, np.eye(3), atol=1e-6)


# =====================================================================================
# Chessboard detection + solvePnP  (gives X_T^C)
# =====================================================================================


def make_object_points(square_size_m: float, pattern_size: tuple[int, int]) -> np.ndarray:
    """Inner-corner coordinates of the chessboard in the target frame (planar, Z=0).

    Args:
        square_size_m: physical edge length of one chessboard square, in metres.
        pattern_size: ``(cols, rows)`` = number of *inner* corners, identical to the
            ``patternSize`` argument of ``cv2.findChessboardCorners``.

    Returns:
        (cols*rows, 3) float64 array.
    """
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), dtype=np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * square_size_m


def detect_chessboard(image_gray: np.ndarray, pattern_size: tuple[int, int]):
    """Detect & sub-pixel-refine inner corners.

    Returns:
        (ok, corners) where corners is (N, 1, 2) float32 ready for solvePnP, or
        (False, None) if not found.
    """
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(image_gray, pattern_size, flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(image_gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def solve_target_to_camera(corners_px, object_points, K, dist) -> np.ndarray | None:
    """Run solvePnP -> X_T^C (target in camera optical frame) as a 4x4. None on failure."""
    ok, rvec, tvec = cv2.solvePnP(object_points, corners_px, K, dist)
    if not ok:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


# =====================================================================================
# Dataset I/O
# =====================================================================================


def write_dataset_config(
    record_dir: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    square_size_m: float,
    pattern_size: tuple[int, int],
    axis_remap: np.ndarray | None,
    image_size_wh: tuple[int, int],
) -> Path:
    """Write the per-dataset config (intrinsics + chessboard params) as dataset.json."""
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=np.float64).ravel().tolist(),
        "square_size_m": float(square_size_m),
        "pattern_size": [int(pattern_size[0]), int(pattern_size[1])],
        "axis_remap": None if axis_remap is None else np.asarray(axis_remap).tolist(),
        "image_size_wh": [int(image_size_wh[0]), int(image_size_wh[1])],
    }
    path = record_dir / "dataset.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def read_dataset_config(record_dir: Path) -> dict:
    """Inverse of :func:`write_dataset_config`. Returns numpy-typed fields."""
    cfg = json.loads((Path(record_dir) / "dataset.json").read_text())
    cfg["camera_matrix"] = np.array(cfg["camera_matrix"], dtype=np.float64)
    cfg["dist_coeffs"] = np.array(cfg["dist_coeffs"], dtype=np.float64)
    cfg["pattern_size"] = tuple(cfg["pattern_size"])
    cfg["axis_remap"] = None if cfg["axis_remap"] is None else np.array(cfg["axis_remap"], dtype=np.float64)
    cfg["image_size_wh"] = tuple(cfg["image_size_wh"])
    return cfg


def write_sample(record_dir: Path, index: int, color_bgr: np.ndarray, pico_pose_xyzq: np.ndarray, meta: dict) -> Path:
    """Persist one sample: image .png + .json carrying the *raw* Pico pose (pre-axis-remap).

    Storing the raw pose (not the matrix) lets you re-run the solver with a different
    ``axis_remap`` from the Phase-2 sanity check without re-collecting data.
    """
    record_dir = Path(record_dir)
    img_path = record_dir / f"sample_{index:03d}.png"
    cv2.imwrite(str(img_path), color_bgr)
    payload = {
        "image": img_path.name,
        "pico_pose_xyzq": [float(v) for v in np.asarray(pico_pose_xyzq).ravel()],
    }
    payload.update(meta or {})
    json_path = record_dir / f"sample_{index:03d}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    return json_path


def list_samples(record_dir: Path) -> list[Path]:
    """Sorted list of sample_XXX.json paths in a record dir."""
    return sorted(Path(record_dir).glob("sample_*.json"))


def load_sample(json_path: Path) -> dict:
    """Load one sample, returning the raw Pico pose as a numpy array."""
    d = json.loads(Path(json_path).read_text())
    d["pico_pose_xyzq"] = np.array(d["pico_pose_xyzq"], dtype=np.float64)
    return d


def save_handeye_result(out_path: Path, method: str, T_cam2gripper: np.ndarray, diagnostics: dict) -> Path:
    """Write the solved X_C^G + diagnostics as handeye_result.json."""
    out = {
        "method": method,
        "T_cam2gripper": T_cam2gripper.tolist(),
        "t_cam_in_gripper_m": T_cam2gripper[:3, 3].tolist(),
        "quat_xyzw_cam_in_gripper": rotmat_to_quat_xyzw(T_cam2gripper[:3, :3]).tolist(),
        "diagnostics": diagnostics,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    return out_path


def load_handeye_result(path: Path) -> tuple[str, np.ndarray, dict]:
    """Load a handeye_result.json -> (method, T_cam2gripper 4x4, diagnostics)."""
    d = json.loads(Path(path).read_text())
    return d["method"], np.array(d["T_cam2gripper"], dtype=np.float64), d.get("diagnostics", {})
