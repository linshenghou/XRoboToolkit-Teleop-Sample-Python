"""Phase 5: cross-validate the solved X_C^G  (answers the "is the result any good?" question).

The governing identity (target is rigidly fixed in the world) is:

    X_T^B = X_G^B @ X_C^G @ X_T^C

If X_C^G is correct, then for *every* sample the recovered world-frame target pose
``X_T^B_i`` must collapse to one and the same transform -- regardless of where the
hand was. Two complementary checks implement that idea:

  1. Consistency (cheap, always run):
        compute X_T^B_i for all i, take their mean, and report the per-sample scatter
        (translation in mm, rotation in deg). A tight cluster (say < ~5 mm and < ~1 deg)
        means the calibration reproduces a *static* target across very different hand
        poses -- the core sanity check. A wide cluster means X_C^G (or some sample) is off.

  2. Leave-one-out (the gold standard):
        drop each sample, re-solve X_C^G on the remaining N-1, measure how much it moved
        (translation mm / rotation deg) versus the full-data solution. A sample whose
        removal swings X_C^G a lot is an outlier -- flag it so you can delete it and
        re-run. If *all* leave-one-out deltas are large, the excitation is fundamentally
        poor and you must collect better data (do not trust the number).

Run example
    python scripts/calibration/handeye_validate.py --record-dir calibration_data/exp01
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from handeye_common import (
    detect_chessboard,
    inverse_se3,
    list_samples,
    load_handeye_result,
    load_sample,
    make_object_points,
    pico_pose_to_matrix,
    read_dataset_config,
    rotmat_distance_deg,
    solve_target_to_camera,
)

METHOD_FLAGS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-validate a solved X_C^G.")
    p.add_argument("--record-dir", type=Path, required=True)
    p.add_argument("--result", type=Path, default=None, help="handeye_result.json (default: <record-dir>/handeye_result.json).")
    p.add_argument("--outlier-t-mm", type=float, default=15.0, help="LOO translation delta above this flags a sample as an outlier.")
    p.add_argument("--outlier-rot-deg", type=float, default=2.0, help="LOO rotation delta above this flags a sample as an outlier.")
    return p.parse_args()


def load_pairs(record_dir: Path):
    """Return (cfg, list of (stem, X_G^B 4x4, X_T^C 4x4)) for every valid sample."""
    cfg = read_dataset_config(record_dir)
    objp = make_object_points(cfg["square_size_m"], cfg["pattern_size"])
    K, dist, axis_remap = cfg["camera_matrix"], cfg["dist_coeffs"], cfg["axis_remap"]
    pairs = []
    for jp in list_samples(record_dir):
        s = load_sample(jp)
        img = cv2.imread(str(record_dir / s["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        ok, corners = detect_chessboard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cfg["pattern_size"])
        if not ok:
            continue
        T_t2c = solve_target_to_camera(corners, objp, K, dist)
        if T_t2c is None:
            continue
        T_g2b = pico_pose_to_matrix(s["pico_pose_xyzq"], axis_remap=axis_remap)
        pairs.append((jp.stem, T_g2b, T_t2c))
    return cfg, pairs


def recover_target_in_base(T_g2b, T_c2g, T_t2c):
    """The identity: X_T^B = X_G^B @ X_C^G @ X_T^C."""
    return T_g2b @ T_c2g @ T_t2c


def mean_se3(Ts: list[np.ndarray]) -> np.ndarray:
    """Chordal mean of SE(3) transforms (translation arithmetic mean + re-orthonormalised rotation)."""
    t_mean = np.mean([T[:3, 3] for T in Ts], axis=0)
    U, _, Vt = np.linalg.svd(np.stack([T[:3, :3] for T in Ts]).sum(axis=0))
    R_mean = U @ Vt
    if np.linalg.det(R_mean) < 0:  # guard against a reflection
        U[:, -1] *= -1
        R_mean = U @ Vt
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R_mean
    out[:3, 3] = t_mean
    return out


def consistency_report(pairs, T_c2g):
    """Check 1: how tightly do all X_T^B_i cluster?"""
    Tb_list = [recover_target_in_base(Tg, T_c2g, Tt) for _, Tg, Tt in pairs]
    if not Tb_list:
        return None
    Tb_mean = mean_se3(Tb_list)
    t_err = [np.linalg.norm(T[:3, 3] - Tb_mean[:3, 3]) * 1000 for T in Tb_list]  # mm
    r_err = [rotmat_distance_deg(T[:3, :3], Tb_mean[:3, :3]) for T in Tb_list]  # deg
    return {
        "mean_Tb": Tb_mean,
        "t_err_mm": t_err,
        "r_err_deg": r_err,
        "t_rms_mm": float(np.sqrt(np.mean(np.square(t_err)))),
        "t_max_mm": float(np.max(t_err)),
        "r_rms_deg": float(np.sqrt(np.mean(np.square(r_err)))),
        "r_max_deg": float(np.max(r_err)),
    }


def leave_one_out(pairs, method_flag, T_full):
    """Check 2: re-solve on N-1, compare each X_C^G to the full-data solution."""
    deltas = []
    stems = [s for s, _, _ in pairs]
    Rg = [Tg[:3, :3] for _, Tg, _ in pairs]
    tg = [Tg[:3, 3] for _, Tg, _ in pairs]
    Rt = [Tt[:3, :3] for _, _, Tt in pairs]
    tt = [Tt[:3, 3] for _, _, Tt in pairs]
    for k in range(len(pairs)):
        idx = [i for i in range(len(pairs)) if i != k]
        R_c2g, t_c2g = cv2.calibrateHandEye(
            R_gripper2base=[Rg[i] for i in idx],
            t_gripper2base=[tg[i] for i in idx],
            R_target2cam=[Rt[i] for i in idx],
            t_target2cam=[tt[i] for i in idx],
            method=method_flag,
        )
        T_loo = np.eye(4, dtype=np.float64)
        T_loo[:3, :3] = R_c2g
        T_loo[:3, 3] = t_c2g.ravel()
        deltas.append(
            (
                stems[k],
                float(np.linalg.norm(T_loo[:3, 3] - T_full[:3, 3]) * 1000),  # mm
                rotmat_distance_deg(T_loo[:3, :3], T_full[:3, :3]),  # deg
            )
        )
    return deltas


def main() -> None:
    args = parse_args()
    record_dir: Path = args.record_dir
    result_path = args.result or (record_dir / "handeye_result.json")

    method, T_c2g, _ = load_handeye_result(result_path)
    cfg, pairs = load_pairs(record_dir)
    n = len(pairs)
    if n < 3:
        raise SystemExit(f"Need >= 3 valid samples to validate, got {n}.")
    print(f"Validating method={method} on {n} samples from {record_dir}.\n")

    # ---- Check 1: target-in-base consistency ----
    rep = consistency_report(pairs, T_c2g)
    print("=== Check 1: X_T^B consistency (target should look static) ===")
    print(f"  translation : RMS={rep['t_rms_mm']:6.2f} mm   max={rep['t_max_mm']:6.2f} mm")
    print(f"  rotation    : RMS={rep['r_rms_deg']:6.3f} deg  max={rep['r_max_deg']:6.3f} deg")
    verdict = "GOOD" if (rep["t_rms_mm"] < 5.0 and rep["r_rms_deg"] < 1.0) else "MARGINAL/POOR"
    print(f"  -> {verdict}  (rule of thumb: RMS t < 5 mm and RMS R < 1 deg)\n")

    print("  per-sample |t_T^B - mean| (mm) / |R| (deg):")
    for stem, te, re in zip([s for s, _, _ in pairs], rep["t_err_mm"], rep["r_err_deg"]):
        print(f"    {stem}:  {te:7.2f} mm   {re:5.3f} deg")

    # ---- Check 2: leave-one-out stability ----
    print("\n=== Check 2: leave-one-out stability of X_C^G ===")
    deltas = leave_one_out(pairs, METHOD_FLAGS[method], T_c2g)
    flagged = []
    print(f"  {'sample':<14}{'|dX_C^G| t (mm)':<18}{'R (deg)'}")
    for stem, dt, dr in deltas:
        flag = ""
        if dt > args.outlier_t_mm or dr > args.outlier_rot_deg:
            flag = "  <-- outlier candidate"
            flagged.append(stem)
        print(f"  {stem:<14}{dt:<18.2f}{dr:<.3f}{flag}")
    if flagged:
        print(f"\n  [hint] {len(flagged)} sample(s) destabilise the solution when removed:")
        print("         inspect their images; if blurred / poorly aimed, delete and re-run handeye_solve.py.")
    else:
        print("\n  -> all samples are consistent; the solution is stable.")
    print(f"\nSummary: {verdict}. LOO outliers flagged: {len(flagged)}.")


if __name__ == "__main__":
    main()
