"""Phase 2: Pico controller coordinate-frame sanity check + axis_remap discovery.

Goal
----
Before hand-eye calibration you must know how the Pico controller's native axes map
onto the physical world (translation test) and whether its rotations are proper
(right-hand rule). This script does both and, on top, produces the exact ``--axis-remap``
matrix that handeye_capture.py accepts.

Live monitor (always on)
    Prints position (m + mm), quaternion [qx, qy, qz, qw], and det(R). The determinant
    is the handedness indicator: a quaternion->matrix conversion ALWAYS yields det = +1
    (a proper SO(3)), so this number should read 1.0000 -- if it ever does not, something
    upstream is broken. Watch which native axis grows/shrinks as you translate along a
    physical direction to learn the labelling by feel.

Guided 3-axis translation protocol (produces axis_remap)
    You pick a right-handed physical triple (X, Y, Z) -- e.g. X=right, Y=up, Z=backward
    (right x up = backward), or the OpenCV optical X=right, Y=down, Z=forward -- then:
        REST  : hold at a neutral reference, press SPACE
        +X    : move ~15 cm along your +X, keep still, press SPACE
        +Y    : move ~15 cm along your +Y, keep still, press SPACE
        +Z    : move ~15 cm along your +Z, keep still, press SPACE
    For each axis k we measure the Pico-frame displacement d_k = p_k - p_rest; the unit
    displacement is row k of the remap M (Pico -> physical). The script then checks
    orthonormality and det(M): a RIGHT-handed physical triple gives det = +1 (a valid
    rotation, safe to feed calibrateHandEye); a left-handed one gives det = -1 and is
    rejected with a hint to flip one axis.

Important note on correctness
    calibrateHandEye only needs *internally consistent* proper rotations -- it does not
    care about the physical labelling of the base axes. So axis_remap is never needed for
    solvability; it only re-expresses X_C^G in a physically meaningful base frame. The one
    hard rule is det(M) = +1 (anything else would corrupt every R into a reflection).

Run
    python scripts/calibration/handeye_sanity_check.py --controller right_controller
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient

from handeye_common import quat_xyzw_to_rotmat

# Protocol steps: (short label, on-screen instruction).
STEPS = [
    ("REST", "hold at a NEUTRAL rest pose, keep still, SPACE"),
    ("+X", "move ~15 cm along your +X (e.g. to the right), keep still, SPACE"),
    ("+Y", "move ~15 cm along your +Y (e.g. upward), keep still, SPACE"),
    ("+Z", "move ~15 cm along your +Z (keep X,Y,Z RIGHT-handed), keep still, SPACE"),
]

WIN_W, WIN_H = 780, 720


def is_valid_pose(pose) -> bool:
    """XR can return a zero quaternion while tracking is not ready."""
    if pose is None:
        return False
    arr = np.asarray(pose, dtype=np.float64)
    return arr.shape[0] >= 7 and np.linalg.norm(arr[3:7]) > 1e-8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pico axis sanity check + axis_remap discovery.")
    p.add_argument("--controller", default="right_controller", choices=["left_controller", "right_controller", "headset"])
    p.add_argument("--out", type=Path, default=Path("calibration_data/axis_remap.json"), help="where to save the discovered remap.")
    p.add_argument("--avg-samples", type=int, default=24, help="poses averaged per SPACE press (noise reduction).")
    p.add_argument("--move-m", type=float, default=0.15, help="suggested displacement per axis (display only).")
    return p.parse_args()


# =====================================================================================
# Pose averaging + remap construction
# =====================================================================================


def average_position(xr: XrClient, controller: str, n: int = 24, dt: float = 0.012) -> np.ndarray:
    """Average ``n`` consecutive Pico translations to suppress jitter at capture time."""
    acc = np.zeros(3, dtype=np.float64)
    got = 0
    deadline = time.monotonic() + max(3.0, n * dt * 8.0)
    while got < n:
        pose = xr.get_pose_by_name(controller)
        if not is_valid_pose(pose):
            if time.monotonic() > deadline:
                raise RuntimeError(f"Timed out waiting for a valid pose from {controller}.")
            time.sleep(dt)
            continue
        acc += np.asarray(pose[:3], dtype=np.float64)
        got += 1
        time.sleep(dt)
    return acc / n


def build_remap(p_rest: np.ndarray, p_x: np.ndarray, p_y: np.ndarray, p_z: np.ndarray):
    """Build M (Pico -> physical) from three axis displacements about the rest pose.

    Row k of M is the unit Pico-frame displacement measured for physical axis k.
    Returns (M 3x3, info dict with magnitudes, orthonormality error, determinant).
    """
    disp = [np.asarray(p, dtype=np.float64) - np.asarray(p_rest, dtype=np.float64) for p in (p_x, p_y, p_z)]
    mags = [float(np.linalg.norm(d)) for d in disp]
    rows = []
    for d, m in zip(disp, mags):
        if m < 1e-6:
            print("  [warn] a displacement was ~0; re-do that axis with a larger move.")
            rows.append(np.zeros(3, dtype=np.float64))
        else:
            rows.append(d / m)
    M = np.stack(rows, axis=0)

    gram = M @ M.T
    offdiag = max(abs(gram[i, j]) for i in range(3) for j in range(3) if i != j)
    diag_err = max(abs(gram[i, i] - 1.0) for i in range(3))
    det = float(np.linalg.det(M))
    info = {"mags_m": mags, "orthogonality_err": float(offdiag), "norm_err": float(diag_err), "det": det}
    return M, info


def validate_remap(info) -> tuple[bool, str]:
    """Return (is_valid, human reason). Valid <=> right-handed + near-orthonormal."""
    if info["det"] < 0.5:
        return False, (
            "det(M) = %.3f < 0  =>  the physical (X,Y,Z) triple you moved along is LEFT-handed.\n"
            "                    That M is a reflection, not a rotation, and would corrupt every pose.\n"
            "                    Flip the sign of ONE axis (e.g. flip +Z to -Z) and re-run." % info["det"]
        )
    if info["orthogonality_err"] > 0.15:
        return False, (
            "axes are not orthogonal (off-diagonal = %.3f).\n"
            "Move along cleaner perpendicular directions; keep the controller orientation fixed during translation."
            % info["orthogonality_err"]
        )
    return True, "det = +1 and axes are orthogonal -> M is a valid rotation (safe for calibrateHandEye)."


# =====================================================================================
# HUD rendering
# =====================================================================================


def _line(img, text, y, scale=0.55, color=(225, 225, 225), thickness=1, x=12):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(state) -> np.ndarray:
    img = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    y = 34

    _line(img, f"Pico sanity check   |   controller = {state['controller']}", y, scale=0.7, color=(255, 255, 255), thickness=2)
    y += 40

    _line(img, "--- live (native Pico frame) ---", y, scale=0.5, color=(160, 160, 160)); y += 26
    t = state["t"]
    _line(img, f"pos (m) :  {t[0]:+.3f}  {t[1]:+.3f}  {t[2]:+.3f}", y, scale=0.6); y += 26
    _line(img, f"pos (mm):  {t[0]*1000:+7.1f}  {t[1]*1000:+7.1f}  {t[2]*1000:+7.1f}", y, scale=0.6); y += 26
    q = state["quat_xyzw"]
    _line(img, f"quat xyzw: {q[0]:+.4f} {q[1]:+.4f} {q[2]:+.4f} {q[3]:+.4f}", y, scale=0.55); y += 26
    det = state["det"]
    dcolor = (0, 220, 120) if abs(det - 1.0) < 1e-3 else (0, 0, 255)
    _line(img, f"det(R) = {det:.4f}   (must be +1.0 for a valid SO(3) from a quaternion)", y, scale=0.55, color=dcolor); y += 34

    _line(img, "--- axis_remap protocol ---", y, scale=0.5, color=(160, 160, 160)); y += 26
    if state["result"] is None:
        label, instr = STEPS[state["step"]]
        _line(img, f"step {state['step'] + 1}/4 : {label}", y, scale=0.62, color=(0, 200, 255), thickness=2); y += 28
        _line(img, instr, y, scale=0.5, color=(200, 200, 200)); y += 26
        if state["p_rest"] is not None:
            d = (t - state["p_rest"]) * 1000.0
            _line(img, f"delta from rest (mm): {d[0]:+7.1f} {d[1]:+7.1f} {d[2]:+7.1f}", y, scale=0.55, color=(0, 200, 200)); y += 26
    else:
        M, info, valid, reason = state["result"]
        rc = (0, 220, 120) if valid else (0, 0, 255)
        _line(img, f"axis_remap  det={info['det']:+.3f}  ortho_err={info['orthogonality_err']:.3f}", y, scale=0.58, color=rc, thickness=2); y += 28
        for i in range(3):
            _line(img, f"  [{M[i, 0]:+.3f} {M[i, 1]:+.3f} {M[i, 2]:+.3f}]", y, scale=0.6, color=rc); y += 24
        y += 6
        flat = ", ".join(f"{v:.4f}" for v in M.flatten())
        _line(img, "--axis-remap " + flat, y, scale=0.42, color=(255, 255, 255)); y += 22
        _line(img, ("VALID: " if valid else "INVALID: ") + reason.splitlines()[0], y, scale=0.45, color=rc); y += 26

    y = WIN_H - 30
    _line(img, "SPACE = capture step    r = reset    q / ESC = quit", y, scale=0.5, color=(170, 170, 170))
    return img


def render_waiting(controller: str) -> np.ndarray:
    img = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    _line(img, f"Pico sanity check   |   controller = {controller}", 34, scale=0.7, color=(255, 255, 255), thickness=2)
    _line(img, "Waiting for a valid XR pose...", 92, scale=0.65, color=(0, 200, 255), thickness=2)
    _line(img, "Make sure XRoboToolkit PC Service is running and the controller is awake/tracked.", 128, scale=0.5)
    _line(img, "q / ESC = quit", WIN_H - 30, scale=0.5, color=(170, 170, 170))
    return img


# =====================================================================================
# Main
# =====================================================================================


def main() -> None:
    args = parse_args()
    xr = XrClient()
    print(
        "Pico sanity check.\n"
        "Pick a RIGHT-handed physical (X, Y, Z) and move the controller along each in turn.\n"
        "Watch the 'delta from rest (mm)' line: the dominant native axis is your mapping.\n"
    )

    captures: dict[str, np.ndarray] = {}
    p_rest: np.ndarray | None = None
    step_i = 0
    result = None  # (M, info, valid, reason)

    win = "handeye sanity check"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        next_invalid_warn = 0.0
        while True:
            pose = xr.get_pose_by_name(args.controller)
            if not is_valid_pose(pose):
                now = time.monotonic()
                if now >= next_invalid_warn:
                    print(f"[wait] {args.controller} pose is not valid yet (zero quaternion).")
                    next_invalid_warn = now + 2.0
                cv2.imshow(win, render_waiting(args.controller))
                key = cv2.waitKey(50) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue
            t = np.asarray(pose[:3], dtype=np.float64)
            quat = np.asarray(pose[3:], dtype=np.float64)  # [qx, qy, qz, qw]
            R = quat_xyzw_to_rotmat(quat)
            det = float(np.linalg.det(R))

            state = {
                "controller": args.controller,
                "t": t,
                "quat_xyzw": quat,
                "det": det,
                "step": step_i,
                "p_rest": p_rest,
                "result": result,
            }
            cv2.imshow(win, render(state))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                captures, p_rest, step_i, result = {}, None, 0, None
                print("[reset] protocol cleared.")
                continue
            if key in (ord(" "), ord("p")):
                if result is not None:
                    print("[info] protocol already done -- press 'r' to re-run.")
                    continue
                if step_i >= len(STEPS):
                    continue
                label = STEPS[step_i][0]
                pos = average_position(xr, args.controller, n=args.avg_samples)
                captures[label] = pos
                print(f"[capture] {label:>4}  pos(m) = {pos.round(4).tolist()}")
                if step_i == 0:
                    p_rest = pos
                step_i += 1

                if step_i == len(STEPS):
                    M, info = build_remap(captures["REST"], captures["+X"], captures["+Y"], captures["+Z"])
                    valid, reason = validate_remap(info)
                    result = (M, info, valid, reason)
                    print("\n=== axis_remap result ===")
                    print(np.round(M, 4))
                    print(f"det = {info['det']:+.4f}   orthogonality_err = {info['orthogonality_err']:.4f}")
                    print(f"axis move magnitudes (m) = {[round(m, 3) for m in info['mags_m']]}")
                    print(("[OK]   " if valid else "[FAIL] ") + reason)
                    if valid:
                        flat = [float(v) for v in M.flatten()]
                        print("\nPaste into handeye_capture.py:\n  --axis-remap " + " ".join(f"{v:.4f}" for v in flat))
                        args.out.parent.mkdir(parents=True, exist_ok=True)
                        args.out.write_text(
                            json.dumps(
                                {
                                    "controller": args.controller,
                                    "samples_m": {k: v.tolist() for k, v in captures.items()},
                                    "axis_remap": M.tolist(),
                                    "axis_remap_flat": flat,
                                    "det": info["det"],
                                    "orthogonality_err": info["orthogonality_err"],
                                    "valid": valid,
                                },
                                indent=2,
                            )
                        )
                        print(f"\nSaved -> {args.out}")
                    else:
                        print("\nNo matrix saved (invalid). Adjust your axis choice and press 'r'.")
    finally:
        cv2.destroyAllWindows()
        xr.close()


if __name__ == "__main__":
    main()
