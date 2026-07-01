"""Print raw XR poses from XRoboToolkit PC Service.

Use this before calibration to verify that the headset/controllers are connected and
tracking. A valid pose has a nonzero quaternion norm.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from xrobotoolkit_teleop.common.xr_client import XrClient


POSE_NAMES = ("headset", "left_controller", "right_controller")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check XRoboToolkit headset/controller poses.")
    parser.add_argument("--rate", type=float, default=5.0, help="print frequency in Hz.")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds to run; 0 means forever.")
    parser.add_argument("--once", action="store_true", help="print one sample and exit.")
    return parser.parse_args()


def valid_pose(pose) -> bool:
    arr = np.asarray(pose, dtype=np.float64)
    return arr.shape[0] >= 7 and np.linalg.norm(arr[3:7]) > 1e-8


def format_pose(name: str, pose) -> str:
    arr = np.asarray(pose, dtype=np.float64)
    if arr.shape[0] < 7:
        return f"{name:16s} INVALID shape={arr.shape}"
    q_norm = float(np.linalg.norm(arr[3:7]))
    status = "OK" if q_norm > 1e-8 else "WAIT"
    return (
        f"{name:16s} {status:4s} "
        f"xyz=({arr[0]:+.3f}, {arr[1]:+.3f}, {arr[2]:+.3f}) "
        f"quat_xyzw=({arr[3]:+.4f}, {arr[4]:+.4f}, {arr[5]:+.4f}, {arr[6]:+.4f}) "
        f"|q|={q_norm:.4f}"
    )


def main() -> None:
    args = parse_args()
    xr = XrClient()
    interval = 1.0 / max(args.rate, 1e-6)
    end_time = None if args.duration <= 0 else time.monotonic() + args.duration

    try:
        while True:
            print(f"\nXR timestamp ns: {xr.get_timestamp_ns()}")
            for name in POSE_NAMES:
                print(format_pose(name, xr.get_pose_by_name(name)))
            print(
                "buttons "
                f"L_trigger={xr.get_key_value_by_name('left_trigger'):.3f} "
                f"R_trigger={xr.get_key_value_by_name('right_trigger'):.3f} "
                f"L_grip={xr.get_key_value_by_name('left_grip'):.3f} "
                f"R_grip={xr.get_key_value_by_name('right_grip'):.3f}",
                flush=True,
            )
            if args.once:
                break
            if end_time is not None and time.monotonic() >= end_time:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        xr.close()


if __name__ == "__main__":
    main()
