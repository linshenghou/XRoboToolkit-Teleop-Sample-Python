"""Pico controller sensor process.

This process records native Pico/XR values plus host-side timing fields. The
read latency fields are not native Pico metadata; they are measured around the
SDK calls so later sync diagnostics can see jitter and stalls.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from .ring_buffer import SharedMemoryRingBuffer
from .timebase import Timebase, midpoint


class PicoClient:
    """Small local wrapper around xrobotoolkit_sdk for controller reads."""

    def __init__(self) -> None:
        import xrobotoolkit_sdk as xrt

        self._xrt = xrt
        self._xrt.init()
        print("XRoboToolkit SDK initialized.")

    def get_pose_by_name(self, name: str):
        if name == "left_controller":
            return self._xrt.get_left_controller_pose()
        if name == "right_controller":
            return self._xrt.get_right_controller_pose()
        if name == "headset":
            return self._xrt.get_headset_pose()
        raise ValueError(f"Invalid pose name: {name}")

    def get_key_value_by_name(self, name: str) -> float:
        if name == "left_trigger":
            return float(self._xrt.get_left_trigger())
        if name == "right_trigger":
            return float(self._xrt.get_right_trigger())
        if name == "left_grip":
            return float(self._xrt.get_left_grip())
        if name == "right_grip":
            return float(self._xrt.get_right_grip())
        raise ValueError(f"Invalid key name: {name}")

    def get_joystick_state(self, controller: str):
        if controller == "left":
            return self._xrt.get_left_axis()
        if controller == "right":
            return self._xrt.get_right_axis()
        raise ValueError(f"Invalid controller name: {controller}")

    def get_timestamp_ns(self) -> int:
        return int(self._xrt.get_time_stamp_ns())

    def close(self) -> None:
        self._xrt.close()


def pico_sample_example() -> dict[str, object]:
    return {
        "timestamp": np.float64(0.0),
        "read_start_timestamp": np.float64(0.0),
        "read_end_timestamp": np.float64(0.0),
        "xr_timestamp_ns": np.int64(0),
        "left_pose": np.zeros(7, dtype=np.float64),
        "right_pose": np.zeros(7, dtype=np.float64),
        "left_valid": np.int8(0),
        "right_valid": np.int8(0),
        "left_trigger": np.float32(0.0),
        "right_trigger": np.float32(0.0),
        "left_grip": np.float32(0.0),
        "right_grip": np.float32(0.0),
        "left_joystick": np.zeros(2, dtype=np.float32),
        "right_joystick": np.zeros(2, dtype=np.float32),
    }


def _as_pose(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    out = np.zeros(7, dtype=np.float64)
    out[: min(7, arr.size)] = arr[:7]
    return out


def _valid_pose(pose: np.ndarray) -> int:
    return int(pose.shape == (7,) and np.linalg.norm(pose[3:7]) > 1e-8)


def _as_joystick(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros(2, dtype=np.float32)
    out[: min(2, arr.size)] = arr[:2]
    return out


class PicoControllerProcess(mp.Process):
    """Poll left/right Pico controllers and publish samples to a ring buffer."""

    def __init__(
        self,
        ring_buffer: SharedMemoryRingBuffer,
        timebase: Timebase,
        *,
        frequency: float = 120.0,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.ring_buffer = ring_buffer
        self.timebase = timebase
        self.frequency = float(frequency)
        self.verbose = verbose
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        xr: PicoClient | None = None
        dt = 1.0 / self.frequency
        next_tick = time.perf_counter()
        try:
            xr = PicoClient()
            print("[PICO] XR client ready, waiting for controller poses", flush=True)
            consecutive_errors = 0
            while not self.stop_event.is_set():
                try:
                    read_start = self.timebase.now()
                    left_pose = _as_pose(xr.get_pose_by_name("left_controller"))
                    right_pose = _as_pose(xr.get_pose_by_name("right_controller"))
                    left_trigger = float(xr.get_key_value_by_name("left_trigger"))
                    right_trigger = float(xr.get_key_value_by_name("right_trigger"))
                    left_grip = float(xr.get_key_value_by_name("left_grip"))
                    right_grip = float(xr.get_key_value_by_name("right_grip"))
                    left_joystick = _as_joystick(xr.get_joystick_state("left"))
                    right_joystick = _as_joystick(xr.get_joystick_state("right"))
                    xr_timestamp_ns = int(xr.get_timestamp_ns())
                    read_end = self.timebase.now()
                except Exception as exc:  # noqa: BLE001
                    consecutive_errors += 1
                    if consecutive_errors == 1 or consecutive_errors % 100 == 0:
                        print(f"[PICO] read error #{consecutive_errors}: {exc}", flush=True)
                    self._sleep_until_next(dt, next_tick)
                    next_tick = time.perf_counter()
                    continue

                consecutive_errors = 0
                sample = {
                    "timestamp": midpoint(read_start, read_end),
                    "read_start_timestamp": read_start,
                    "read_end_timestamp": read_end,
                    "xr_timestamp_ns": xr_timestamp_ns,
                    "left_pose": left_pose,
                    "right_pose": right_pose,
                    "left_valid": _valid_pose(left_pose),
                    "right_valid": _valid_pose(right_pose),
                    "left_trigger": left_trigger,
                    "right_trigger": right_trigger,
                    "left_grip": left_grip,
                    "right_grip": right_grip,
                    "left_joystick": left_joystick,
                    "right_joystick": right_joystick,
                }
                self.ring_buffer.put(sample)

                if not self.ready_event.is_set() and (sample["left_valid"] or sample["right_valid"]):
                    print(
                        "[PICO] tracking "
                        f"L={sample['left_valid']} R={sample['right_valid']}",
                        flush=True,
                    )
                    self.ready_event.set()

                if self.verbose:
                    latency_ms = (read_end - read_start) * 1000.0
                    print(
                        "[PICO] "
                        f"L={sample['left_valid']} R={sample['right_valid']} "
                        f"latency={latency_ms:.3f}ms",
                        flush=True,
                    )

                next_tick += dt
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()
        except Exception as exc:  # noqa: BLE001
            print(f"[PICO] FATAL: {exc}", flush=True)
        finally:
            if xr is not None:
                try:
                    xr.close()
                except Exception:
                    pass
            print("[PICO] Stopped.", flush=True)

    @staticmethod
    def _sleep_until_next(dt: float, next_tick: float) -> None:
        sleep_s = next_tick + dt - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test Pico controller polling.")
    parser.add_argument("--frequency", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--buffer-size", type=int, default=1024)
    args = parser.parse_args()

    timebase = Timebase.create()
    ring = SharedMemoryRingBuffer(pico_sample_example(), capacity=args.buffer_size)
    proc = PicoControllerProcess(ring, timebase, frequency=args.frequency)
    proc.start()
    if not proc.ready_event.wait(timeout=8.0):
        proc.stop()
        proc.join(timeout=2.0)
        raise SystemExit("ERROR: Pico process did not become ready.")

    start = time.time()
    try:
        while time.time() - start < args.duration:
            try:
                latest = ring.get_latest()
            except IndexError:
                time.sleep(0.05)
                continue
            latency_ms = (
                float(latest["read_end_timestamp"]) - float(latest["read_start_timestamp"])
            ) * 1000.0
            left = latest["left_pose"][:3]
            right = latest["right_pose"][:3]
            print(
                "t={:.6f} latency={:.3f}ms "
                "L={} ({:+.3f},{:+.3f},{:+.3f}) "
                "R={} ({:+.3f},{:+.3f},{:+.3f}) count={}".format(
                    float(latest["timestamp"]),
                    latency_ms,
                    int(latest["left_valid"]),
                    float(left[0]),
                    float(left[1]),
                    float(left[2]),
                    int(latest["right_valid"]),
                    float(right[0]),
                    float(right[1]),
                    float(right[2]),
                    ring.count,
                ),
                flush=True,
            )
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        proc.stop()
        proc.join(timeout=3.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)


if __name__ == "__main__":
    main()
