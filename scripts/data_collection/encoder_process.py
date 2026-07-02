"""Magnetic encoder sensor process for data collection."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

from .encoder_driver import (
    EncoderCalibration,
    create_instrument,
    read_raw,
    resolve_serial_port,
)
from .ring_buffer import SharedMemoryRingBuffer
from .timebase import Timebase, midpoint


def encoder_sample_example() -> dict[str, object]:
    return {
        "timestamp": np.float64(0.0),
        "read_start_timestamp": np.float64(0.0),
        "read_end_timestamp": np.float64(0.0),
        "raw": np.int32(-1),
        "normalized": np.float32(np.nan),
        "metric": np.float32(np.nan),
        "valid": np.int8(0),
    }


class EncoderProcess(mp.Process):
    """Poll a BRT magnetic encoder and publish samples to a ring buffer."""

    def __init__(
        self,
        ring_buffer: SharedMemoryRingBuffer,
        timebase: Timebase,
        *,
        port: str | None = None,
        usb_serial: str | None = None,
        baudrate: int = 9600,
        slave_addr: int = 1,
        frequency: float = 30.0,
        calibration: EncoderCalibration | None = None,
        raw_only: bool = False,
        side: str | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.ring_buffer = ring_buffer
        self.timebase = timebase
        self.port = port
        self.usb_serial = usb_serial
        self.baudrate = baudrate
        self.slave_addr = slave_addr
        self.frequency = float(frequency)
        self.calibration = calibration or EncoderCalibration.load()
        self.raw_only = raw_only
        self.side = side
        self.verbose = verbose
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    @property
    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        port = resolve_serial_port(
            port=self.port,
            usb_serial=self.usb_serial,
            baudrate=self.baudrate,
            slave_addr=self.slave_addr,
        )
        label = f" [{self.side}]" if self.side else ""
        if port is None:
            print(f"[ENCODER]{label} ERROR: no serial port found.", flush=True)
            return

        inst = None
        dt = 1.0 / self.frequency
        next_tick = time.perf_counter()
        try:
            try:
                inst = create_instrument(port, slave_addr=self.slave_addr, baudrate=self.baudrate)
            except Exception as exc:
                print(f"[ENCODER]{label} ERROR: could not open {port}: {exc}", flush=True)
                return
            probe = read_raw(inst)
            if probe is None:
                print(f"[ENCODER]{label} ERROR: encoder did not respond on {port}.", flush=True)
                return
            print(f"[ENCODER]{label} Started on {port}, raw={probe}", flush=True)
            self.ready_event.set()

            consecutive_failures = 0
            while not self.stop_event.is_set():
                read_start = self.timebase.now()
                raw = read_raw(inst)
                read_end = self.timebase.now()
                if raw is None:
                    consecutive_failures += 1
                    if consecutive_failures == 1 or consecutive_failures % 50 == 0:
                        print(
                            f"[ENCODER]{label} read failed #{consecutive_failures} on {port}",
                            flush=True,
                        )
                    sample = {
                        "timestamp": midpoint(read_start, read_end),
                        "read_start_timestamp": read_start,
                        "read_end_timestamp": read_end,
                        "raw": -1,
                        "normalized": np.nan,
                        "metric": np.nan,
                        "valid": 0,
                    }
                else:
                    consecutive_failures = 0
                    normalized = np.nan if self.raw_only else self.calibration.normalise(int(raw))
                    metric = np.nan if self.raw_only else self.calibration.metric_m(int(raw))
                    sample = {
                        "timestamp": midpoint(read_start, read_end),
                        "read_start_timestamp": read_start,
                        "read_end_timestamp": read_end,
                        "raw": int(raw),
                        "normalized": normalized,
                        "metric": metric,
                        "valid": 1,
                    }
                self.ring_buffer.put(sample)
                if self.verbose and raw is not None:
                    latency_ms = (read_end - read_start) * 1000.0
                    print(
                        f"[ENCODER]{label} raw={raw} "
                        f"norm={sample['normalized']:.3f} latency={latency_ms:.3f}ms",
                        flush=True,
                    )

                next_tick += dt
                sleep_s = next_tick - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.perf_counter()
        finally:
            if inst is not None:
                try:
                    inst.serial.close()
                except Exception:
                    pass
            print(f"[ENCODER]{label} Stopped.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test one encoder process.")
    parser.add_argument("--port", default=None)
    parser.add_argument("--usb-serial", default=None)
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--slave", type=int, default=1)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--buffer-size", type=int, default=256)
    parser.add_argument("--raw-only", action="store_true")
    args = parser.parse_args()

    timebase = Timebase.create()
    ring = SharedMemoryRingBuffer(encoder_sample_example(), capacity=args.buffer_size)
    proc = EncoderProcess(
        ring,
        timebase,
        port=args.port,
        usb_serial=args.usb_serial,
        baudrate=args.baudrate,
        slave_addr=args.slave,
        frequency=args.frequency,
        raw_only=args.raw_only,
    )
    proc.start()
    if not proc.ready_event.wait(timeout=5.0):
        proc.stop()
        proc.join(timeout=2.0)
        raise SystemExit("ERROR: encoder process did not become ready.")

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
            print(
                "t={:.6f} raw={} norm={:.3f} valid={} latency={:.3f}ms count={}".format(
                    float(latest["timestamp"]),
                    int(latest["raw"]),
                    float(latest["normalized"]),
                    int(latest["valid"]),
                    latency_ms,
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
