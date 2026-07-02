"""Hot-start Pico, RealSense cameras, and encoders, then record episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

from .encoder_driver import EncoderCalibration, find_port_by_usb_serial
from .encoder_process import EncoderProcess, encoder_sample_example
from .pico_process import PicoControllerProcess, pico_sample_example
from .ring_buffer import SharedMemoryRingBuffer
from .timebase import Timebase


DEFAULT_ENCODER_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "encoder_mapping.json"
DEFAULT_CAMERA_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "realsense_cameras.json"
DEFAULT_REALSENSE_PYTHON = Path(sys.executable)
ENCODER_ROLES = ("left_encoder", "right_encoder")

SESSION_METADATA_README = """# Data-collection session metadata

This directory contains one hot-start recording session.

## Layout

- `metadata.json`: session-level config and episode list.
- `episode_*/metadata.json`: one episode summary.
- `episode_*/cameras/`: RealSense RGB videos and camera timestamps.
- `episode_*/lowdim/`: Pico and encoder samples sliced to the episode window.

All saved `timestamp` arrays use the same host timebase, so low-dimensional
signals can be aligned directly to camera `color_timestamps.npy`.
"""


def _session_dir(root: Path) -> Path:
    session = root / time.strftime("session_%Y%m%d_%H%M%S")
    session.mkdir(parents=True, exist_ok=False)
    return session


def _save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _slice_by_window(
    ring: SharedMemoryRingBuffer,
    count_start: int,
    t_start: float,
    t_stop: float,
) -> tuple[dict[str, np.ndarray], bool]:
    produced = ring.count - count_start
    if produced <= 0:
        empty = {
            key: np.empty((0,) + spec.shape, dtype=spec.dtype)
            for key, spec in ring.specs.items()
        }
        return empty, False
    truncated = produced > ring.capacity
    batch = ring.get_last_k(min(produced, ring.capacity))
    ts = batch["timestamp"]
    mask = (ts >= t_start) & (ts <= t_stop)
    return {key: value[mask] for key, value in batch.items()}, truncated


def _flush_pending_keys() -> None:
    try:
        import msvcrt
    except ImportError:
        return
    while msvcrt.kbhit():
        msvcrt.getwch()


def _read_key(prompt: str, valid: set[str]) -> str:
    _flush_pending_keys()
    print(prompt, flush=True)
    try:
        import msvcrt

        while True:
            key = msvcrt.getwch().lower()
            if key in valid:
                print(key, flush=True)
                return key
    except ImportError:
        while True:
            value = input("> ").strip().lower()
            if value in valid:
                return value


def _play_prompt_sound(kind: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    patterns = {
        "start": [(1200, 120), (1500, 120)],
        "stop": [(520, 260)],
        "error": [(420, 120), (420, 120), (420, 120)],
    }

    def worker() -> None:
        try:
            import winsound

            for freq, duration_ms in patterns[kind]:
                winsound.Beep(freq, duration_ms)
                time.sleep(0.04)
        except Exception:
            print("\a", end="", flush=True)

    threading.Thread(target=worker, daemon=True).start()


def _role_label(role: str) -> str:
    return role.removesuffix("_encoder")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _resolve_encoder_ports(config_path: Path) -> tuple[dict, list[tuple[str, str]]]:
    config = _load_json(config_path)
    roles = config.get("roles", {}) or {}
    resolved: list[tuple[str, str]] = []
    for role in ENCODER_ROLES:
        entry = roles.get(role)
        if not entry:
            continue
        port = None
        usb_serial = entry.get("usb_serial")
        if usb_serial:
            port = find_port_by_usb_serial(str(usb_serial))
        if port is None:
            port = entry.get("port")
        if port is None:
            raise RuntimeError(f"could not resolve encoder role {role}")
        resolved.append((role, str(port)))
    return config, resolved


def _calibration_from_entry(entry: dict, *, raw_only: bool) -> EncoderCalibration:
    if raw_only:
        return EncoderCalibration()
    cal_data = entry.get("calibration") or {}
    stroke_mm = cal_data.get("stroke_mm")
    if stroke_mm is None and entry.get("gripper_length_m") is not None:
        stroke_mm = float(entry["gripper_length_m"]) * 1000.0
    if stroke_mm is None and entry.get("gripper_length") is not None:
        stroke_mm = float(entry["gripper_length"]) * 1000.0
    return EncoderCalibration(
        raw_closed=cal_data.get("raw_closed"),
        raw_open=cal_data.get("raw_open"),
        stroke_mm=stroke_mm,
    )


class _CameraServer:
    def __init__(self, proc: subprocess.Popen) -> None:
        self.proc = proc

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def start_episode(self, cameras_dir: Path, *, timeout_s: float) -> bool:
        ready = cameras_dir / "realsense_ready.json"
        done = cameras_dir / "realsense_done.json"
        for marker in (ready, done):
            if marker.exists():
                marker.unlink()
        self._send(f"START {cameras_dir}")
        return self._wait(ready, timeout_s=timeout_s)

    def stop_episode(self, cameras_dir: Path, *, timeout_s: float) -> bool:
        self._send("STOP")
        return self._wait(cameras_dir / "realsense_done.json", timeout_s=timeout_s)

    def quit(self) -> None:
        try:
            self._send("QUIT")
        except Exception:
            pass

    def _wait(self, marker: Path, *, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if marker.exists():
                return True
            if self.proc.poll() is not None:
                return False
            time.sleep(0.02)
        return False


def _start_camera_server(
    python_exe: Path,
    *,
    width: int,
    height: int,
    fps: int,
    max_cameras: int | None,
    camera_config: Path,
    timebase: Timebase,
    ready_file: Path,
    log_file: Path,
) -> _CameraServer:
    cmd = [
        str(python_exe),
        "-m",
        "scripts.data_collection.realsense_rgb_record",
        "--serve",
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--camera-config",
        str(camera_config),
        "--wall0",
        str(timebase.wall0),
        "--perf0",
        str(timebase.perf0),
        "--ready-file",
        str(ready_file),
    ]
    if max_cameras is not None:
        cmd.extend(["--max-cameras", str(max_cameras)])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return _CameraServer(proc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("data/sessions"))
    parser.add_argument("--pico-frequency", type=float, default=120.0)
    parser.add_argument("--encoder-frequency", type=float, default=30.0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--max-cameras", type=int, default=3)
    parser.add_argument("--max-episode-s", type=float, default=180.0)
    parser.add_argument("--encoder-mapping", type=Path, default=DEFAULT_ENCODER_CONFIG_PATH)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH)
    parser.add_argument("--encoder-raw-only", action="store_true")
    parser.add_argument("--realsense-python", type=Path, default=DEFAULT_REALSENSE_PYTHON)
    parser.add_argument("--no-camera", action="store_true")
    parser.add_argument("--no-encoder", action="store_true")
    parser.add_argument("--no-pico", action="store_true")
    parser.add_argument("--no-cue-sound", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cue_sound = not args.no_cue_sound
    session_dir = _session_dir(args.output_dir)
    _write_text(session_dir / "README_metadata.md", SESSION_METADATA_README)
    timebase = Timebase.create()
    print(f"[SESSION] {session_dir}", flush=True)

    cap_pico = int(args.max_episode_s * args.pico_frequency) + 512
    cap_enc = int(args.max_episode_s * args.encoder_frequency) + 256

    pico_ring = None
    pico = None
    if not args.no_pico:
        pico_ring = SharedMemoryRingBuffer(pico_sample_example(), cap_pico)
        pico = PicoControllerProcess(pico_ring, timebase, frequency=args.pico_frequency)

    encoder_config: dict = {}
    resolved_ports: list[tuple[str, str]] = []
    encoder_rings: dict[str, SharedMemoryRingBuffer] = {}
    encoders: dict[str, EncoderProcess] = {}
    if not args.no_encoder:
        encoder_config, resolved_ports = _resolve_encoder_ports(args.encoder_mapping)
        role_entries = encoder_config.get("roles", {}) or {}
        for role, port in resolved_ports:
            entry = role_entries.get(role) or {}
            calibration = _calibration_from_entry(entry, raw_only=args.encoder_raw_only)
            ring = SharedMemoryRingBuffer(encoder_sample_example(), cap_enc)
            encoder_rings[role] = ring
            encoders[role] = EncoderProcess(
                ring,
                timebase,
                port=port,
                usb_serial=entry.get("usb_serial"),
                baudrate=int(entry.get("baudrate", encoder_config.get("baudrate", 9600))),
                slave_addr=int(entry.get("slave", encoder_config.get("slave", 1))),
                frequency=args.encoder_frequency,
                calibration=calibration,
                raw_only=args.encoder_raw_only,
                side=_role_label(role),
            )
            print(f"[SESSION] encoder {role}: port={port}", flush=True)

    camera: _CameraServer | None = None
    episode_index = 0
    metadata = {
        "schema": "pico_realsense_encoder_session_v1",
        "session_dir": str(session_dir),
        "metadata_readme": "README_metadata.md",
        "timebase": {"wall0": timebase.wall0, "perf0": timebase.perf0},
        "pico": None
        if args.no_pico
        else {"frequency": args.pico_frequency, "fields": list(pico_sample_example())},
        "encoder_mapping": encoder_config,
        "encoder_resolved_ports": {role: port for role, port in resolved_ports},
        "encoder_raw_only": args.encoder_raw_only,
        "encoder_frequency": None if args.no_encoder else args.encoder_frequency,
        "camera": None
        if args.no_camera
        else {
            "width": args.camera_width,
            "height": args.camera_height,
            "fps": args.camera_fps,
            "max_cameras": args.max_cameras,
            "config": str(args.camera_config),
        },
        "episodes": [],
    }

    try:
        if not args.no_camera:
            ready_file = session_dir / "cameras_rig_ready.json"
            camera = _start_camera_server(
                args.realsense_python,
                width=args.camera_width,
                height=args.camera_height,
                fps=args.camera_fps,
                max_cameras=args.max_cameras,
                camera_config=args.camera_config,
                timebase=timebase,
                ready_file=ready_file,
                log_file=session_dir / "realsense_serve.log",
            )
            print("[SESSION] warming up cameras...", flush=True)
            if not camera._wait(ready_file, timeout_s=30.0):
                raise RuntimeError(f"RealSense server did not become ready; see {session_dir / 'realsense_serve.log'}")
            print("[SESSION] cameras ready", flush=True)

        for role, proc in encoders.items():
            proc.start()
            if not proc.ready_event.wait(timeout=5.0):
                raise RuntimeError(f"encoder {role} did not become ready")
        if pico is not None:
            pico.start()
            if not pico.ready_event.wait(timeout=8.0):
                raise RuntimeError("Pico process did not become ready")

        print("[SESSION] all enabled sensors hot. c=start episode, q=quit", flush=True)

        while True:
            cmd = _read_key(f"\n[SESSION] episode {episode_index}: press c to record, q to quit", {"c", "q"})
            if cmd == "q":
                break

            ep_dir = session_dir / f"episode_{episode_index:03d}"
            cameras_dir = ep_dir / "cameras"
            t_start = timebase.now()
            _play_prompt_sound("start", enabled=cue_sound)
            print("[SESSION] recording started", flush=True)
            pico_count_start = pico_ring.count if pico_ring is not None else 0
            enc_count_start = {role: ring.count for role, ring in encoder_rings.items()}

            if camera is not None and not camera.start_episode(cameras_dir, timeout_s=15.0):
                print("[SESSION] WARNING: camera did not confirm START.", flush=True)

            _read_key("[SESSION] recording... press q to stop/save", {"q"})
            t_stop = timebase.now()
            _play_prompt_sound("stop", enabled=cue_sound)
            print("[SESSION] saving episode...", flush=True)

            cam_frames = None
            if camera is not None:
                if camera.stop_episode(cameras_dir, timeout_s=20.0):
                    try:
                        done = json.loads((cameras_dir / "realsense_done.json").read_text(encoding="utf-8"))
                        cam_frames = done.get("frames")
                    except Exception:
                        pass
                else:
                    print("[SESSION] WARNING: camera did not confirm STOP.", flush=True)

            lowdim_dir = ep_dir / "lowdim"
            ep_meta = {
                "episode": episode_index,
                "t_start": t_start,
                "t_stop": t_stop,
                "duration": t_stop - t_start,
                "camera_frames": cam_frames,
                "pico_samples": 0,
                "encoder_samples": {},
            }
            if pico_ring is not None:
                data, truncated = _slice_by_window(pico_ring, pico_count_start, t_start, t_stop)
                _save_npz(lowdim_dir / "pico_controllers.npz", data)
                ep_meta["pico_samples"] = int(len(data["timestamp"]))
                if truncated:
                    print("[SESSION] WARNING: Pico ring truncated.", flush=True)

            for role, ring in encoder_rings.items():
                data, truncated = _slice_by_window(ring, enc_count_start[role], t_start, t_stop)
                label = _role_label(role)
                _save_npz(lowdim_dir / f"encoder_{label}.npz", data)
                ep_meta["encoder_samples"][role] = int(len(data["timestamp"]))
                if truncated:
                    print(f"[SESSION] WARNING: encoder {role} ring truncated.", flush=True)

            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "metadata.json").write_text(json.dumps(ep_meta, indent=2), encoding="utf-8")
            metadata["episodes"].append(ep_meta)
            print(
                f"[SESSION] saved episode {episode_index}: "
                f"dur={ep_meta['duration']:.1f}s pico={ep_meta['pico_samples']} "
                f"enc={ep_meta['encoder_samples']} cam_frames={cam_frames} -> {ep_dir}",
                flush=True,
            )
            episode_index += 1
    except KeyboardInterrupt:
        print("\n[SESSION] interrupted.", flush=True)
    finally:
        if camera is not None:
            camera.quit()
        for proc in encoders.values():
            if proc.pid is not None:
                proc.stop()
        if pico is not None and pico.pid is not None:
            pico.stop()
        for proc in encoders.values():
            if proc.pid is None:
                continue
            proc.join(timeout=3.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        if pico is not None and pico.pid is not None:
            pico.join(timeout=3.0)
            if pico.is_alive():
                pico.terminate()
                pico.join(timeout=1.0)
        if camera is not None:
            try:
                camera.proc.wait(timeout=10.0)
            except Exception:
                camera.proc.terminate()
        metadata["episode_count"] = episode_index
        (session_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[SESSION] done. {episode_index} episode(s). Metadata: {session_dir / 'metadata.json'}", flush=True)


if __name__ == "__main__":
    main()
