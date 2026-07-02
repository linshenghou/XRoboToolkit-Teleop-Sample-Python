"""Record RGB-only RealSense streams for a raw data-collection smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np


METADATA_FIELDS = {
    "metadata_frame_timestamp": "frame_timestamp",
    "metadata_sensor_timestamp": "sensor_timestamp",
    "metadata_time_of_arrival": "time_of_arrival",
    "metadata_backend_timestamp": "backend_timestamp",
}
DEFAULT_CAMERA_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "realsense_cameras.json"
)
CAMERA_METADATA_README = """# RealSense camera metadata guide

This directory contains one camera episode recorded by `realsense_rgb_record`.

## Most important files

- `metadata.json`: episode-level camera metadata and camera list.
- `cam*/color.mp4`: RGB video for each RealSense camera.
- `cam*/color_timestamps.npy`: primary timestamp for each video frame. Use this
  for sync with `lowdim/*.npz` because it is on the shared host timebase.
- `cam*/sample.png`: preview image from the last frame, for quick inspection only.

## Timestamp files inside each `cam*` folder

- `color_timestamps.npy`: host midpoint timestamp in seconds. This is the
  recommended timeline for downstream alignment with tracker/encoder data.
- `device_timestamps.npy`: RealSense device timestamp from `frame.get_timestamp()`.
  Useful for diagnostics, but it is not the shared host clock.
- `timestamp_domain.npy`: RealSense timestamp-domain label for each frame.
- `frame_counter.npy`: raw RealSense frame counter metadata when available.
- `receive_durations_ms.npy`: host-side `wait_for_frames` duration in ms. Use it
  to diagnose camera stalls; do not use it as frame time.
- `metadata_*.npy`: raw RealSense SDK metadata fields. These are optional
  diagnostics and may contain NaN when the camera/driver does not expose them.

## Practical downstream rule

For most conversion or synchronization code:

1. Read `camX/color.mp4`.
2. Load `camX/color_timestamps.npy`.
3. Treat `color_timestamps[i]` as the timestamp for video frame `i`.
4. Interpolate `lowdim/tracker.npz` and `lowdim/encoder_*.npz` onto those
   camera timestamps.

`wall0` and `perf0` in `metadata.json` are the anchors used to build the shared
host timebase. Downstream code usually does not need them once timestamps are
already saved.
"""


def _import_realsense():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment. "
            "Use a Python environment with pyrealsense2 installed."
        ) from exc
    return rs


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required for RGB video recording.") from exc
    return cv2


def _host_time(wall0: float | None, perf0: float | None) -> float:
    if wall0 is None or perf0 is None:
        return time.time()
    return wall0 + (time.perf_counter() - perf0)


def _devices(rs, *, attempts: int = 3, backoff_s: float = 2.0) -> list:
    """Enumerate connected RealSense devices, skipping platform cameras.

    ``query_devices()`` can transiently raise ``RuntimeError`` when a device
    is mid-recovery / FW-update (e.g. "Failed to create FW update device").
    Retry with backoff before giving up — the glitch usually clears on the
    next enumeration (a fresh ``rs.context()`` each attempt).
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            devices = []
            for dev in rs.context().query_devices():
                name = dev.get_info(rs.camera_info.name)
                if name.lower() == "platform camera":
                    continue
                devices.append(dev)
            return devices
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(
                f"[RS] device enumeration failed (attempt {attempt}/{attempts}): {exc}",
                flush=True,
            )
            if attempt < attempts:
                time.sleep(backoff_s)
    raise RuntimeError(
        f"RealSense device enumeration failed after {attempts} attempts: {last_exc}"
    )


def _device_info(rs, dev) -> dict[str, str]:
    out = {}
    for field in ("name", "product_line", "serial_number", "firmware_version"):
        try:
            out[field] = dev.get_info(getattr(rs.camera_info, field))
        except Exception:
            pass
    return out


def _load_camera_roles(path: Path | None) -> tuple[dict[str, str], dict]:
    if path is None or not path.exists():
        return {}, {}
    data = json.loads(path.read_text(encoding="utf-8"))
    roles = data.get("roles", {})
    role_by_serial = {}
    for role, config in roles.items():
        serial = config.get("serial_number") if isinstance(config, dict) else config
        if serial:
            role_by_serial[str(serial)] = str(role)
    return role_by_serial, data


def _timestamp_domain(frame) -> str:
    getter = getattr(frame, "get_frame_timestamp_domain", None)
    if getter is None:
        getter = getattr(frame, "get_timestamp_domain", None)
    if getter is None:
        return "unknown"
    try:
        domain = getter()
    except Exception:
        return "unknown"
    name = getattr(domain, "name", None)
    if name:
        return str(name)
    return str(domain)


def _frame_metadata(rs, frame, metadata_name: str) -> float:
    metadata_enum = getattr(rs.frame_metadata_value, metadata_name, None)
    if metadata_enum is None:
        return float("nan")
    try:
        if not frame.supports_frame_metadata(metadata_enum):
            return float("nan")
        return float(frame.get_frame_metadata(metadata_enum))
    except Exception:
        return float("nan")


class _CameraRig:
    """Owns the RealSense pipelines. Started once and kept warm across episodes.

    Between episodes the caller should periodically call :meth:`drain` so the
    SDK buffers do not back up (and the sensors stay hot) while idle.
    """

    def __init__(
        self,
        rs,
        cv2,
        *,
        width: int,
        height: int,
        fps: int,
        max_cameras: int | None,
        camera_config_path: Path | None,
        wall0: float | None,
        perf0: float | None,
    ) -> None:
        self.rs = rs
        self.cv2 = cv2
        self.width = width
        self.height = height
        self.fps = fps
        self.max_cameras = max_cameras
        self.wall0 = wall0
        self.perf0 = perf0
        self.role_by_serial, self.camera_config = _load_camera_roles(camera_config_path)
        self.camera_config_path = camera_config_path
        self.pipelines: list = []
        # Per camera: (cam_idx, serial, info, role)
        self.cameras: list[tuple[int, str, dict, str | None]] = []

    def start(self) -> None:
        rs = self.rs
        devices = _devices(rs)
        if self.max_cameras is not None:
            devices = devices[: self.max_cameras]
        if not devices:
            raise SystemExit("ERROR: no RealSense devices found.")
        for cam_idx, dev in enumerate(devices):
            info = _device_info(rs, dev)
            serial = info.get("serial_number", f"camera_{cam_idx}")
            role = self.role_by_serial.get(serial)
            print(f"[RS-REC] cam{cam_idx} role={role} {info}", flush=True)
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(
                rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps
            )
            pipeline = rs.pipeline()
            pipeline.start(cfg)
            self.pipelines.append(pipeline)
            self.cameras.append((cam_idx, serial, info, role))
        time.sleep(1.0)  # warmup: let AE/AWB settle before any episode

    def drain(self, *, timeout_ms: int = 1000) -> None:
        """Pull and discard one frameset per camera to keep pipelines fresh."""
        for pipeline in self.pipelines:
            try:
                pipeline.wait_for_frames(timeout_ms=timeout_ms)
            except Exception:
                pass

    def stop(self) -> None:
        for pipeline in self.pipelines:
            try:
                pipeline.stop()
            except Exception:
                pass
        print("[RS-REC] pipelines stopped", flush=True)


class _EpisodeWriter:
    """Records one episode to ``output_dir`` using a started :class:`_CameraRig`."""

    def __init__(self, rig: _CameraRig, output_dir: Path) -> None:
        self.rig = rig
        self.output_dir = output_dir
        import av

        self._av = av
        self.cam_dirs: list[Path] = []
        self.writers: list = []
        self.camera_meta: list[dict] = []
        self.host_timestamps: list[list[float]] = []
        self.device_timestamps: list[list[float]] = []
        self.timestamp_domains: list[list[str]] = []
        self.frame_counters: list[list[float]] = []
        self.metadata_values: dict[str, list[list[float]]] = {
            output_name: [] for output_name in METADATA_FIELDS
        }
        self.receive_durations_ms: list[list[float]] = []
        self.last_images: dict[int, np.ndarray] = {}

        output_dir.mkdir(parents=True, exist_ok=True)
        for cam_idx, serial, info, role in rig.cameras:
            cam_dir = output_dir / f"cam{cam_idx}"
            cam_dir.mkdir(parents=True, exist_ok=True)
            self.cam_dirs.append(cam_dir)
            video_path = cam_dir / "color.mp4"
            container = av.open(str(video_path), mode="w")
            stream = container.add_stream("h264", rate=rig.fps)
            stream.width = rig.width
            stream.height = rig.height
            stream.pix_fmt = "yuv420p"
            stream.options = {
                "crf": "18",
                "preset": "veryfast",
                "profile": "high",
            }
            self.writers.append((container, stream))
            self.camera_meta.append(
                {
                    "camera_index": cam_idx,
                    "role": role,
                    "serial_number": serial,
                    "info": info,
                    "video_path": str(video_path),
                }
            )
            self.host_timestamps.append([])
            self.device_timestamps.append([])
            self.timestamp_domains.append([])
            self.frame_counters.append([])
            for per_field_values in self.metadata_values.values():
                per_field_values.append([])
            self.receive_durations_ms.append([])

    def capture_round(self) -> None:
        """Pull one frame from every camera and append it to this episode."""
        rs = self.rig.rs
        for cam_idx, pipeline in enumerate(self.rig.pipelines):
            recv_start = _host_time(self.rig.wall0, self.rig.perf0)
            frameset = pipeline.wait_for_frames(timeout_ms=5000)
            recv_end = _host_time(self.rig.wall0, self.rig.perf0)
            color_frame = frameset.get_color_frame()
            if not color_frame:
                continue
            image = np.asanyarray(color_frame.get_data())
            container, stream = self.writers[cam_idx]
            frame = self._av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image), format="bgr24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
            self.last_images[cam_idx] = image
            self.host_timestamps[cam_idx].append(
                recv_start + 0.5 * (recv_end - recv_start)
            )
            self.device_timestamps[cam_idx].append(color_frame.get_timestamp() / 1000.0)
            self.timestamp_domains[cam_idx].append(_timestamp_domain(color_frame))
            self.frame_counters[cam_idx].append(
                _frame_metadata(rs, color_frame, "frame_counter")
            )
            for output_name, metadata_name in METADATA_FIELDS.items():
                self.metadata_values[output_name][cam_idx].append(
                    _frame_metadata(rs, color_frame, metadata_name)
                )
            self.receive_durations_ms[cam_idx].append((recv_end - recv_start) * 1000.0)

    def finalize(self, *, duration: float | None = None) -> int:
        """Flush writers, save per-camera arrays + metadata.json. Returns frames."""
        rig = self.rig
        cv2 = rig.cv2
        for container, stream in self.writers:
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        for cam_idx, cam_dir in enumerate(self.cam_dirs):
            np.save(cam_dir / "color_timestamps.npy", np.asarray(self.host_timestamps[cam_idx]))
            np.save(cam_dir / "device_timestamps.npy", np.asarray(self.device_timestamps[cam_idx]))
            np.save(cam_dir / "timestamp_domain.npy", np.asarray(self.timestamp_domains[cam_idx]))
            np.save(cam_dir / "frame_counter.npy", np.asarray(self.frame_counters[cam_idx]))
            for output_name, per_camera_values in self.metadata_values.items():
                np.save(cam_dir / f"{output_name}.npy", np.asarray(per_camera_values[cam_idx]))
            np.save(
                cam_dir / "receive_durations_ms.npy",
                np.asarray(self.receive_durations_ms[cam_idx]),
            )
            if cam_idx in self.last_images:
                cv2.imwrite(str(cam_dir / "sample.png"), self.last_images[cam_idx])
            print(
                f"[RS-REC] cam{cam_idx} frames={len(self.host_timestamps[cam_idx])}",
                flush=True,
            )
        total_frames = sum(len(ts) for ts in self.host_timestamps)
        metadata = {
            "schema": "realsense_rgb_raw_v2",
            "metadata_readme": "README_metadata.md",
            "video_codec": "h264",
            "video_codec_options": {
                "crf": "18",
                "preset": "veryfast",
                "profile": "high",
            },
            "width": rig.width,
            "height": rig.height,
            "fps": rig.fps,
            "duration": duration,
            "wall0": rig.wall0,
            "perf0": rig.perf0,
            "camera_config": (
                str(rig.camera_config_path) if rig.camera_config_path else None
            ),
            "camera_roles": dict(rig.role_by_serial),
            "camera_config_data": rig.camera_config,
            "timestamp_files": {
                "color_timestamps.npy": "host midpoint timestamp, seconds",
                "device_timestamps.npy": "RealSense frame.get_timestamp(), seconds",
                "timestamp_domain.npy": "RealSense timestamp domain per frame",
                "frame_counter.npy": "RealSense frame counter metadata, raw SDK value",
                "receive_durations_ms.npy": "host wait_for_frames duration, milliseconds",
                **{
                    f"{output_name}.npy": (
                        f"RealSense {metadata_name} metadata, raw SDK value"
                    )
                    for output_name, metadata_name in METADATA_FIELDS.items()
                },
            },
            "cameras": self.camera_meta,
            "field_descriptions": {
                "schema": "Metadata schema version for this camera episode.",
                "video_codec": "Codec used for the recorded MP4 files.",
                "video_codec_options": "Encoder settings used when writing the video.",
                "width": "Recorded color frame width in pixels.",
                "height": "Recorded color frame height in pixels.",
                "fps": "Nominal recording frame rate.",
                "duration": "Episode duration in seconds, if known.",
                "wall0": "Shared host-time anchor from the parent process.",
                "perf0": "Monotonic clock anchor used with wall0 to reconstruct host time.",
                "camera_config": "Path to the camera role mapping config, if provided.",
                "camera_roles": "Resolved serial number to role mapping.",
                "camera_config_data": "Raw camera config JSON used for this episode.",
                "timestamp_files": "Human-readable meanings of the saved timing arrays.",
                "cameras": "Per-camera metadata entries for each cam* folder.",
            },
        }
        (self.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        _write_metadata_readme(self.output_dir / "README_metadata.md", CAMERA_METADATA_README)
        return total_frames


def _write_marker(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_metadata_readme(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _serve(rig: _CameraRig, ready_file: Path | None) -> None:
    """Command-driven loop: keep pipelines warm, record episodes on demand.

    Reads newline commands on stdin:
        START <episode_dir>   begin recording into <episode_dir>
        STOP                  finalize the current episode
        QUIT                  finalize (if recording) and exit

    On START the recorder writes ``<episode_dir>/realsense_ready.json`` once the
    first frames are flowing; on STOP/finalize it writes
    ``<episode_dir>/realsense_done.json`` with the frame count. The orchestrator
    polls those files to stay in lock-step.
    """
    import queue
    import threading

    cmd_q: "queue.Queue[str]" = queue.Queue()

    def reader() -> None:
        for line in sys.stdin:
            cmd_q.put(line.strip())
        cmd_q.put("QUIT")  # stdin closed → shut down

    threading.Thread(target=reader, daemon=True).start()

    if ready_file is not None:
        _write_marker(ready_file, {"ready_timestamp": _host_time(rig.wall0, rig.perf0)})
    print("[RS-REC] serve ready", flush=True)

    episode: _EpisodeWriter | None = None
    episode_dir: Path | None = None
    episode_start: float | None = None
    while True:
        # Drive the cameras: record when an episode is active, else drain.
        if episode is not None:
            try:
                episode.capture_round()
            except Exception as exc:  # noqa: BLE001
                print(f"[RS-REC] capture error: {exc}", flush=True)
            if episode_dir is not None and not (episode_dir / "realsense_ready.json").exists():
                if any(episode.host_timestamps):
                    _write_marker(
                        episode_dir / "realsense_ready.json",
                        {"ready_timestamp": _host_time(rig.wall0, rig.perf0)},
                    )
        else:
            rig.drain()

        try:
            cmd = cmd_q.get_nowait()
        except queue.Empty:
            continue
        if not cmd:
            continue
        verb, _, arg = cmd.partition(" ")
        verb = verb.upper()
        if verb == "START":
            if episode is not None:
                print("[RS-REC] already recording; ignoring START", flush=True)
                continue
            episode_dir = Path(arg.strip())
            episode = _EpisodeWriter(rig, episode_dir)
            episode_start = _host_time(rig.wall0, rig.perf0)
            print(f"[RS-REC] START {episode_dir}", flush=True)
        elif verb == "STOP":
            if episode is None:
                print("[RS-REC] not recording; ignoring STOP", flush=True)
                continue
            duration = (
                _host_time(rig.wall0, rig.perf0) - episode_start
                if episode_start is not None
                else None
            )
            frames = episode.finalize(duration=duration)
            assert episode_dir is not None
            _write_marker(
                episode_dir / "realsense_done.json",
                {"frames": frames, "duration": duration},
            )
            print(f"[RS-REC] STOP {episode_dir} frames={frames}", flush=True)
            episode = None
            episode_dir = None
            episode_start = None
        elif verb == "QUIT":
            if episode is not None:
                duration = (
                    _host_time(rig.wall0, rig.perf0) - episode_start
                    if episode_start is not None
                    else None
                )
                frames = episode.finalize(duration=duration)
                if episode_dir is not None:
                    _write_marker(
                        episode_dir / "realsense_done.json",
                        {"frames": frames, "duration": duration},
                    )
            print("[RS-REC] QUIT", flush=True)
            break
        else:
            print(f"[RS-REC] unknown command: {cmd!r}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-cameras", type=int, default=None)
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=DEFAULT_CAMERA_CONFIG_PATH,
        help=(
            "JSON RealSense role config used to label cameras by serial number "
            f"(default: {DEFAULT_CAMERA_CONFIG_PATH})."
        ),
    )
    parser.add_argument("--ready-file", type=Path, default=None)
    parser.add_argument("--wall0", type=float, default=None)
    parser.add_argument("--perf0", type=float, default=None)
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Command-driven mode: keep pipelines warm and record episodes on "
        "START/STOP/QUIT commands read from stdin (for episodic polling).",
    )
    args = parser.parse_args()

    rs = _import_realsense()
    cv2 = _import_cv2()
    rig = _CameraRig(
        rs,
        cv2,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_cameras=args.max_cameras,
        camera_config_path=args.camera_config,
        wall0=args.wall0,
        perf0=args.perf0,
    )
    try:
        rig.start()
        if args.serve:
            _serve(rig, args.ready_file)
            return

        # One-shot mode (unchanged behaviour): record a single fixed-duration
        # episode straight into --output-dir.
        if args.duration <= 0:
            raise ValueError("--duration must be positive")
        if args.output_dir is None:
            raise SystemExit("ERROR: --output-dir is required in one-shot mode.")
        episode = _EpisodeWriter(rig, args.output_dir)
        if args.ready_file is not None:
            _write_marker(
                args.ready_file,
                {"ready_timestamp": _host_time(args.wall0, args.perf0)},
            )
        end_time = time.time() + args.duration
        while time.time() < end_time:
            episode.capture_round()
        episode.finalize(duration=args.duration)
    finally:
        rig.stop()
        print("[RS-REC] stopped", flush=True)


if __name__ == "__main__":
    main()
