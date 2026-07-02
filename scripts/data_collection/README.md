# Pico + RealSense + Encoder Data Collection

This package records synchronized episodes from:

- Pico left/right controllers through `XrClient`
- up to three RealSense RGB streams
- left/right magnetic encoders

All streams use a shared host timebase. Pico and encoder samples include
`read_start_timestamp` and `read_end_timestamp` so read latency can be measured
after collection.

## Run

From the repo root:

```bash
python -m scripts.data_collection.collect_session
```

Useful flags:

```bash
python -m scripts.data_collection.collect_session \
  --camera-width 640 --camera-height 480 --camera-fps 30 --max-cameras 3 \
  --pico-frequency 120 \
  --encoder-frequency 30 \
  --encoder-raw-only
```

Hotkeys:

- `c`: start an episode
- `q`: stop/save the current episode or quit when idle

## Output

```text
data/sessions/session_YYYYMMDD_HHMMSS/
  metadata.json
  realsense_serve.log
  episode_000/
    metadata.json
    cameras/
      metadata.json
      cam0/color.mp4
      cam0/color_timestamps.npy
      cam1/...
      cam2/...
    lowdim/
      pico_controllers.npz
      encoder_left.npz
      encoder_right.npz
```

## Pico Fields

`pico_controllers.npz` contains:

- `timestamp`
- `read_start_timestamp`
- `read_end_timestamp`
- `xr_timestamp_ns`
- `left_pose`
- `right_pose`
- `left_valid`
- `right_valid`
- `left_trigger`
- `right_trigger`
- `left_grip`
- `right_grip`
- `left_joystick`
- `right_joystick`

`timestamp` is the midpoint between `read_start_timestamp` and
`read_end_timestamp`.
