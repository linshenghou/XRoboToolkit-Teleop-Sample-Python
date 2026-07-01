#!/usr/bin/env bash
set -euo pipefail

ENV_DIR=".venv-calibration"
PYTHON_VERSION="3.10"
INSTALL_SYSTEM_PACKAGES=0

usage() {
    cat <<'EOF'
Usage:
  bash scripts/calibration/setup_uv_calibration_ubuntu.sh [options]

Options:
  --env-dir PATH          Virtualenv path. Default: .venv-calibration
  --python VERSION        Python version for uv venv. Default: 3.10
  --system-packages       Install common Ubuntu runtime packages with apt/sudo.
  -h, --help              Show this help.

This sets up the minimal environment for scripts/calibration:
  - numpy
  - opencv-python
  - pyrealsense2
  - xrobotoolkit_sdk, via XRoboToolkit-PC-Service-Pybind/setup_ubuntu.sh
  - this repo, editable, without installing the full simulation/hardware dependency set
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env-dir)
            ENV_DIR="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --system-packages)
            INSTALL_SYSTEM_PACKAGES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "This setup script is intended for Ubuntu/Linux." >&2
    exit 1
fi

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" != "ubuntu" ]]; then
        echo "Warning: expected Ubuntu, detected ${PRETTY_NAME:-unknown Linux}."
    else
        echo "Detected ${PRETTY_NAME}."
    fi
fi

if ! command -v uv >/dev/null 2>&1; then
    cat >&2 <<'EOF'
uv is not installed.

Install it first, then rerun this script:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  source "$HOME/.local/bin/env"
EOF
    exit 1
fi

if [[ "$INSTALL_SYSTEM_PACKAGES" -eq 1 ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        libgl1 \
        libglib2.0-0 \
        python3-dev \
        v4l-utils
fi

echo "[1/5] Creating uv environment: ${ENV_DIR} (Python ${PYTHON_VERSION})"
uv venv --python "${PYTHON_VERSION}" "${ENV_DIR}"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

echo "[2/5] Installing Python calibration dependencies"
uv pip install --upgrade pip setuptools wheel
uv pip install numpy opencv-python pyrealsense2

echo "[3/5] Installing XRoboToolkit SDK binding"
mkdir -p dependencies
if [[ ! -d dependencies/XRoboToolkit-PC-Service-Pybind/.git ]]; then
    git clone https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git dependencies/XRoboToolkit-PC-Service-Pybind
else
    git -C dependencies/XRoboToolkit-PC-Service-Pybind pull --ff-only
fi
pushd dependencies/XRoboToolkit-PC-Service-Pybind >/dev/null
python - <<'PY'
from pathlib import Path

path = Path("setup_ubuntu.sh")
text = path.read_text()
text = text.replace("pip install pybind11 -y", "python -m pip install pybind11")
path.write_text(text)
PY
bash setup_ubuntu.sh
popd >/dev/null

echo "[4/5] Installing this repo for calibration imports"
uv pip install -e . --no-deps

echo "[5/5] Verifying imports"
python - <<'PY'
import cv2
import numpy
import pyrealsense2
import xrobotoolkit_sdk
from xrobotoolkit_teleop.common.xr_client import XrClient

print("cv2", cv2.__version__)
print("numpy", numpy.__version__)
print("pyrealsense2 OK")
print("xrobotoolkit_sdk OK")
print("xrobotoolkit_teleop OK")
PY

cat <<EOF

Done.

Activate this environment with:
  source ${ENV_DIR}/bin/activate

Run calibration from the repo root, for example:
  python scripts/calibration/handeye_sanity_check.py --controller right_controller
  python scripts/calibration/handeye_capture.py --record-dir calibration_data/exp01 --controller right_controller --square-size 0.025 --pattern 11 8 --pico-trigger auto
EOF
