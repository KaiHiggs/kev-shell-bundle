#!/usr/bin/env bash
# Linux + ROCm. Installs the pinned upstream and verifies that torch sees the 6800 XT.
# Idempotent: re-running repairs a missing .git, a missing origin, or a partial checkout.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
COMMIT=2074cc44256200bb37540103fa4572901faa10f3
REPO=https://github.com/jaredpalmer/kev

if [ -f /etc/arch-release ]; then
  echo 'Arch detected.'
  pacman -Q rocm-hip-sdk >/dev/null 2>&1 || { echo 'ERROR: missing ROCm: sudo pacman -S --needed rocm-hip-sdk rocm-opencl-runtime python-pytorch-rocm uv' >&2; exit 1; }
  id -nG | tr ' ' '\n' | grep -qxE 'render|video' || echo 'WARNING: add yourself to render,video and re-login: sudo usermod -aG render,video $USER'
  rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-z]*' | sed 's/^/rocminfo device: /' || true
fi

# --- upstream repo: create, repair the remote, fetch the pinned commit -------------------------------------
if [ ! -d upstream/.git ]; then
  rm -rf upstream
  echo "cloning $REPO"
  git clone -q "$REPO" upstream
else
  if ! git -C upstream remote get-url origin >/dev/null 2>&1; then
    echo 'upstream has no origin remote; adding it'
    git -C upstream remote add origin "$REPO"
  fi
fi
git -C upstream remote set-url origin "$REPO"
if ! git -C upstream rev-parse --verify -q "$COMMIT^{commit}" >/dev/null; then
  echo "fetching pinned commit $COMMIT"
  if ! git -C upstream fetch -q origin "$COMMIT"; then
    echo 'fetch by sha refused; fetching the default branch instead'
    git -C upstream fetch -q --tags origin
  fi
fi
if ! git -C upstream rev-parse --verify -q "$COMMIT^{commit}" >/dev/null; then
  echo "ERROR: commit $COMMIT not found in $REPO. Check network access, then re-run." >&2
  git -C upstream log --oneline -1 >&2 || true
  exit 1
fi
git -C upstream checkout -q "$COMMIT"
echo "upstream at $(git -C upstream rev-parse --short HEAD)"

# --- python env ---------------------------------------------------------------------------------------------
cd upstream
command -v uv >/dev/null || { echo 'ERROR: install uv first: sudo pacman -S --needed uv (or curl -LsSf https://astral.sh/uv/install.sh | sh)' >&2; exit 1; }

if [ -f /etc/arch-release ] && pacman -Q python-pytorch-rocm >/dev/null 2>&1; then
  echo 'using the distro python-pytorch-rocm (system ROCm build)'
  uv venv --system-site-packages --allow-existing .venv
  uv sync --frozen --extra serve --no-dev --no-install-package torch
else
  uv sync --frozen --extra serve --no-dev
  if ! .venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    # PyPI ships the CUDA wheel; ROCm wheels come from PyTorch's index.
    uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall torch
  fi
fi

.venv/bin/python - <<'PY'
import torch
print('torch', torch.__version__, 'hip', torch.version.hip)
assert torch.cuda.is_available(), 'torch does not see a ROCm device; check rocminfo and the render,video groups'
p = torch.cuda.get_device_properties(0)
print('device', p.name, f'{p.total_memory/2**30:.1f} GiB')
x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
print('bf16 matmul ok', float((x @ x).sum()))
PY
