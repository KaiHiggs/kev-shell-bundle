#!/usr/bin/env bash
# Linux + ROCm. Installs the pinned upstream and verifies that torch sees the GPU.
# Idempotent: repairs a missing .git, a missing origin, a partial checkout, or a venv without torch.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
COMMIT=2074cc44256200bb37540103fa4572901faa10f3
REPO=https://github.com/jaredpalmer/kev
ROCM_WHEELS=https://download.pytorch.org/whl/rocm6.3

echo "python: $(command -v python3) $(python3 -V 2>&1 | awk '{print $2}')"
if [ -f /etc/arch-release ]; then
  echo 'Arch detected.'
  pacman -Q rocm-hip-sdk >/dev/null 2>&1 || { echo 'ERROR: missing ROCm: sudo pacman -S --needed rocm-hip-sdk rocm-opencl-runtime uv' >&2; exit 1; }
  pacman -Q python-pytorch-rocm 2>/dev/null || echo 'note: python-pytorch-rocm not installed; the ROCm wheel will be used instead'
  id -nG | tr ' ' '\n' | grep -qxE 'render|video' || echo 'WARNING: add yourself to render,video and re-login: sudo usermod -aG render,video $USER'
  rocminfo 2>/dev/null | grep -m1 -o 'gfx[0-9a-z]*' | sed 's/^/rocminfo device: /' || true
fi

# --- upstream repo: create, repair the remote, fetch the pinned commit -------------------------------------
if [ ! -d upstream/.git ]; then
  rm -rf upstream
  # The pinned commit is main's tip: depth 1 is one commit of blobs instead of ~180 MB of history.
  echo "cloning $REPO (shallow)"
  git clone -q --depth 1 "$REPO" upstream
else
  if ! git -C upstream remote get-url origin >/dev/null 2>&1; then
    echo 'upstream has no origin remote; adding it'
    git -C upstream remote add origin "$REPO"
  fi
fi
git -C upstream remote set-url origin "$REPO"
if ! git -C upstream rev-parse --verify -q "$COMMIT^{commit}" >/dev/null; then
  echo "fetching pinned commit $COMMIT"
  if ! git -C upstream fetch -q --depth 1 origin "$COMMIT"; then
    echo 'fetch by sha refused; deepening the clone instead'
    git -C upstream fetch -q --unshallow origin || git -C upstream fetch -q --tags origin
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

# Reuse the distro's ROCm torch only when the venv can actually import it; otherwise use the ROCm wheel.
SYS_TORCH_OK=0
if python3 -c 'import torch' 2>/dev/null; then SYS_TORCH_OK=1; fi
if [ "$SYS_TORCH_OK" = 1 ]; then
  echo "system python has torch: $(python3 -c 'import torch;print(torch.__version__, torch.version.hip)')"
  uv venv --system-site-packages --allow-existing .venv
  uv sync --frozen --extra serve --no-dev --no-install-package torch
else
  uv venv --allow-existing .venv
  uv sync --frozen --extra serve --no-dev --no-install-package torch
fi

if ! .venv/bin/python -c 'import torch' 2>/dev/null; then
  echo "torch not importable in .venv; installing the ROCm wheel"
  uv pip install --python .venv/bin/python --index-url "$ROCM_WHEELS" torch
fi

.venv/bin/python - <<'PY'
import torch
print('torch', torch.__version__, 'hip', torch.version.hip)
assert torch.cuda.is_available(), ('torch does not see a ROCm device: check rocminfo, the render,video groups, '
                                   'and that ROCm supports this gfx target')
p = torch.cuda.get_device_properties(0)
print('device', p.name, f'{p.total_memory/2**30:.1f} GiB')
x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
print('bf16 matmul ok', float((x @ x).sum()))
PY
