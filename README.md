# Fine-tune Kev-0.8B on the shell-call dataset

Self-contained bundle. Pinned upstream, 2,350 labelled records, a trainer command at the full 2,048-token
context, and a scoring script. No network access after the first download.

---

## First: Unsloth on the Windows machine will not work here

Two independent blockers, either one is fatal:

1. **No ROCm PyTorch on Windows.** ROCm is a Linux userspace. The Windows "HIP SDK" does not ship `torch`
   wheels, and ROCm inside WSL2 does not support gfx1030 (the 6800 XT's chip). `torch.cuda.is_available()`
   is therefore `False` on Windows for that card, whatever the installer says.
2. **Unsloth cannot train this model anyway.** It wraps Hugging Face `Trainer`/SFT for standard causal LMs.
   Kev's model is a decision model: a pointer head over per-question option branches, with a packed
   state+branch layout and its own loss. Porting that into Unsloth's kernels would be a research project,
   and the result would no longer be the model that is deployed.

Kev itself supports ROCm (`README`: "CUDA or ROCm if you have a GPU"; the device probe falls back
CUDA/ROCm → MPS → CPU). So the shortest path is a Linux install on that same machine.

## What to do on the 6800 XT box

One of:

- **Arch (chosen):** `sudo pacman -S --needed rocm-hip-sdk rocm-opencl-runtime python-pytorch-rocm uv`, then
  `sudo usermod -aG render,video $USER` and re-login. `bootstrap.sh` detects Arch, uses the distro's
  `python-pytorch-rocm` (built against the system ROCm, so no version skew) and skips the PyPI torch wheel.
- **Any other Linux (Ubuntu/Debian/Fedora):** AMD's ROCm 6.3 install; `bootstrap.sh` then pulls the ROCm
  torch wheel from PyTorch's index automatically.
- **No Linux wanted?** Rent one NVIDIA GPU (L4 / A10G / 4090) for an hour. This bundle is unchanged — CUDA
  is the default path.

16 GB VRAM is enough for 0.8B LoRA at 2,048 tokens (`~6–9 GB` expected). On Arch, `rocminfo | grep gfx`
must print `gfx1030` before training. If `torch.cuda.is_available()` is still false after bootstrap, the
usual cause is a session that predates the `render`/`video` group change.

`bootstrap.sh` verifies the GPU and prints the device name and memory before touching the dataset. If it
fails, nothing else runs.

## Run it

```sh
tar xzf kev-shell-bundle.tar.gz && cd kev-shell-bundle
bash scripts/bootstrap.sh          # pinned upstream + ROCm torch + a real bf16 matmul on the GPU
bash scripts/train.sh              # 0.8B LoRA, 2048 ctx, 2 epochs -> runs/kev-shell-0.8b
bash scripts/eval.sh               # serves the checkpoint, scores holdout-512 + calibration-512
```

Expected: **1,418 train records**, ~177 optimizer steps per epoch (`accum 8`), so **1–2 hours** on a 6800 XT.
Logs: `runs/train.log`, `runs/serve.log`, scores in `runs/eval-holdout/summary.json`.

## What the numbers mean

`eval-finetune.mjs` posts each held-out command to the local endpoint with the **production** question set
and compares the answer to the label. It reports per-question accuracy and, for the safety-relevant head,
how many true `deny` calls were **not** predicted deny:

```
deny_truth 41 released_as_non_deny 7
```

The second number is the one that matters. A model that is accurate but releases denies is worse than the
rules. Compare against the untouched pretrained baseline before deciding anything.

Splits are by **run**: `holdout-512.jsonl` is three runs the model never saw (mentor-tribe, nymphes-ocr,
ssle2); `calibration-512.jsonl` is four more. Do not tune on the holdout twice.

## Long commands

`*-long.jsonl` holds the 260 rows whose state exceeds 512 tokens (57 of them in holdout). They are shipped
so you can score them, but the serving policy for anything past the trained context is **review**, not
truncate: a truncated command is exactly the input a classifier must refuse to judge.

## If the trainer dies

| symptom | fix |
|---|---|
| `bf16` kernel crash on ROCm | add `--weights_dtype fp32 --checkpointing 1` to `train.sh` |
| HIP out of memory | keep `--checkpointing 1`, or `--lora_targets attn` (fewer modules) |
| `torch.cuda.is_available()` false | ROCm not installed for gfx1030, or the wrong wheel; `rocminfo` must list the card |
| HF download stalls | pre-download the two pinned repos, then re-run with `HF_HUB_OFFLINE=1` |

Send back: `runs/train.log` tail, `runs/eval-holdout/summary.json`, `runs/eval-calibration/summary.json`.

## Files

```
data/                train|calibration|holdout (full), *-512 (trainable), *-long (>512 tokens)
eval/                eval-finetune.mjs, questions.mjs (production definitions), policy.mjs
scripts/             bootstrap.sh, train.sh, serve.sh, eval.sh
pin.txt              upstream commit, base revision, tokenizer checkpoint, dataset limits
MANIFEST.sha256      checksum of every file above
```

Dataset provenance: 16 fleet stores, 7,998 real bash calls, 620 behaviour strata, 169 hand-classified
(143 corrected), 368 assistant-authored seed templates. Every real-call stratum carries a recorded decision
in the source repository's `data/approvals.jsonl`.
