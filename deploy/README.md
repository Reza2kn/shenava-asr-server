# Deployment

The server runs as a systemd service (`shenava-asr.service`) on the CUDA host.

## NVIDIA / CUDA prerequisites (CUDA 13 driver, e.g. RTX 5080 = sm_120)

tract-cuda uses NVRTC to JIT-compile kernels **at runtime** for the GPU's architecture.
It needs:

1. A CUDA toolkit **include tree** at `/usr/local/cuda/include` with the CCCL/libcudacxx
   headers (`cuda/std/cstdint`, `cuda_fp16.h`, …) — NVRTC compiles against these.
2. A CUDA 13.0 `libnvrtc.so` (unversioned) resolvable through `LD_LIBRARY_PATH` — the
   `cuda-13000` cudarc feature dlopens `libnvrtc.so` / `libnvrtc64_13.so`. CUDA ≤12.6's
   NVRTC **cannot compile for sm_120** (Blackwell) — use the 13.0 toolkit.

On Stallion the working setup is:

```
export LD_LIBRARY_PATH=/home/rezo/tract-cuda13-libs:\
  /home/rezo/.local/lib/python3.14/site-packages/nvidia/cu13/lib:\
  /opt/bina-flax-site/nvidia/cudnn/lib:...\  # cublas, cufft, cusparse, cusolver, nccl, cuda_runtime, nvjitlink, cuda_nvrtc
```

`/home/rezo/tract-cuda13-libs/libnvrtc.so` → CUDA 13.0 NVRTC (from the `nvidia-cu13` pip
package or conda `cuda-nvrtc-13.0.88`). NVRTC cubins are cached in
`~/.cache/tract/0.23.4-pre/cuda/13000/cubins` after the first startup.

## First-run on a host

```bash
git clone https://github.com/Reza2kn/shenava-asr-server
cd shenava-asr-server
HF_TOKEN="hf_..." ./run.sh   # downloads offline+streaming assets and Nemotron when authorized,
                              # builds native Rust features (+ CUDA if nvidia-smi), serves :3000
```

For systemd (Stallion):

```bash
cp deploy/shenava-asr.service /etc/systemd/system/
systemctl daemon-reload && systemctl enable --now shenava-asr
```

## Verifying CUDA is active

Startup log should show:

```
tract backend: gpu-or-cpu (cuda available)
tract-cuda: device 0 = "NVIDIA GeForce RTX 5080 ..."
Compiling FlashAttn to .../cubins/...    # once, then cached
```

If it says `(cpu available)`, CUDA's `check()` failed — see the cuda/README or enable
`RUST_LOG=tract_cuda=debug`.
