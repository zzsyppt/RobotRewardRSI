# Phase -1 installation notes

Status: installation and A-D smoke validation complete (2026-06-20).

## Storage policy

All large files must live below `/data/yuyangliu`:

```text
/data/yuyangliu/AGENT             project workspace
/data/yuyangliu/envs              Python environments
/data/yuyangliu/cache             package and simulator caches
/data/yuyangliu/vendor            Isaac Lab and Unitree repositories
/data/yuyangliu/runs              training logs and checkpoints
```

Do not install Isaac Sim or training artifacts under `/home`.

## Initial compatibility target

- Python 3.11
- Isaac Sim 5.1.0
- PyTorch 2.7.0 + CUDA 12.8 wheels
- Isaac Lab v2.3.0 source: `3c6e67bb5c7ada942a6d1884ab69338f57596f77`
- Unitree RL Lab source: `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3`
- Unitree ROS G1 assets: `d6f13aad60320ce1d60a07b82a76b5a553f2a0a9`

Important compatibility pins are `setuptools==80.9.0`, `wheel==0.45.1`,
`typing_extensions==4.12.2`, `click==8.1.7`, `transformers==4.56.2`,
`huggingface_hub==0.34.4`, `typer==0.12.5`, `onnx==1.19.1`, and
`onnxscript==0.5.6`. Later-phase audit additions are `ninja==1.13.2` and
`jsonschema==4.25.1`.

The server lacks system `libGLU.so.1` and the account has no passwordless sudo.
The installer therefore extracts `libglu1-mesa` under
`/data/yuyangliu/vendor/system-libs` and the smoke scripts prepend that directory
to `LD_LIBRARY_PATH`.

The Unitree G1 assets use the official `unitree_ros` URDF fallback because the
server could not reach Hugging Face to download `unitree_model`. Apply
`setup/unitree_g1_urdf.patch` to the pinned Unitree RL Lab source.

See the Phase -1 completion log for evidence and artifact paths.

The complete clean-server procedure, activation settings, validation sequence,
lock files, API connectivity, TensorBoard setup, and recovery instructions are
in `setup/environment_reproduction.md`. Treat that document and the files below
`setup/locks/` as authoritative; this file is only a short status summary.
