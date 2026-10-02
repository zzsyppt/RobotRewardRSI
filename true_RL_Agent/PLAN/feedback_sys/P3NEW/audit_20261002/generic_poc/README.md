# Generic description proof of concept (2026-10-02)

Audit artefacts for `../../GENERIC_DESCRIPTION_DESIGN_2026-10-02_ZH.md`. Nothing here is production code.

Regenerate the inputs (read-only on the server, CPU only, nothing written there; about 3 min per Stair run):

```bash
ssh humanoid-server 'cd /tmp && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /data/yuyangliu/envs/isaaclab6_validation/bin/python -B -' < server_stair_dump.py > stair_steps.npz
```

```bash
ssh humanoid-server 'cd /tmp && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /data/yuyangliu/envs/isaaclab6_validation/bin/python -B -' < server_stair_trunk.py > stair_trunk.npz
```

```bash
ssh humanoid-server 'cd /tmp && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /data/yuyangliu/envs/isaaclab6_validation/bin/python -B -' < server_push_dump.py > push_steps.npz
```

Then, in this directory with the three `.npz` files and `../stair_modes_out.json` copied alongside:

- `run_generic_push.py`: success envelope, excursions and grouping on Push.
- `run_generic_stair.py` then `analyze_generic_stair.py`: the same core on Stair, compared with the audit's four forms.
- `analyze_trunk.py`: clearance of trunk sample points above the stair surface, by form and in successes.

`generic_describe.py` is the shared core. It contains no task or robot names. The `.npz` dumps (about 62 MB) are not stored in the repository.
