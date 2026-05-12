# UR3 Block Push Inference Server for VersatIL

Standalone simulation server for evaluating VersatIL policies on the QFAT UR3
Block Push task. A VersatIL client process connects over ZMQ, receives
low-dimensional state observations, and sends back 2D end-effector target
actions; the server drives the simulator, records rollouts, and computes
evaluation metrics.

The simulator dynamics and task logic are based on QFAT's UR3 task, with the
simulator code vendored under `gym_custom/`.
This is a separate MuJoCo UR3 environment and is not derived from the IBC
XArm BlockPush implementation.
The `versatil_inference/` package is only the environment-side wrapper. The
policy client lives in the VersatIL codebase and is run with
`python -m versatil.endpoints.test`.

The server uses raw simulator state/action values by default, because VersatIL
normalizes observations and unnormalizes actions inside the policy client before
sending actions to the environment server.

## Layout

- `gym_custom/` - UR3 simulator code and assets adapted from QFAT.
- `versatil_inference/` - ZMQ server, parallel episode manager, rollout
  recorder, and evaluation entry point.

## Install

Install Miniforge with `mamba` available if needed.

```bash
cd /mnt/cluster/workspaces/mazzalore/ur3_blockpush
mamba env create -f environment.yml
mamba run -n ur3_blockpush bash -lc \
  'UV_PROJECT_ENVIRONMENT=$MAMBA_ROOT_PREFIX/envs/ur3_blockpush uv sync'
```

`uv sync` installs `versatil-constants>=0.2.1`, which provides the shared
UR3 wire protocol constants used by both this server and VersatIL.

For headless rendering, set the MuJoCo backend before running:

```bash
export MUJOCO_GL=egl
```

## Run

Start the simulator server:

```bash
cd /mnt/cluster/workspaces/mazzalore/ur3_blockpush
mamba activate ur3_blockpush
python -m versatil_inference.run_evaluation \
  --num_trials 50 \
  --max_parallel_envs 10 \
  --port 5556 \
  --output_folder /mnt/cluster/workspaces/mazzalore/eval/ur3_blockpush \
  --use_wandb false
```

Then run the policy client from the VersatIL checkout against that server:

```bash
cd /path/to/VersatIL
mamba run -n versatil python -m versatil.endpoints.test \
  --checkpoint_path /path/to/checkpoint_dir \
  --checkpoint_name latest-999.ckpt \
  --model_server_address 127.0.0.1 \
  --model_server_port 5556 \
  --temporal_aggregation \
  --max_steps 1000
```

Use `--record_video true` on the server if rollout AVI files are needed.
Trajectory CSV files and `results.csv` are always written under the configured
`output_folder`.

## Wire Protocol

Observation keys match the VersatIL UR3 configs:

| Key | Shape | Source |
|---|---:|---|
| `ur3_ee_pos` | `(2,)` | end-effector xy position |
| `ur3_block1_pos` | `(2,)` | first block xy position |
| `ur3_block2_pos` | `(2,)` | second block xy position |

Actions are received as structured VersatIL action dictionaries and flattened
to the raw `ur3_ee_target_action` 2D target before stepping the simulator.

## Metrics

The server reports:

- `reward`: total episode reward, one point for each achieved UR3 goal.
- `p1`: whether block 1 reached its target.
- `p2`: whether block 2 reached its target.
- `mean_goals (/2)`: `mean(p1 + p2)`.
- `behavior_order_entropy`: effective number of successful two-goal orders
  over `1->2` and `2->1`.

## Notes

- Observation keys match VersatIL UR3 configs:
  `ur3_ee_pos`, `ur3_block1_pos`, and `ur3_block2_pos`.
- Action input is the raw 2D end-effector target for
  `ur3_ee_target_action`.
- Optional QFAT-style normalized simulator I/O is still available through
  `--normalize_io true --stats_path /path/to/data_stats.json`, but this should
  not be used with standard VersatIL checkpoints unless you intentionally want
  to bypass the policy client's normalizer contract.

## Credits

The UR3 task, `gym_custom/` simulator tree, and normalized wrapper convention
are adapted from [ziyadsheeba/qfat](https://github.com/ziyadsheeba/qfat), the
official implementation of *Quantization-Free Autoregressive Action
Transformer*. Vendored UR driver components retain their original notices,
including python-urx LGPL-3.0 files under
`gym_custom/envs/real/ur/drivers/urx/` and MIT-licensed URplus driver files
under `gym_custom/envs/real/ur/drivers/URplus/`.

## License

This repository is distributed under Apache-2.0. Vendored third-party files may
carry different file-level licenses; those notices are retained in place.
