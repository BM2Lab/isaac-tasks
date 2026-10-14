# V2 local simulator endpoint

`scripts/serve_visual_reach.py` exposes the V1 FR3 visual-reaching task to a
separate learning process over a private local Unix-domain socket. It imports
contracts but never imports learning, LeRobot, Transformers or hardware drivers.
The action/scene/controller contract remains [V1](v1_visual_reach.md).

The server supports one simulated environment and one connected client. It
requires exact contract handshaking; observation identity includes a session UUID,
episode counter, policy step and simulation-clock nanoseconds. Only measured q7,
RGB and language are policy inputs. Action chunks are physical meter-valued rows;
the server validates them on acceptance and at each execution. No smoothing or
automatic action fallback is introduced.

`hello`, `observe`, `submit`, `advance`, `status`, `reset`, `close` use protocol 1,
length-prefixed JSON capped at 2 MiB. Images are lossless uint8 RGB/HWC base64.
Reset—including Isaac Lab automatic reset—invalidates the queue and old episode
context. The returned terminal metrics reference the old episode; returned images
and state reference the new one. Disconnection/idle timeout stops this simulation
session, clears the queue, and removes its own socket endpoint.

Run with the pinned Isaac direnv environment and an explicit audited asset:

```bash
direnv exec . python -u scripts/serve_visual_reach.py \
  --contract /path/to/fr3_visual_reach_v1.json \
  --socket /short/private/ext4/runtime/bridge.sock \
  --device cpu --viz none --idle-timeout 300
```

The parent directory must be owned by the user, mode 0700; the server creates a
missing private parent, sets socket mode 0600 and refuses existing endpoints.
Use POSIX runtime storage, not source/datasets/NTFS. A client must connect before
the idle timeout and use the exact same typed contract. Wall-clock inference
pauses simulation; 20 Hz is simulated time, not an asynchronous real-time promise.

For finite mock verification, add `--test-episode-length-s 0.2` to force an
automatic timeout quickly, then run the robot-learning repository's
`scripts/run_local_sim_smoke.py` in its **own** environment with matching
`--socket` and `--contract`. Do not install learning into Isaac. On this host the
complete two-terminal commands are documented in robot-learning's
`docs/v2_local_sim.md`. A short mock timeout is expected; it does not measure
goal-reaching capability.

New bridge modules:

- `bridges/action_queue.py`: exact capture matching, ordered chunk consumption,
  per-execution deadline and reset invalidation.
- `bridges/local_session.py`: single-env Gym adapter, capture encoding and
  terminal/new-episode identity separation; runtime/model/contract receipt.
- `bridges/local_server.py`: bounded RPC dispatch, private socket lifecycle and
  duplicate-request rejection.

V2 verification on 2026-10-14 passed 15 task tests, the separate client/contract
tests and real two-process mock rollout, including automatic/manual reset,
stale/expired/duplicate rejection, fresh frames and clean endpoint cleanup.
No legacy task files or simulator dependencies were changed. Pretrained SmolVLA
native actions remain incompatible; adaptation and dataset collection are later gates.
