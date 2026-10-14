# V3 expert label endpoint

The existing V2 simulator server now supports **opt-in** expert label generation
for offline visual-reaching demonstrations. Scene, IK controller, physical action
semantics and normal V2 queue behavior are unchanged.

Start `scripts/serve_visual_reach.py` with `--enable-expert-demonstrations` and
without the short V2 timeout override. The `expert_action` RPC requires the exact
current capture context and an empty action queue. It returns a one-row typed
ActionChunk and `label_source='privileged_scripted_ik_expert'`. The server still
applies nothing until the client submits and advances through the normal queue.
Default sessions reject this privileged RPC.

Target coordinates are read internally by the expert but never added to policy
observations. Action labels are base-frame meter-valued increments relative to
measured TCP position at execution; the existing IK term holds initial orientation.
The serialization margin keeps labels strictly inside the 5 mm norm bound.

The isolated learning process owns episode buffering, storage validation,
collection manifests, train/validation splitting and LeRobot export. No LeRobot,
NumPy dataset writer or learning import is introduced into the simulator backend.
See robot-learning's `docs/v3_demonstrations.md` for two-terminal collection/export
commands and dataset semantics. Use enough `--max-requests` for the finite job
(each expert policy step uses label, submit and advance RPCs).

The simulator receipt includes actual target-volume/controller settings and
source hashes as well as the existing asset/contract/runtime provenance. A V3
verification run on 2026-10-14 collected two successful episodes totaling 255
policy steps, then verified their complete LeRobot round trip outside Isaac.
No hardware, production dataset or model training was performed.
