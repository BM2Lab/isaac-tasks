# fr3 assets

Model provenance, joint order, TCP and conversion metadata are pending. No assets downloaded.
## V1 audited bare-flange asset

`model.json` records provenance, joint order, home, command bounds and hashes.
The geometry is external, not bundled or downloaded. Set `ISAAC_TASKS_FR3_USD`
to the exact audited `fr3v21_bare_flange_live_v1/fr3v2.usda` asset. Missing files
or a different SHA-256 fail before Kit launch. Copy the **entire asset directory**
when moving hosts: relative mesh references must remain intact. The USD hash
checks the root layer, not the contents of every referenced mesh.

The source is the existing FR3 hardware-model audit, not the Panda Nucleus asset.
The historical project remains unmodified. Logical `robot_base` maps to
`fr3_link0`; logical `bare_flange` maps to the **link origin** of `fr3_flange`
(not its center of mass). There is no TCP offset or gripper. Quaternions are XYZW
in the pinned Isaac Lab checkout.

V1 uses a new simple implicit-PD IK baseline: K=400, D=80 on all seven joints,
effort bounds 87 Nm for joints 1–4 and 12 Nm for joints 5–7, armature 0.003 kg m².
Gravity is enabled and a model gravity feedforward effort is supplied. Armature
is a numerical stabilizer, not identified motor inertia. This is **not** a
claim of parity with the Franky torque controller or hardware safety limits.
