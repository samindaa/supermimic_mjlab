"""Unitree G1 (29 DoF) entity on mjlab, following BeyondMimic's actuator model.

The MJCF is mjlab's ``g1_29dof_rev_1_0``, the revision the LAFAN1 motions were retargeted to. Joint
drives are MuJoCo position actuators with a 10 Hz natural frequency and damping ratio 2 on the
reflected motor armature (mjlab's ``G1_ARTICULATION``, ported from BeyondMimic), and the action
scale is 0.25 * effort_limit / stiffness, as in BeyondMimic.
"""

from __future__ import annotations

import re

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants as g1
from mjlab.entity import EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.string import resolve_expr

MJCF_PATH = str(g1.G1_XML)

# Joint order of the LAFAN1 retarget CSVs (Unitree SDK order); also the MJCF / mjlab joint order.
G1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]  # fmt: skip

# BeyondMimic's tracked bodies; the anchor/root is the pelvis.
TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link", "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]  # fmt: skip

# BeyondMimic's knees-bent default pose (pelvis at 0.76 m).
INIT_STATE = g1.KNEES_BENT_KEYFRAME


def _actuator_of(joint_name: str):
    for act in g1.G1_ARTICULATION.actuators:
        if any(re.fullmatch(p, joint_name) for p in act.target_names_expr):
            return act
    raise KeyError(joint_name)


def action_scale(joint_name: str) -> float:
    act = _actuator_of(joint_name)
    return 0.25 * act.effort_limit / act.stiffness


def default_joint_pos_list() -> list[float]:
    return list(resolve_expr(INIT_STATE.joint_pos, tuple(G1_JOINT_NAMES), 0.0))


# Robot-ground contacts only (self-collisions disabled, for speed), with friction on every contact so
# that the per-environment friction of the sim context applies to all of them.
GROUND_ONLY_COLLISION = CollisionCfg(geom_names_expr=(".*_collision",), contype=0, conaffinity=1, condim=3, priority=0)


def g1_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(GROUND_ONLY_COLLISION,),
        spec_fn=g1.get_spec,
        articulation=g1.G1_ARTICULATION,
    )
