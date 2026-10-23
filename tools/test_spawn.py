"""Spawn G1s on mjlab (MuJoCo-Warp), report ordering, drive parameters and stepping throughput."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

import torch
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import Simulation
from mjlab.terrains import TerrainEntityCfg

from supermimic.env import DECIMATION, PHYSICS_DT, make_sim_cfg
from supermimic.robot import g1_cfg

scene = Scene(SceneCfg(num_envs=args.num_envs, env_spacing=0.0, terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": g1_cfg()}), args.device)
sim = Simulation(num_envs=args.num_envs, cfg=make_sim_cfg(), spec=scene.spec, device=args.device)
scene.initialize(mj_model=sim.mj_model, model=sim.model, data=sim.data)
robot = scene["robot"]
m = sim.mj_model
print("JOINTS", robot.joint_names)
print("BODIES", robot.body_names)
print("default q", robot.data.default_joint_pos[0])
print("stiffness", torch.tensor(m.actuator_gainprm[robot.indexing.ctrl_ids.cpu().numpy(), 0]))
print("armature", torch.tensor(m.dof_armature[robot.indexing.joint_v_adr.cpu().numpy()]))
print("mass", m.body_mass[robot.indexing.body_ids.cpu().numpy()].sum())
robot.write_root_state_to_sim(robot.data.default_root_state)
robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
target = robot.data.default_joint_pos.clone()
for it in range(3):
    torch.cuda.synchronize()
    t0 = time.time()
    n = 100
    for _ in range(n):
        robot.set_joint_position_target(target + 0.1 * torch.randn_like(target))
        for _ in range(DECIMATION):
            scene.write_data_to_sim()
            sim.step()
            scene.update(PHYSICS_DT)
        _ = robot.data.joint_pos
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"steps/s (control 50Hz) = {n * args.num_envs / dt:.0f}; root z mean {sim.data.qpos[:, 2].mean():.3f}")
