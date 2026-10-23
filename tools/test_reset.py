"""Check that state() reflects a reset immediately (no stale data)."""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
parser = argparse.ArgumentParser(); parser.add_argument("--device", type=str, default="cuda:0"); args = parser.parse_args()
import torch
from supermimic.env import TrackingSim
from supermimic.kinematics import G1Kinematics
from supermimic.mdp import MDP, unflatten_state
from supermimic.motion import MotionLib
from supermimic.robot import default_joint_pos_list
dev = args.device
kin = G1Kinematics(dev); motion = MotionLib.from_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "lafan1_g1"), kin, dev)
mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list(), device=dev))
env = TrackingSim(dev, 64, motion, mdp, eval_mode=True)
ids = torch.arange(64, device=dev)
for _ in range(30):
    env.step(torch.zeros(64, 29, device=dev))
fr = motion.clip_start[ids % len(motion.names)] + 200
env.reset(ids, fr)
S = env.state(); ref = motion.ref_state(fr)
print("no update:", [round((a - b).abs().max().item(), 4) for a, b in zip(S, ref)])
env.refresh()
S = env.state()
print("after refresh:", [round((a - b).abs().max().item(), 4) for a, b in zip(S, ref)])
out = env.step(torch.zeros(64, 29, device=dev))
print("step s vs ref:", [round((a - b).abs().max().item(), 4) for a, b in zip(unflatten_state(out["s"]), ref)])
