"""batchsim.eval :: run any policy over N latency-aware worlds and report task + timing metrics.

A policy is a callable  act(obs) -> dict(pts [N, p, 6], knot_dt, t_anchor (optional), valve [N] (optional))
working on the host-side observation of BatchTwin.observe(). The evaluator measures the policy's own wall-clock
compute per decision and (with measure_compute=True) feeds THAT into the twin as the inference latency, so a slow
policy is penalised exactly as it would be on the arm; otherwise the configured infer_* distribution is used.

    python -m batchsim.eval --nworld 64 --cameras zed wrist_d405        (hold-still policy: a plumbing check)
"""
import argparse, json, os, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from batchsim.twin import BatchTwin  # noqa: E402
from batchsim.world import NJ  # noqa: E402


def evaluate(tw, policy, decisions=100, decision_dt=0.1, images=True, measure_compute=False, on_decision=None):
    n_tick = int(round(decision_dt / tw.cfg["cmd_dt"]))
    N, dev = tw.N, tw.dev
    st = dict(lat=[], img_age={c: [] for c in tw.cams}, q_age=[], compute=[], v_peak=torch.zeros(N, device=dev))
    held_any = torch.zeros(N, dtype=torch.bool, device=dev)
    t0 = time.time()
    for k in range(decisions):
        obs = tw.observe(images=images)
        tc = time.time()
        act = policy(obs)
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        comp = time.time() - tc
        lat = tw.command(act["pts"], act.get("knot_dt", decision_dt), t_anchor=act.get("t_anchor"), valve=act.get("valve"),
                         compute=(comp if measure_compute else None))
        tw.advance(n_tick)
        st["lat"].append(lat); st["q_age"].append(obs["q_age"]); st["compute"].append(comp)
        for c in tw.cams:
            if images:
                st["img_age"][c].append(obs["img_age"][c])
        st["v_peak"] = torch.maximum(st["v_peak"], tw.world.qvel[:, :NJ].abs().max(1).values)
        held_any |= tw.suction.held >= 0
        if on_decision is not None:
            on_decision(k, tw, obs, act)
    wall = time.time() - t0
    ms = lambda xs: round(1000 * float(torch.stack(xs).float().mean()), 1) if xs else None
    d = tw.drive.stats
    return dict(worlds=N, decisions=decisions, sim_s=decisions * decision_dt, wall_s=round(wall, 1), realtime_factor=round(N * decisions * decision_dt / wall, 1),
                policy_ms_per_decision=round(1000 * sum(st["compute"]) / decisions, 2), command_latency_ms=ms(st["lat"]), joint_fb_age_ms=ms(st["q_age"]),
                image_age_ms={c: ms(v) for c, v in st["img_age"].items()}, peak_joint_speed_deg_s=round(float(torch.rad2deg(st["v_peak"]).mean()), 1),
                held_any_frac=float(held_any.float().mean()), vsat_frac=float((d["vsat"] / max(1, d["ticks"])).mean()),
                reference_ran_out_frac=float((d["late"] / max(1, d["ticks"])).mean()), ferror_ticks=float(d["ferror"].mean()),
                healthy_frac=float(tw.healthy().float().mean()))


class HoldPolicy:
    """Keeps the arm where the joint feedback says it is (plumbing check and timing baseline)."""

    def __call__(self, obs):
        return dict(pts=obs["q"][:, None, :].expand(-1, 2, -1), knot_dt=1.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default="~/pnp_dataset_mega/scene_980/scene.xml")
    ap.add_argument("--nworld", type=int, default=64); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--level", default="real")
    ap.add_argument("--dr", type=int, default=1); ap.add_argument("--decisions", type=int, default=50); ap.add_argument("--cameras", nargs="*", default=[])
    ap.add_argument("--joint_model", type=int, default=0)
    a = ap.parse_args()
    tw = BatchTwin(a.xml, a.nworld, a.device, level=a.level, dr=bool(a.dr), cameras=tuple(a.cameras), joint_model=a.joint_model)
    print(json.dumps(evaluate(tw, HoldPolicy(), a.decisions, images=bool(a.cameras)), indent=1))
