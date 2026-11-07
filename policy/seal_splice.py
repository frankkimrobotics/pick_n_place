#!/usr/bin/env python3
"""seal_splice :: hand the contact phase (press -> seal -> lift) to the RL seal controller.

Why (from the 2026-10-24 real runs):
  - bspline grazed an edge at tau 0.08, latched suction on marginal contact, lost it.
  - ycbgrasp (and goaldiv, by offline replay) cannot recover from off-goal hover: both
    drift toward distractors instead of re-centering on the goal.
The splice replaces exactly that regime with deterministic alignment + a policy trained
for it: CFM approach -> scripted align over the graspmap goal -> seal RL actor
(press to the force window via the torque proxy, suction, dwell, lift, retry).

The actor is rl/seal_sac.Actor trained at 10 Hz decisions (seal_sac13, DECIM=25), the
same rate as the real executor. Observation mapping (sim channel -> real source):
  q, qd          URDF rad / rad s^-1      stream joints (lcnc deg -> URDF rad), finite diff
  terr           qref - q (rad)           last commanded reference - measured
  tau proxy[2]   F*lever*gain+bias+noise  J2/J3 torque rise over the contact guard's
                                          frozen baseline, times --tau_scale (sim proxy
                                          ~0.7 at firm contact vs real rise ~0.105 -> ~6.7;
                                          training DR tolerates 4x error)
  tipz           base frame, table=0      FK tip z minus --table_z
  suction        commanded state          state["suction"]
  t/T_MAX        splice-local timer       T_MAX = 3.5 s
8-frame ObsStack at the decision rate, ActGate hysteresis on suction/lift — identical
to training.
"""
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "rl"))

T_MAX = 3.5
LIFT_MM = 25.0


class SealSplice:
    def __init__(self, ckpt, tau_scale=6.7, table_z=0.030, dry=False, dt=0.02):
        from seal_sac import Actor, ActGate, ObsStack, DEP1, K_STACK  # noqa: E402
        self.actor = Actor()
        ck = torch.load(os.path.expanduser(ckpt), map_location="cpu", weights_only=False)
        self.actor.load_state_dict(ck["actor"] if "actor" in ck else ck)
        self.actor.eval()
        self.gate = ActGate()
        self.stack = ObsStack()
        self.dep1 = DEP1
        self.tau_scale = float(tau_scale)
        self.table_z = float(table_z)
        self.dry = dry
        self.dt = float(dt)          # decision period (0.02 s = 50 Hz, the actor's native rate)
        print(f"[splice] seal actor {os.path.basename(str(ckpt))} tau_scale {tau_scale} "
              f"table_z {table_z}{' DRY' if dry else ''}", flush=True)

    # ------------------------------------------------------------------ obs
    def _obs(self, q, qd, terr, tau_rise23, tipz, suction, t):
        dep = np.concatenate([
            np.asarray(q, np.float32),
            np.asarray(qd, np.float32),
            np.asarray(terr, np.float32),
            np.asarray(tau_rise23, np.float32) * self.tau_scale,   # signed, like the sim proxy;
                                                                   # the REAL io.read() abs()es the rise
                                                                   # (press-positive) before scaling
            [tipz - self.table_z, float(suction), t / T_MAX],
        ]).astype(np.float32)
        assert dep.shape == (self.dep1,), dep.shape
        return dep

    @torch.no_grad()
    def act(self, dep_stacked):
        a, _ = self.actor(torch.from_numpy(dep_stacked[None]).float(), deterministic=True)
        return self.gate(a[0].numpy())       # [vz mm/s, suction 0/1, lift 0/1(latched)]

    # ------------------------------------------------------------------ run
    def run(self, io, clock=time):
        """Blocking contact-phase loop at 10 Hz. `io` provides the robot interface:
          io.read()  -> dict(q, qd, terr, tau_rise23, tip)   (URDF rad / rad/s / rad / raw / m)
          io.move_dz(dz_m)            one 0.1 s vertical reference segment (guarded)
          io.suction(on)              command the pump
          io.lift(dz_m)               scripted lift leg (guarded)
        Returns "lifted" | "timeout" | "aborted". `clock` is injectable so the sim
        validation can step time per decision instead of waiting wall-clock.
        """
        self.gate.reset()
        s = io.read()
        suction = 0
        dep = self.stack.reset(self._obs(s["q"], s["qd"], s["terr"], s["tau_rise23"],
                                         s["tip"][2], suction, 0.0))
        t0 = clock.time()
        lift_started = None
        while True:
            t = clock.time() - t0
            if t > T_MAX:
                if suction:
                    io.suction(False)
                return "timeout"
            ae = self.act(dep)
            vz_mm, suc_cmd, lift_cmd = float(ae[0]), int(ae[1]), int(ae[2])
            if self.dry:
                print(f"[splice-dry] t{t:4.1f} vz {vz_mm:6.1f} suc {suc_cmd} lift {lift_cmd} "
                      f"tipz {s['tip'][2]:.3f} tau {np.max(np.abs(s['tau_rise23'])):.3f}", flush=True)
            else:
                if suc_cmd != suction:
                    io.suction(bool(suc_cmd))
                    suction = suc_cmd
                if lift_cmd and lift_started is None:
                    lift_started = t
                    ok = io.lift(LIFT_MM / 1000.0)
                    return "lifted" if ok else "aborted"
                io.move_dz(np.clip(vz_mm, -30, 30) * 1e-3 * self.dt)   # vz over one decision
            s = io.read()
            dep = self.stack.push(self._obs(s["q"], s["qd"], s["terr"], s["tau_rise23"],
                                            s["tip"][2], suction, t))
            # pace to the decision rate
            sl = self.dt - ((clock.time() - t0) - t)
            if sl > 0:
                clock.sleep(sl)
