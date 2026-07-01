#!/usr/bin/env python3
"""contact_detector :: fused suction-cup contact detection for the streaming controller.

Combines three complementary signals (each used for its STRENGTH, not equal-weight voting):

  1. depth gap  (rim vs 5px outer ring, GapMonitor.latest_gap) -> PROXIMITY / ARM gate.
     Reaches ~0 as a flat surface meets the rim plane; fires ~1-2 cm before contact.
  2. blue-dot   (cup dome image displacement, GapMonitor.latest_bluey, +up = compress)
                -> EARLIEST true-contact cue (the cup starts compressing at first touch).
  3. J2/J3 torque (drive torqfb on :9999) -> FIRM-contact CONFIRM + independent backstop.
     Rises ~4x above pose-drift only after several mm of compression, but is robust and
     works even when vision is occluded/lost.

Onset order in a descent is gap -> blue-dot -> torque, so:
  gap (or a kinematic z-gate) ARMS  ->  blue-dot TRIGGERS early  ->  torque CONFIRMS / backstops.

The gate matters: blue-dot is sensitive but vision-noisy, so it may only trigger CONTACT once
ARMED (near a surface). Torque-firm and torque-hard are standalone (they almost never
false-positive: pose-drift maxes ~0.05, firm is 0.08, hard 0.13), so they fire even unarmed --
that is the safety backstop for occluded / vision-blind contacts.

Thresholds are the 2026-07-01 measured values (normalized torque units, metres, pixels).
Feed it whatever signals you have each servo tick; missing ones (None) degrade gracefully.
On CONTACT the caller sends {"hold"} to online_servo.
"""


class ContactDetector:
    def __init__(self,
                 tau_firm=0.08, tau_hard=0.13, w_j3=0.5,   # torque: J2 |dev| + 0.5*J3 |dev|
                 v_on=2.5, v_strong=6.0,                    # blue-dot up-displacement (px)
                 g_arm=0.012, g_contact=0.002,              # depth gap (m): arm / near-contact
                 z_arm=0.015,                               # kinematic arm: |cup_z - surface_z| (m)
                 debounce=3):                               # consecutive ticks to confirm (~0.1-0.15 s)
        self.tau_firm = tau_firm; self.tau_hard = tau_hard; self.w_j3 = w_j3
        self.v_on = v_on; self.v_strong = v_strong
        self.g_arm = g_arm; self.g_contact = g_contact; self.z_arm = z_arm
        self.debounce = debounce
        self.reset()

    def reset(self):
        self.base_tau = None
        self.state = "APPROACH"; self.armed = False; self.contact = False; self.why = None
        self._cand = 0
        self.last = {}

    def set_baseline(self, torque):
        """Capture the torque baseline (median over ~1 s) at the pre-contact pose, per approach.
        blue-dot and gap are already relative (GapMonitor tracks their own baselines)."""
        self.base_tau = list(torque) if torque is not None else None

    def update(self, torque=None, blue_dy=None, gap=None, cup_z=None, surface_z=None):
        """One tick. Pass whatever is available (None = that sensor is unavailable this tick).
          torque   : 6-vector from :9999   (uses J2=idx1, J3=idx2)
          blue_dy  : cup-dome up-displacement px (GapMonitor bluedy, +up = compression)
          gap      : median(ring)-median(rim) in metres (GapMonitor latest_gap)
          cup_z    : FK tcp/tip z (m) -- kinematic proximity (needs recalibrated tcp)
          surface_z: expected contact z (m) from detection -- kinematic proximity
        Returns a dict with .contact, .state, .why, .conf and the raw features.
        """
        # ---- per-signal features (None-safe) ----
        tau = None
        if torque is not None and self.base_tau is not None:
            tau = abs(torque[1] - self.base_tau[1]) + self.w_j3 * abs(torque[2] - self.base_tau[2])
        firm = tau is not None and tau > self.tau_firm
        hard = tau is not None and tau > self.tau_hard
        v_on = blue_dy is not None and blue_dy > self.v_on
        v_strong = blue_dy is not None and blue_dy > self.v_strong
        g_near = gap is not None and gap < self.g_arm
        g_contact = gap is not None and gap < self.g_contact
        z_near = (cup_z is not None and surface_z is not None
                  and abs(cup_z - surface_z) < self.z_arm)

        # ---- ARM (latching): near a surface by vision OR kinematics, or a real cue already present ----
        if g_near or z_near or firm or v_strong:
            self.armed = True

        # ---- CONFIRM rules (debounced) ----
        #   torque-hard  : immediate, standalone  (hard press -- must stop, even vision-blind)
        #   torque-firm  : standalone              (robust firm contact, ~no false positives)
        #   armed + blue : the EARLY/gentle path   (proximity-gated cup-compression onset)
        #   armed + gap~0: flat-surface contact    (ring meets the rim plane)
        confirm = hard or firm or (self.armed and v_on) or (self.armed and g_contact)
        self._cand = self._cand + 1 if confirm else 0
        if not self.contact and self._cand >= self.debounce:
            self.contact = True
            self.why = ("torque-hard" if hard else "torque-firm" if firm
                        else "vision(blue-dot)+armed" if v_on else "depth-gap~0+armed")

        # confidence score (logging / tuning only)
        conf = round(0.4 * self.armed + 0.5 * v_on + 0.3 * v_strong + 0.6 * firm, 2)
        self.state = "CONTACT" if self.contact else ("ARMED" if self.armed else "APPROACH")
        self.last = {"contact": self.contact, "state": self.state, "why": self.why, "conf": conf,
                     "tau": None if tau is None else round(tau, 4),
                     "blue_dy": blue_dy, "gap": gap, "armed": self.armed}
        return self.last


# ---- integration sketch (inside a slow descent loop, ~20-50 Hz) ----
#   det = ContactDetector()
#   det.set_baseline(read_torque())               # ~1 s at pregrasp, before descending
#   while descending:
#       tq = mon.torque                           # from :9999
#       dy = mon.latest_bluey()[1]                # blue-dot up-displacement (px)
#       gp = mon.latest_gap()                     # rim-vs-ring gap (m)
#       cz = fk_z(read_cur())                     # cup tip z (recalibrated tcp)
#       r  = det.update(tq, dy, gp, cup_z=cz, surface_z=obj_top_z)
#       if r["contact"]:
#           sock.sendall(b'{"hold": true}\n')     # freeze the welded reference
#           print("CONTACT via", r["why"], "conf", r["conf"]); break
#       sleep(0.03)
