# Streaming touch + contact-detection methods

Summary of the streaming pick-and-place / touch work and the contact-detection
investigation on the MyCobot Pro 630 (cuRobo planner + 4 ms Pi-side servo).

## Architecture (online streaming)

```
cuRobo planner (:9997) ── plan full traj, slice 0.4s chunk every 0.1s (sliding window)
   online_planner_node.py --mode chunk --horizon 0.4 --plan-hz 10
        ── /planner/weld_chunks ──> chunk_to_pi.py ──TCP──> Pi :9994
   online_servo.py (Pi, loaded by elerob_online.hal as `ctrl`):
        welds chunks into q_ref(t) on a chunk thread; SERVOS @ 250 Hz (period_ms=4)
        target = q_ref(now + lead)   ← feed-forward lead cancels the constant dead-time
   feedback streamed on :9999 (joints + torque)
```

- **4 ms control loop is real** — `controller_params period_ms=4`; the redesigned
  `online_servo.py` runs the servo at 250 Hz with pure-PD (no integral windup).
- **`elerob_online.hal`** moves the PID + `pro_socketcan` off the 20 ms slow-thread onto a
  fast thread (start 10 ms, then 4 ms) + `motor_time_interval` to match — otherwise the
  4 ms loop is downsampled to 50 Hz at the drive. Step-response: dead-time onset 310 ms
  (20 ms) → 233 ms (10 ms); the slow rise (~1 s to 50%) is the drive itself (not thread-fixable).
- **Dead-time is a pipeline**, not per-command accumulation: ~0.8 s once to fill, then the
  chunk cadence. A constant lead compensates it for *known* trajectories; only *reactive*
  events (contact) pay the full delay.

## Contact detection — what works and what doesn't

| method | flag | result |
|---|---|---|
| **Open-loop touch** | `--open-touch` | **WORKS, reliable.** Descend to the detected top (`surf−margin`), hold; the compliant cup presses. No sensing. Best for wide circular tops. |
| **Torque rise** | `--torque-stop` | Implemented. Watches summed per-joint `pro600.joint{i}_torqfb` rise (in the `:9999` stream) past `--torque-thresh`. Needs the torque-stream `robot_hal.py` reloaded. For comparison. |
| **Gap (depth)** | `--gap-stop` | **Proximity only.** `gap = median(ring depth) − median(rim depth)`; fires a *constant* ~+41 mm above the surface, never at contact. A `--gap-descend` calibrated offset is fragile. |
| **Blue-dot / dome (vision)** | (legacy) | Unreliable. Template match dies near contact; dome-shift is proximity-contaminated. |

### Key finding: depth/vision contact is impossible here
The suction cup sits **inside the D405's near-field blind spot**, so the camera cannot
measure the cup's *own* depth — the "cup" ROI sees the object *past* the cup, identical to
the surrounding ring (`depth_compare.py`: dome-depth and 5px-ring-depth track within ~0 the
entire descent). With no cup reference, **no depth comparison can mark contact**, and the
cup compression that vision would key on isn't measurable in depth. So: use **open-loop**
(controller is accurate to ~1 mm) or **torque**, not the camera.

## Tools

- `servo_touch.py` — touch/pick-place; `--stream` (4 ms path), `--open-touch`, `--torque-stop`,
  `--gap-stop`/`--ring-px`/`--gap-descend`, ROI `--xmin/--xmax/--ymin/--ymax`, `--max-objects`.
- `online_planner_node.py` — cuRobo → 0.4 s sliding-window chunks.
- `touch_chunk.py` + `plot_touch_chunk.py` — touch+return via the chunk path, records an mcap
  rosbag + D405/D435 frames + phase/contact, and plots trajectory/phase/contact/RGB.
- `step_response.py` — end-to-end dead-time measurement.
- `move_to_point.py`, `return_via_servo.py`, `test_servo_stream.py` — streaming move helpers.
- `depth_compare.py`, `vision_debug.py`, `vision_touch.py` — contact-signal diagnostics (kept
  for reference; superseded by open-touch/torque).
