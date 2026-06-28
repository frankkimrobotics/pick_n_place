#!/usr/bin/env python3
"""suction_test :: switch the REAL robot's suction cup ON/OFF — NO arm motion.

This ONLY sets the LinuxCNC HAL digital-output pin that drives the suction
pump/valve. It is the exact pin mycobot_mpc/robot_hal.py uses:

    SUCTION_PIN = "pro600.digital_out00"     # 1 = suction ON, 0 = OFF

It never commands a joint/Cartesian move — it just does `halcmd setp <pin> <0|1>`
(and `halcmd getp` to read it back). Safe to run while the arm holds its pose.

Run ON the robot's LinuxCNC Raspberry Pi (halcmd is local there):
    python3 suction_test.py --cycle 3        # ON/OFF 3 times (default)
    python3 suction_test.py --on             # latch suction ON
    python3 suction_test.py --off            # latch suction OFF

Or drive it over SSH from the desktop (needs the Raspi IP):
    export ROBOT_IP=10.0.0.27
    python3 suction_test.py --cycle 3 --host $ROBOT_IP --user pi

Equivalent raw one-liners on the Pi:
    halcmd setp pro600.digital_out00 1   # ON
    halcmd setp pro600.digital_out00 0   # OFF
    halcmd getp pro600.digital_out00     # read state
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

SUCTION_PIN = "pro600.digital_out00"     # same pin as mycobot_mpc/robot_hal.py


def _run(cmd, host, user):
    """Run a shell command locally (on the Pi) or over SSH; return CompletedProcess."""
    if host:
        full = ["ssh", f"{user}@{host}", "bash", "-lc", shlex.quote(cmd)]
    else:
        full = ["bash", "-lc", cmd]
    return subprocess.run(full, capture_output=True, text=True)


def set_pin(value, host, user):
    r = _run(f"halcmd setp {SUCTION_PIN} {int(value)}", host, user)
    if r.returncode != 0:
        raise RuntimeError(f"halcmd setp failed: {(r.stderr or r.stdout).strip()}")


def get_pin(host, user):
    r = _run(f"halcmd getp {SUCTION_PIN}", host, user)
    if r.returncode != 0:
        raise RuntimeError(f"halcmd getp failed: {(r.stderr or r.stdout).strip()}")
    return r.stdout.strip()


def main():
    ap = argparse.ArgumentParser(
        description="Toggle the suction digital-out pin (NO arm motion).")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--on", action="store_true", help="suction ON and exit")
    g.add_argument("--off", action="store_true", help="suction OFF and exit")
    g.add_argument("--cycle", type=int, metavar="N", help="N ON/OFF cycles (default 3)")
    ap.add_argument("--on-sec", type=float, default=2.0, help="seconds held ON per cycle")
    ap.add_argument("--off-sec", type=float, default=2.0, help="seconds held OFF per cycle")
    ap.add_argument("--host", default=os.environ.get("ROBOT_IP", "").strip(),
                    help="Raspi IP for SSH (default: ROBOT_IP env; empty = run locally on the Pi)")
    ap.add_argument("--user", default="pi", help="SSH user on the Raspi (default: pi)")
    args = ap.parse_args()

    where = f"ssh {args.user}@{args.host}" if args.host else "local halcmd (on the Pi)"
    print(f"[suction] pin={SUCTION_PIN}  via {where}  —  NO arm motion")

    # connectivity / pin-exists probe
    try:
        state0 = get_pin(args.host, args.user)
        print(f"[suction] current pin state: {state0}")
    except Exception as e:
        print(f"[suction] ERROR: cannot reach halcmd ({e}).", file=sys.stderr)
        print("  Run this ON the robot's LinuxCNC Pi, or pass --host <ROBOT_IP>.",
              file=sys.stderr)
        sys.exit(1)

    try:
        if args.on:
            set_pin(1, args.host, args.user)
            print(f"[suction] ON   -> pin={get_pin(args.host, args.user)}")
        elif args.off:
            set_pin(0, args.host, args.user)
            print(f"[suction] OFF  -> pin={get_pin(args.host, args.user)}")
        else:
            n = args.cycle if args.cycle else 3
            for i in range(n):
                set_pin(1, args.host, args.user)
                print(f"  cycle {i+1}/{n}: ON   pin={get_pin(args.host, args.user)}")
                time.sleep(args.on_sec)
                set_pin(0, args.host, args.user)
                print(f"  cycle {i+1}/{n}: OFF  pin={get_pin(args.host, args.user)}")
                time.sleep(args.off_sec)
            print("[suction] cycle test done (ended OFF).")
    except KeyboardInterrupt:
        set_pin(0, args.host, args.user)
        print("\n[suction] interrupted -> forced OFF")
        sys.exit(130)


if __name__ == "__main__":
    main()
