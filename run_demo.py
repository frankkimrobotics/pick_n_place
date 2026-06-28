"""run_demo :: CLI entry-point for the pick-and-place simulation.

    conda activate curobo2
    cd /home/lisc-frank/Desktop/2026
    python -m pick_and_place.run_demo --place-mode planned
    python -m pick_and_place.run_demo --place-mode segmented --box-pose 0.0 0.32 -0.10

Artifacts (eye-in-hand RGBD, grasp still, demo video, report.json) -> outputs/.
"""
import argparse
import json
import os
import sys

# allow both `python -m pick_and_place.run_demo` and `python run_demo.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config as C


def main():
    ap = argparse.ArgumentParser(description="MyCobot Pro 630 pick-and-place simulation")
    ap.add_argument("--place-mode", choices=["planned", "segmented"], default="planned",
                    help="planned = collision-free plan w/ object OBB as collision volume; "
                         "segmented = approach+grasp+lift+move+release point-to-point")
    ap.add_argument("--box-pose", type=float, nargs="+", default=None,
                    metavar="X Y Z",
                    help="place-box bottom-centre in base frame (metres). "
                         "Default %s" % (C.BOX_POSE_XYZ.tolist(),))
    ap.add_argument("--object-pose", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="pick-object base position on the table (metres).")
    ap.add_argument("--no-artifacts", action="store_true",
                    help="skip rendering the video/figures (faster)")
    args = ap.parse_args()

    box_xyz = None
    if args.box_pose is not None:
        box_xyz = np.array(args.box_pose[:3], float)

    import pipeline
    import evaluate as EV

    rec = pipeline.run_pipeline(place_mode=args.place_mode, box_xyz=box_xyz,
                                object_xyz=args.object_pose,
                                save_artifacts=not args.no_artifacts)
    ev = EV.evaluate(rec)
    report = EV.format_report(rec, ev)
    print("\n" + report)

    # persist report.json (strip non-serialisable sim/scene handles)
    os.makedirs(C.OUT_DIR, exist_ok=True)
    clean = {k: v for k, v in rec.items() if not k.startswith("_")}
    clean["evaluation"] = ev

    def _json_default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return float(o)

    with open(os.path.join(C.OUT_DIR, "report.json"), "w", encoding="utf-8") as f:
        json.dump(clean, f, indent=2, default=_json_default)
    with open(os.path.join(C.OUT_DIR, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nreport -> {os.path.join(C.OUT_DIR, 'report.json')}")
    return 0 if ev["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
