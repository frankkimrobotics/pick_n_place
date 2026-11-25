"""pnp :: command line. `pnp tasks | run <task> [--set k=v] [--] [args] | config show [--diff] [--json] |
config validate | env check | pi sync [--run]`. Runs under any Python 3.11+ (bin/pnp picks ilpolicy)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pnp.config import REPO, load, _toml
from pnp.config.schema import validate

TASKS = Path(__file__).with_name("tasks.toml")


def _tasks():
    return _toml.loads(TASKS.read_text())


def _fmt(cfg, s):
    """expand {paths.x} / {pi.host} placeholders."""
    out, flat = s, cfg.flat()
    for k, v in flat.items():
        out = out.replace("{" + k + "}", str(v))
    return os.path.expanduser(out)


def cmd_tasks(a):
    cfg = load()
    t = _tasks()
    w = max(len(k) for k in t)
    for name, spec in t.items():
        tag = " [robot]" if spec.get("needs_robot") else " [service]" if spec.get("service") else ""
        print(f"{name:{w}}  {spec['env']:9} {spec['description']}{tag}")
        if a.verbose:
            print(f"{'':{w}}  script {_fmt(cfg, spec['script'])}")
            for key in ("inputs", "outputs", "notes"):
                if key in spec:
                    v = spec[key]
                    print(f"{'':{w}}  {key:7} {', '.join(_fmt(cfg, x) for x in v) if isinstance(v, list) else _fmt(cfg, v)}")


def cmd_run(a):
    a.set = (a.set or []) + (getattr(a, "run_set", None) or [])
    if a.args and a.args[0] == "--":
        a.args = a.args[1:]
    if any(x in ("--dry-run", "--set") for x in a.args):
        sys.exit("pnp run: put --dry-run / --set BEFORE the task name; everything after it goes to the script")
    cfg = load(experiment=a.experiment, overrides=a.set)
    t = _tasks()
    if a.task not in t:
        sys.exit(f"unknown task '{a.task}'; `pnp tasks` lists them")
    spec = t[a.task]
    if spec.get("needs_robot"):
        bad = validate(cfg)
        if bad:
            sys.exit("config invalid, refusing to move the robot:\n  " + "\n  ".join(bad))
    py = _fmt(cfg, cfg.interpreters[spec["env"]])
    script = _fmt(cfg, spec["script"])
    if not script.startswith("/"):
        script = str(REPO / script)
    env = dict(os.environ)
    pp = [str(REPO)] + [_fmt(cfg, p) for p in spec.get("pythonpath", [])]
    env["PYTHONPATH"] = ":".join(pp + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    if a.experiment:
        env["PNP_EXPERIMENT"] = a.experiment
    for item in a.set or []:                         # children that use pnp.config see the same overrides
        k, _, v = item.partition("=")
        env["PNP_" + k.strip().upper().replace(".", "__")] = v.strip()
    cmd = [py, script] + [_fmt(cfg, x) for x in spec.get("args", [])] + a.args
    print("+ " + " ".join(cmd), file=sys.stderr)
    if a.dry_run:
        return
    os.chdir(_fmt(cfg, spec.get("cwd", str(REPO))))
    sys.exit(subprocess.call(cmd, env=env))


def cmd_config_show(a):
    cfg = load(experiment=a.experiment, overrides=a.set)
    if a.json:
        print(json.dumps(cfg.to_dict(), indent=1)); return
    print("layers: " + " -> ".join(cfg.layers))
    rows = cfg.diff() if a.diff else {k: (None, v, cfg.origin(k)) for k, v in cfg.flat().items()}
    if a.diff and not rows:
        print("(effective config equals defaults.toml plus the machine layer; nothing overridden)"); return
    w = max((len(k) for k in rows), default=10)
    for k, (d, v, o) in rows.items():
        print(f"{k:{w}}  {v!r:<40} {o}" + (f"   (default {d!r})" if a.diff else ""))


def cmd_config_validate(a):
    cfg = load(experiment=a.experiment, overrides=a.set)
    bad = validate(cfg)
    if bad:
        print("INVALID:"); [print("  " + b) for b in bad]; sys.exit(1)
    print(f"valid: {len(cfg.flat())} values across {len(cfg.layers)} layers")


PROBES = {"ilpolicy": "torch, numpy", "mjwarp": "mujoco, numpy, matplotlib", "mj4": "mujoco, numpy",
          "curobo2": "curobo, torch", "curobo3": "curobo, torch", "sam3": "torch", "system": "mcap, numpy"}


def cmd_env_check(a):
    cfg = load()
    ok = True
    for name in cfg.interpreters.keys():
        py = os.path.expanduser(cfg.interpreters[name])
        if name == "nodejs":
            r = subprocess.run([py, "--version"], capture_output=True, text=True)
            print(f"{name:9} {'ok ' if r.returncode == 0 else 'FAIL'} {r.stdout.strip()}"); ok &= r.returncode == 0; continue
        if not os.path.exists(py):
            print(f"{name:9} FAIL missing {py}"); ok = False; continue
        mods = PROBES.get(name, "numpy")
        r = subprocess.run([py, "-c", f"import sys; import {mods}; print(sys.version.split()[0])"], capture_output=True, text=True)
        print(f"{name:9} {'ok ' if r.returncode == 0 else 'FAIL'} py {r.stdout.strip() or '-'}  imports: {mods}" + ("" if r.returncode == 0 else "  <- " + r.stderr.strip().splitlines()[-1]))
        ok &= r.returncode == 0
    sys.exit(0 if ok else 1)


def cmd_pi_sync(a):
    cfg = load()
    src = os.path.join(cfg.paths.mycobot_mpc, "robot_hal.py")
    dst = f"{cfg.pi.user}@{cfg.pi.host}:{cfg.pi.robot_hal_path}"
    rev = subprocess.run(["git", "-C", cfg.paths.mycobot_mpc, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cmd = ["rsync", "-av", "--backup", "--suffix", f".bak_{rev}", src, dst]
    print("+ " + " ".join(cmd))
    if a.run:
        sys.exit(subprocess.call(cmd))
    print("(dry run; add --run to copy. Phase 5 adds a version stamp the Pi reports in its :9998 status)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pnp", description=__doc__)
    ap.add_argument("--experiment", "-e", help="configs/experiments/<name>.toml")
    ap.add_argument("--set", action="append", metavar="section.key=value")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tasks"); t.add_argument("-v", "--verbose", action="store_true"); t.set_defaults(fn=cmd_tasks)
    r = sub.add_parser("run", usage="pnp run [--dry-run] [--set k=v]... TASK [-- SCRIPT ARGS]  (options go BEFORE the task name)")
    r.add_argument("--dry-run", action="store_true", help="print the command, do not execute")
    r.add_argument("--set", action="append", dest="run_set", metavar="section.key=value")
    r.add_argument("task"); r.add_argument("args", nargs=argparse.REMAINDER); r.set_defaults(fn=cmd_run)
    c = sub.add_parser("config"); cs = c.add_subparsers(dest="sub", required=True)
    s = cs.add_parser("show"); s.add_argument("--diff", action="store_true"); s.add_argument("--json", action="store_true"); s.set_defaults(fn=cmd_config_show)
    cs.add_parser("validate").set_defaults(fn=cmd_config_validate)
    e = sub.add_parser("env"); es = e.add_subparsers(dest="sub", required=True); es.add_parser("check").set_defaults(fn=cmd_env_check)
    p = sub.add_parser("pi"); ps = p.add_subparsers(dest="sub", required=True); q = ps.add_parser("sync"); q.add_argument("--run", action="store_true"); q.set_defaults(fn=cmd_pi_sync)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
