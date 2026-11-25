# Restructuring pick_and_place for maintainability and agent operation

Status: plan, 2026-11-25. Phase 0 (the foundation) is implemented alongside this document and
changes nothing that exists; later phases move files and need the gates described in section 6.

## 1. What the codebase is today, measured

| fact | value |
|---|---|
| Python files, lines | 253 files, 52,102 lines |
| scripts at the repository root | 116 |
| scripts with their own `argparse` | 156 |
| `sys.path.insert` or `append` hacks | 162 |
| files with a hard-coded path, serial number, host or port | 140 |
| configuration systems | two (`config.py` with 42 constants, `realsim/config.py` with levels and DR), plus 8 files holding 15 to 34 module-level constants each |
| dataset generator variants sharing `run_scene`, `build_scene_xml`, `sample_scene` | 6 |
| paired `_zed` / non-`zed` scripts in `policy/` | 3 pairs |
| RL environment versions in `rl/` | 4 |
| interpreters needed | 5 conda envs (`curobo2`, `mjwarp`, `sam3`, `ilpolicy`, `curobo3`) plus system Python |
| tests | 6 files, no runner config, no `pyproject.toml`, no `requirements.txt` |
| agent-facing documentation | none; `README.md` is 869 lines of history-ordered prose |
| root scripts untouched since before August | 72 of 116 |
| uncommitted work | 8 modified, 17 untracked, 9 `.bak` strays |
| copies of the Pi controller repository | two, divergent: `frankkimrobotics/mycobot_mpc` (June, 630-line `robot_hal.py`, no streaming) and `~/Desktop/2026/mycobot_mpc` (September, 938 lines, B-spline streaming, suction, torque). The second is the one the Pi runs. `pick_and_place` references `mycobot_mpc` paths 82 times. |

The costs these numbers produce: an agent cannot tell a live script from a dead one, cannot know
which interpreter a script needs without reading the README, cannot change the robot's IP, a camera
serial or a gain in one place, and cannot verify that a change is safe because there is no test
gate. Every one of those is a configuration or structure problem, not a code-quality problem; the
code itself is heavily commented and the comments are accurate.

## 2. Target shape

```
pick_and_place/
  pnp/                      importable package, no sys.path hacks anywhere
    config/                 loader, schema, provenance            (phase 0, done)
    robot/                  PiLink, HAL constants, calibration, geometry, FK
    sim/                    MJCF builder, twin, plunger, rim seal, env_warp
    data/                   ONE dataset generator; the six variants become configs
    policy/                 features, training, rollout, real executor
    rl/                     seal env + SAC, DAgger, planner sweep
    runlog/                 the messaging layer (docs/runlog_design.md)
    cli.py                  `pnp` entry point                     (phase 0, done)
    tasks.toml              task registry                         (phase 0, done)
  configs/
    defaults.toml           every tunable with its unit and source (phase 0, done)
    machine/<hostname>.toml paths, interpreters, serials, hosts   (phase 0, done)
    experiments/*.toml      one file per generator / training / eval recipe
  scripts/                  thin wrappers: parse args -> cfg -> call pnp.*
  archive/2026-06_07/       the 72 dead root scripts, unchanged, with an index of what each did
  tests/                    pytest, one smoke test per task in tasks.toml
  docs/                     designs, reports, diagrams
  AGENTS.md                 the map an agent reads first           (phase 0, done)
  pyproject.toml            `pip install -e .` into every env
```

Rules that make the shape hold:

1. **Nothing in `pnp/` reads a hard-coded path, serial, host, port or gain.** It reads `cfg`.
2. **Every runnable thing is in `tasks.toml`** with its interpreter, inputs and outputs. If it is not
   there, it is not supported.
3. **Variants are configs, not copies.** A new experiment is a new `configs/experiments/*.toml`.
4. **The Pi controller has one canonical repository**, `~/Desktop/2026/mycobot_mpc`, and `pnp pi sync`
   is the only way code reaches the Pi. The Pi reports the version it runs in its status reply.

## 3. The config system (phase 0, implemented)

**Format.** TOML, parsed by the standard library (`tomllib`, Python 3.11+; `tomli` on 3.10). No
YAML, no OmegaConf, no Hydra: nothing to install into five environments and nothing that resolves
values by magic an agent has to reason about.

**Layers**, each a plain file, applied in this order, later wins:

| layer | file | holds |
|---|---|---|
| defaults | `configs/defaults.toml` | every tunable, with a comment giving its unit and where the value came from |
| machine | `configs/machine/<hostname>.toml` | absolute paths, interpreter paths, camera serials, Pi host; the only file with a machine's name in it |
| experiment | `configs/experiments/<name>.toml` | the parameters one recipe changes, nothing else |
| environment | `PNP_<SECTION>__<KEY>=value` | one-off overrides without touching files |
| command line | `--set section.key=value` | the same, from any `pnp` command |

**Provenance.** The loaded object remembers where every value came from. `pnp config show` prints
the effective configuration with the file and line each value came from, and `pnp config show --diff`
prints only what differs from the defaults. An agent can answer "why is `vel_cmd_max` 850" without
grepping.

**Validation.** `pnp config validate` checks every value against `pnp/config/schema.py`: type, unit,
range, and cross-checks such as `len(home_q) == 6` and `base_q` inside `joint_limits`. It runs
automatically before any task that moves the robot.

**Access.** `cfg.robot.base_q`, `cfg.pi.host`, `cfg.cameras.wrist.serial`. Read-only after load.

**Compatibility.** `config.py` and `realsim/config.py` stay untouched in phase 0. In phase 2 their
constants are re-pointed at `pnp.config` so the 76 files that `import config` keep working while the
values move into TOML. The 140 files with their own hard-coded values are migrated one at a time,
each migration checked by the task's smoke test.

## 4. Why these choices

- **TOML over YAML.** Standard library, unambiguous types, no anchors or implicit conversions. An
  agent editing it cannot produce a valid-looking file that means something else.
- **Layered files over one big file.** The machine layer is the only thing that differs between this
  desktop, a MacBook and the Pi; experiments stay small because they state only their deltas.
- **Provenance over convention.** The single most common agent failure in this repository's history
  is editing a constant that was not the one in effect. Showing the origin of every value removes the
  guess.
- **A task registry over README prose.** The README's three quick-start commands each need a
  different interpreter and a `PYTHONPATH`. `pnp run <task>` encodes that once.
- **Archive, do not delete.** The 72 dead scripts hold calibration procedures and hardware notes
  that are still the only record of some experiments. They move, with an index, and stop appearing
  as candidates for edits.

## 5. Phases

| phase | work | effort | gate |
|---|---|---|---|
| 0 | `pnp/config`, `configs/`, `pnp/cli.py`, `tasks.toml`, `AGENTS.md`, this plan. Nothing moved. | done | `pnp config validate`, `pnp env check`, `pnp tasks` |
| 1 | Move the 72 pre-August root scripts to `archive/2026-06_07/` with `INDEX.md`; delete the 9 `.bak` strays after confirming each is in git history; commit the 25 in-flight files first | half a day | `git grep` shows no live import of an archived module; the three quick-start tasks still run |
| 2 | Re-point `config.py` and `realsim/config.py` at `pnp.config`; migrate the 140 hard-coded files in order of import count (`config` 76, `geometry` 48, `latency_common` 10 first) | 1 to 2 days, mechanical, agent-sized units | `pnp config show --diff` empty after each file; smoke test of the owning task |
| 3 | Move modules into `pnp/` by import count, rewrite imports, delete the 162 `sys.path` hacks; add `pyproject.toml` and `pip install -e .` into each env | 1 to 2 days | `python -c "import pnp.<module>"` in each env listed for it; full `pytest` |
| 4 | Collapse the six dataset generators into `pnp.data.generate` plus six experiment configs; the `_zed` pairs into flags; archive `env_v2`, `env_v3` | 2 to 3 days | byte-identical scene XML and `meta.json` for one seed per variant before and after |
| 5 | `pnp pi sync` with version stamping; retire the June `mycobot_mpc` copy; `runlog` lands per its own design | 1 day, needs the robot | the Pi's `:9998` status reports the synced commit |

Each phase is independently useful and independently revertible. An agent can execute phase 2 and
3 units one file at a time against the gates, which is the point: the work becomes a queue of small
verified edits instead of one large risky change.

## 6. What an agent can do after phase 0 that it could not do before

```
pnp tasks                         # every supported thing to run, with its interpreter
pnp run train_zed --set train.epochs=5
pnp config show --diff            # what is non-default right now, and why
pnp config validate               # before any robot motion
pnp env check                     # are the five environments present and importable
```

and read `AGENTS.md` instead of 869 lines of README to find out which of those applies.
