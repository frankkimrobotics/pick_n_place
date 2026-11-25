"""pnp.config :: layered TOML configuration with provenance.

    from pnp.config import load
    cfg = load()                                  # defaults -> machine -> experiment -> env -> --set
    cfg.robot.base_q, cfg.pi.host, cfg.cameras.wrist.serial
    cfg.origin("pi.vel_cmd_max")                  # -> "configs/defaults.toml:71"
    cfg.diff()                                    # {key: (default, effective, origin)} for non-defaults

Layers (later wins):
  configs/defaults.toml                 every tunable, with unit and source in a comment
  configs/machine/<hostname>.toml       paths, interpreters, serials, hosts (the only machine-specific file)
  configs/experiments/<name>.toml       what one recipe changes       (load(experiment="name"))
  PNP_<SECTION>__<KEY>=value            environment override          (PNP_PI__HOST=10.0.0.27)
  --set section.key=value               command-line override         (load(overrides=[...]))
Values in env/--set are parsed as TOML literals when possible (850 -> int, true -> bool, [1,2] -> list).
Parser: tomllib (3.11+) or tomli (pip install tomli) on 3.8-3.10. No other dependency.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path

try:
    import tomllib as _toml
except ImportError:                                   # Python <= 3.10
    try:
        import tomli as _toml                          # type: ignore
    except ImportError as e:                           # pragma: no cover
        raise ImportError("pnp.config needs tomllib (Python 3.11+) or `pip install tomli`") from e

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _index_lines(text):
    """'table.key' -> line number, from the raw TOML text (tomllib keeps no positions)."""
    idx, table = {}, ""
    for n, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            table = s.strip("[]").strip() + "."
        elif s and not s.startswith("#") and "=" in s:
            idx[table + s.split("=", 1)[0].strip().strip('"')] = n
    return idx


def _literal(raw):
    try:
        return _toml.loads(f"v = {raw}")["v"]
    except Exception:
        return raw


def _set(d, dotted, value):
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


class Cfg:
    """Read-only attribute view over a nested dict."""
    __slots__ = ("_d", "_path")

    def __init__(self, d, path=""):
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, k):
        try:
            v = self._d[k]
        except KeyError:
            raise AttributeError(f"config has no key '{self._path}{k}'") from None
        return Cfg(v, f"{self._path}{k}.") if isinstance(v, dict) else v

    def __getitem__(self, k):
        return getattr(self, k)

    def __setattr__(self, k, v):
        raise AttributeError("config is read-only; use configs/*.toml, PNP_ env vars or --set")

    def __contains__(self, k):
        return k in self._d

    def keys(self):
        return self._d.keys()

    def to_dict(self):
        return dict(self._d)

    def __repr__(self):
        return f"Cfg({self._path.rstrip('.') or 'root'}: {list(self._d)})"


class Config(Cfg):
    __slots__ = ("_origins", "_defaults", "_layers")

    def __init__(self, d, origins, defaults, layers):
        super().__init__(d)
        object.__setattr__(self, "_origins", origins)
        object.__setattr__(self, "_defaults", defaults)
        object.__setattr__(self, "_layers", layers)

    def origin(self, dotted):
        return self._origins.get(dotted, "unset")

    def get(self, dotted, default=None):
        return _flatten(self._d).get(dotted, default)

    def flat(self):
        return _flatten(self._d)

    def diff(self):
        """values that differ from defaults+machine: {key: (baseline, effective, origin)}"""
        flat, dflt = self.flat(), self._defaults
        return {k: (dflt.get(k), v, self._origins.get(k)) for k, v in flat.items() if dflt.get(k) != v}

    @property
    def layers(self):
        return list(self._layers)


def machine_file(hostname=None):
    return CONFIGS / "machine" / f"{hostname or socket.gethostname()}.toml"


def load(experiment=None, machine=None, overrides=None, env=True):
    """Build the effective config. `overrides` = ["section.key=value", ...]."""
    merged, origins, layers = {}, {}, []

    def apply_file(path, label):
        if not path.exists():
            return
        text = path.read_text()
        data, lines = _toml.loads(text), _index_lines(text)
        rel = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)
        for k, v in _flatten(data).items():
            _set(merged, k, v)
            origins[k] = f"{rel}:{lines.get(k, '?')}"
        layers.append(f"{label}: {rel}")

    apply_file(CONFIGS / "defaults.toml", "defaults")
    apply_file(machine_file(machine), "machine")
    defaults = dict(_flatten(merged))          # baseline = defaults + machine; diff() shows only experiment/env/--set
    if experiment:
        p = Path(experiment)
        apply_file(p if p.suffix == ".toml" else CONFIGS / "experiments" / f"{experiment}.toml", "experiment")
    if env:
        for name, raw in os.environ.items():
            if name.startswith("PNP_") and "__" in name:
                key = name[4:].lower().replace("__", ".")
                _set(merged, key, _literal(raw)); origins[key] = f"env:{name}"; layers.append(f"env: {name}")
    for item in overrides or []:
        key, _, raw = item.partition("=")
        _set(merged, key.strip(), _literal(raw.strip())); origins[key.strip()] = "--set"; layers.append(f"set: {item}")
    return Config(merged, origins, defaults, layers)
