"""realsim.wsp :: weight-space patches for a trained policy (no gradient step, no training loop).

A patch is a small .pt file: {"ops": [op, ...], "note": str}. Each op edits ONE tensor of the
loaded policy in place:

    dict(key="net.blocks.3.attn.out_proj.weight", kind="set" | "add" | "mul", value=tensor,
         index=None | tuple-of-slices-as-lists)        # index selects a sub-block of the tensor
    dict(key="lo" | "hi", ...)                         # the action un-normaliser (policy.lo / policy.hi)

Keys are prefixed with the module: "net." (denoiser) or "obs_enc." (observation encoder).
"""
import torch


def _target(policy, key):
    if key in ("lo", "hi"):
        return getattr(policy, key)
    mod, name = key.split(".", 1)
    return dict(getattr(policy, mod).named_parameters())[name]


def apply_patch(policy, path, verbose=True):
    patch = torch.load(path, map_location="cpu", weights_only=False) if isinstance(path, str) else path
    n_edit = 0
    with torch.no_grad():
        for op in patch["ops"]:
            w = _target(policy, op["key"])
            idx = tuple(slice(*i) for i in op["index"]) if op.get("index") else ...
            v = torch.as_tensor(op["value"]).to(w.device, w.dtype)
            before = w[idx].clone()
            if op["kind"] == "set":
                w[idx] = v
            elif op["kind"] == "add":
                w[idx] += v
            elif op["kind"] == "mul":
                w[idx] *= v
            n_edit += int((w[idx] != before).sum())
    n_all = sum(p.numel() for p in policy.net.parameters()) + sum(p.numel() for p in policy.obs_enc.parameters())
    if verbose:
        print(f"[wsp] {patch.get('note', '')}: {len(patch['ops'])} ops, {n_edit} weights changed "
              f"({100 * n_edit / n_all:.5f} % of {n_all / 1e6:.0f} M)", flush=True)
    return n_edit, n_all
