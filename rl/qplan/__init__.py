"""qplan :: Q-Planning (arXiv 2608.21204) on the Pro 630 twin.

Frozen policy pi (rl/weights/resid3_fast_best.pt) + an off-policy Q over ACTION CHUNKS
(HL-Gauss categorical regression, two heads) that re-ranks N chunk proposals at run time.
Only Q is ever retrained; pi is frozen.

    collect.py   roll pi / an arbitrary proposal / the planner in the twin -> episode shards
    critic.py    Q_phi(obs, chunk): MLP [512,512,256], 51-bin HL-Gauss, EMA target, H-step bootstrap
    planner.py   N=32 chunk proposals, Q-filter + softmax-Q weighting; paired eval + ablations
    iterate.py   the self-improvement loop (deploy -> append -> retrain -> eval)
"""
from .common import H, ACT_DIM, OBS_DIM, CHUNK_DIM  # noqa: F401
