"""Architecture-agnostic residual-stream capture and intervention.

Nothing here assumes a Llama-style module naming scheme. We locate the
repeated block list by structure (longest homogeneous nn.ModuleList) and hook
*block outputs*, which is the residual stream for any pre-norm transformer and
the inter-block state for SSM / hybrid stacks. Block outputs may be a bare
tensor or a tuple whose first element is the hidden state; both are handled, so
swapping in a Mamba-hybrid later does not require touching this file.
"""

from __future__ import annotations

import contextlib
from typing import Sequence

import torch
import torch.nn as nn


def find_block_list(model: nn.Module) -> tuple[str, nn.ModuleList]:
    """Locate the stack of repeated blocks. Returns (qualified_name, module_list)."""
    best: tuple[str, nn.ModuleList] | None = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 4:
            kinds = {type(c).__name__ for c in mod}
            if len(kinds) <= 2 and (best is None or len(mod) > len(best[1])):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate a repeated block list in this model")
    return best


def _hidden(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
        return out[0]
    raise TypeError(f"unexpected block output type: {type(out)}")


def _rewrap(out, new_hidden):
    if isinstance(out, torch.Tensor):
        return new_hidden
    rest = list(out[1:])
    return [new_hidden, *rest] if isinstance(out, list) else tuple([new_hidden, *rest])


@contextlib.contextmanager
def capture_last_token(blocks: nn.ModuleList, layers: Sequence[int]):
    """Capture the hidden state at the FINAL sequence position of each forward pass.

    With left-padded batches the final position is the last real token for every
    row, so this is a well-defined 'decision state' across a batch.
    Yields a dict {layer_index: tensor[batch, d_model]} filled on each forward.
    """
    store: dict[int, torch.Tensor] = {}
    handles = []

    def make(i: int):
        def hook(_mod, _args, out):
            store[i] = _hidden(out)[:, -1, :].detach().to(torch.float16).cpu()
        return hook

    try:
        for i in layers:
            handles.append(blocks[i].register_forward_hook(make(i)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def steer(
    blocks: nn.ModuleList,
    layer: int,
    vector: torch.Tensor,
    alpha: float,
    positions: str = "all",
):
    """Contrastive activation addition: h <- h + alpha * v at one block's output."""
    def hook(_mod, _args, out):
        h = _hidden(out)
        v = vector.to(device=h.device, dtype=h.dtype)
        if positions == "all":
            h = h + alpha * v
        elif positions == "last":
            h = h.clone()
            h[:, -1, :] += alpha * v
        else:
            raise ValueError(f"unknown positions={positions!r}")
        return _rewrap(out, h)

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def ablate(blocks: nn.ModuleList, layer: int, vector: torch.Tensor):
    """Directional ablation: project the direction out of the residual stream."""
    def hook(_mod, _args, out):
        h = _hidden(out)
        u = vector.to(device=h.device, dtype=torch.float32)
        u = u / (u.norm() + 1e-8)
        h32 = h.to(torch.float32)
        h32 = h32 - (h32 @ u).unsqueeze(-1) * u
        return _rewrap(out, h32.to(h.dtype))

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextlib.contextmanager
def nullcontext_hooks():
    yield
