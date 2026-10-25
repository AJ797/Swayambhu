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
    relative: bool = True,
    row_mask: torch.Tensor | None = None,
):
    """Contrastive activation addition: h <- h + alpha * v at one block's output.

    `row_mask` ([batch] of 0/1) restricts the intervention to some rows of the
    batch. Since a round batches (episode x firm), this lets us steer a single
    firm and ask whether one intervened agent can pull the whole market -- a
    sharper causal claim than steering every agent at once.

    `relative=True` (the default) interprets `alpha` as a FRACTION OF THE LOCAL
    RESIDUAL NORM rather than an absolute coefficient, i.e. the applied step is
    alpha * mean(||h||) * v_hat. This matters a great deal in practice: residual
    norms grow by orders of magnitude with depth, so a fixed absolute alpha that
    visibly steers one layer is an imperceptible nudge at another and a
    destructive blast at a third. Dosing relative to ||h|| makes alpha
    comparable across layers and models, and makes a layer sweep meaningful.
    """
    stats: dict[str, float] = {}

    def hook(_mod, _args, out):
        h = _hidden(out)
        v = vector.to(device=h.device, dtype=torch.float32)
        v = v / (v.norm() + 1e-8)
        h32 = h.to(torch.float32)
        if relative:
            ref = h32.norm(dim=-1).mean()
            step = alpha * ref
        else:
            step = torch.as_tensor(alpha, device=h32.device, dtype=torch.float32)
        stats["resid_norm"] = float(h32.norm(dim=-1).mean())
        stats["step_norm"] = float(step)

        delta = step * v                                  # [d_model]
        delta = delta.unsqueeze(0).unsqueeze(0)           # [1, 1, d_model]
        if row_mask is not None:
            m = row_mask.to(device=h32.device, dtype=torch.float32)
            if m.shape[0] != h32.shape[0]:
                raise ValueError(
                    f"row_mask has {m.shape[0]} rows but batch is {h32.shape[0]}"
                )
            delta = delta * m.view(-1, 1, 1)

        if positions == "all":
            # every position, prompt included. Broader than the CAA convention.
            h32 = h32 + delta
        elif positions == "generated":
            # Contrastive Activation Addition as published adds the vector at
            # token positions AFTER the user's prompt. With a KV cache the
            # prefill pass carries the whole prompt (seq_len > 1) and each
            # decode step carries one new token, so skipping the multi-token
            # pass reproduces that convention.
            if h32.shape[1] == 1:
                h32 = h32 + delta
        elif positions == "last":
            h32 = h32.clone()
            h32[:, -1:, :] = h32[:, -1:, :] + delta
        else:
            raise ValueError(f"unknown positions={positions!r}")
        return _rewrap(out, h32.to(h.dtype))

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield stats
    finally:
        handle.remove()


@contextlib.contextmanager
def residual_norms(blocks: nn.ModuleList, layers: Sequence[int]):
    """Record mean ||h|| per layer. Needed to choose sane steering doses and
    worth reporting, since it explains why a given alpha works where it does."""
    store: dict[int, float] = {}
    handles = []

    def make(i: int):
        def hook(_mod, _args, out):
            store[i] = float(_hidden(out).to(torch.float32).norm(dim=-1).mean())
        return hook

    try:
        for i in layers:
            handles.append(blocks[i].register_forward_hook(make(i)))
        yield store
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def steer_bank(
    blocks: nn.ModuleList,
    layer: int,
    bank: torch.Tensor,
    alpha: float,
    tau: float = 0.05,
    gate: str = "softmax",
    relative: bool = True,
):
    """Token-conditional, softmax-gated mixture over a bank of frozen directions.

        g_i(h_t) = softmax_i( cos(h_t, v_i) / tau )
        h_t  <-  h_t + alpha * ||h_t|| * sum_i g_i(h_t) * v_i_hat

    `bank` is [k, d_model]; rows are L2-normalised here. The gate is
    PARAMETER-FREE -- it reads cosine similarity against the bank rather than a
    learned projection -- so there is nothing to train and nothing to overfit.

    Two properties worth noting against the single-vector `steer`:

    * The mixture is recomputed per token, so different positions receive
      different combinations rather than one global offset.
    * The dose uses ||h_t|| AT EACH POSITION, not the batch-and-sequence mean.
      Single-vector steering rescales by a mean that is taken over the whole
      prompt at prefill and over one token during decode, so its effective
      coefficient silently differs between the two phases. This does not.

    gate="uniform" replaces the softmax with an equal mixture, which isolates
    how much of any effect is due to gating rather than to the bank's span.

    Two calibration notes, both found by unit test rather than by assumption:

    * `tau` must be small. Cosine similarities between a hidden state and a
      direction are typically near zero, so softmax(cos / 1.0) is almost flat
      and the "gate" barely gates -- measured entropy 1.082 against a uniform
      ceiling of ln 3 = 1.099. The default is therefore 0.05, and
      `gate_entropy` is reported so the selectivity actually achieved can be
      checked rather than assumed.
    * The mixture `g @ V` is a convex combination of unit vectors, so its norm
      is at most 1 and is smaller when the gate is undecided. The effective
      dose is thus alpha * ||mix||, not alpha. `mix_norm` is reported so this
      is comparable with single-vector steering instead of silently differing.
    """
    stats: dict[str, float] = {}

    def hook(_mod, _args, out):
        h = _hidden(out)
        h32 = h.to(torch.float32)
        V = bank.to(device=h32.device, dtype=torch.float32)
        V = V / (V.norm(dim=-1, keepdim=True) + 1e-8)          # [k, d]

        norms = h32.norm(dim=-1, keepdim=True)                  # [b, s, 1]
        if gate == "uniform":
            g = torch.full(
                (*h32.shape[:-1], V.shape[0]), 1.0 / V.shape[0], device=h32.device
            )
        else:
            cos = (h32 @ V.T) / (norms + 1e-8)                  # [b, s, k]
            g = torch.softmax(cos / max(tau, 1e-6), dim=-1)
        mix = g @ V                                             # [b, s, d]
        step = alpha * norms if relative else alpha
        stats["mix_norm"] = float(mix.norm(dim=-1).mean())
        stats["gate_max"] = float(g.max())
        stats["gate_entropy"] = float(
            -(g.clamp_min(1e-9) * g.clamp_min(1e-9).log()).sum(-1).mean()
        )
        return _rewrap(out, (h32 + step * mix).to(h.dtype))

    handle = blocks[layer].register_forward_hook(hook)
    try:
        yield stats
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
