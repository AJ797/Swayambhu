"""Local GPU backend exposing the SAME interface as the Modal `Runner`.

Every method is wrapped so that `runner.step.remote(...)` works identically
whether we are on a Modal A100 or a 6GB laptop card. Experiment scripts are
therefore backend-agnostic: nothing but an env var changes.

The 6GB-card concession is `micro_batch`: a round's (episodes x firms) prompts
are chunked automatically, so batch size is a memory knob rather than a
correctness constraint.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .agents import parse_response
from .capture import ablate, capture_last_token, steer
from .game import BertrandConfig

DEFAULT_MODEL = os.environ.get("SWAY_LOCAL_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


class _Bound:
    """Makes `obj.method.remote(...)` work for a plain local callable."""

    def __init__(self, fn):
        self._fn = fn

    def remote(self, *a, **k):
        return self._fn(*a, **k)

    def __call__(self, *a, **k):
        return self._fn(*a, **k)


class LocalRunner:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        act_root: Path = Path("data/acts"),
        micro_batch: int = 12,
        device: str = "cuda",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.act_root = Path(act_root)
        self.micro_batch = micro_batch
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        kw = dict(attn_implementation="sdpa")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.bfloat16, **kw
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, **kw
            )
        self.model.to(device).eval()
        got = next(self.model.parameters()).dtype
        assert got == torch.bfloat16, f"expected bfloat16, loaded {got}"

        from .capture import find_block_list

        self.block_path, self.blocks = find_block_list(self.model)
        self.n_layers = len(self.blocks)
        self.d_model = int(self.model.config.hidden_size)

        # Modal-compatible call surface
        self.info = _Bound(self._info)
        self.capture = _Bound(self._capture)
        self.generate = _Bound(self._generate)
        self.step = _Bound(self._step)

        print(
            f"[local] {model_name}: blocks at {self.block_path!r}, "
            f"n_layers={self.n_layers}, d_model={self.d_model}, "
            f"micro_batch={micro_batch}"
        )

    # ---------- internals ----------

    def _texts(self, chats, suffix=""):
        if isinstance(suffix, str):
            suffix = [suffix] * len(chats)
        return [
            self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) + s
            for c, s in zip(chats, suffix)
        ]

    def _encode(self, chats, suffix=""):
        b = self.tok(self._texts(chats, suffix), return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in b.items()}

    def _intervention(self, spec):
        import contextlib

        if not spec:
            return contextlib.nullcontext()
        v = self.torch.tensor(spec["vector"], dtype=self.torch.float32, device=self.device)
        if spec.get("mode", "add") == "ablate":
            return ablate(self.blocks, int(spec["layer"]), v)
        mask = spec.get("row_mask")
        if mask is not None:
            mask = self.torch.tensor(mask, dtype=self.torch.float32, device=self.device)
        return steer(
            self.blocks, int(spec["layer"]), v, float(spec["alpha"]),
            spec.get("positions", "all"), spec.get("relative", True), mask,
        )

    def _chunks(self, n):
        mb = max(1, self.micro_batch)
        return [(i, min(i + mb, n)) for i in range(0, n, mb)]

    def _cap(self, chats, layers, suffix=""):
        """Chunked capture -> [n, len(layers), d_model] fp16."""
        if isinstance(suffix, str):
            suffix = [suffix] * len(chats)
        out = []
        for lo, hi in self._chunks(len(chats)):
            enc = self._encode(chats[lo:hi], suffix[lo:hi])
            with self.torch.no_grad(), capture_last_token(self.blocks, layers) as store:
                self.model(**enc)
            out.append(np.stack([store[i].float().numpy() for i in layers], axis=1))
            del enc
            self.torch.cuda.empty_cache()
        return np.concatenate(out, axis=0).astype("float16")

    def _gen(self, chats, max_new_tokens, temperature, seed, steering, suffix=""):
        if isinstance(suffix, str):
            suffix = [suffix] * len(chats)
        texts = []
        for lo, hi in self._chunks(len(chats)):
            if seed is not None:
                self.torch.manual_seed(seed + lo)
            # a row_mask is indexed against the FULL batch, so slice it to the
            # chunk or the intervention lands on the wrong agents
            spec = steering
            if steering and steering.get("row_mask") is not None:
                spec = dict(steering)
                spec["row_mask"] = list(steering["row_mask"])[lo:hi]
            enc = self._encode(chats[lo:hi], suffix[lo:hi])
            kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature and temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                kwargs.update(do_sample=False)
            with self.torch.no_grad(), self._intervention(spec):
                out = self.model.generate(**enc, **kwargs)
            texts += self.tok.batch_decode(
                out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
            )
            del enc, out
            self.torch.cuda.empty_cache()
        return texts

    # ---------- public API (mirrors modal_app.Runner) ----------

    def _info(self) -> dict:
        return {
            "model": self.model_name,
            "gpu": "local:" + self.torch.cuda.get_device_name(0),
            "block_path": self.block_path,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "block_type": type(self.blocks[0]).__name__,
        }

    def _capture(self, chats, layers=None, suffix=""):
        layers = list(range(self.n_layers)) if layers is None else [int(x) for x in layers]
        return self._cap(chats, layers, suffix)

    def _generate(
        self, chats, max_new_tokens=96, temperature=0.0, seed=None, steering=None, suffix=""
    ):
        return self._gen(chats, max_new_tokens, temperature, seed, steering, suffix)

    def _step(
        self,
        chats,
        comms,
        cfg,
        layers=None,
        max_new_tokens=110,
        temperature=0.7,
        seed=None,
        run_id="main",
        tag="",
        steering=None,
        do_capture=True,
    ) -> dict:
        bcfg = BertrandConfig(**cfg)
        layers = list(range(self.n_layers)) if layers is None else [int(x) for x in layers]

        pre_reasoning = self._cap(chats, layers, "") if do_capture else None
        texts = self._gen(chats, max_new_tokens, temperature, seed, steering, "")

        saved = None
        if do_capture:
            prefixes = [parse_response(t, bcfg, comms).prefix_to_price for t in texts]
            pre_price = self._cap(chats, layers, prefixes)
            d = self.act_root / run_id
            d.mkdir(parents=True, exist_ok=True)
            saved = str(d / (tag.replace("/", "__") + ".npz"))
            np.savez(
                saved,
                pre_reasoning=pre_reasoning,
                pre_price=pre_price,
                layers=np.array(layers, dtype="int32"),
            )

        return {"texts": texts, "saved": saved, "layers": layers, "n": len(texts)}
