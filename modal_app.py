"""Modal harness: one warm GPU container serving batched generation + activation
capture + steering for the Bertrand collusion experiments.

Design notes
------------
* HF transformers, NOT vLLM. The entire project depends on forward hooks, and a
  paged-attention server does not expose the residual stream.
* Throughput comes from batching across EPISODES, not across time. Agents move
  sequentially within an episode, but N independent episodes batch cleanly at
  each round.
* The container is kept warm (scaledown_window) so we pay the model load once
  rather than on every experiment.
* The experiment logic lives in `swayambhu/`, not here, so the identical code
  runs against `LocalRunner` on a laptop GPU.

Usage
-----
    modal run modal_app.py::gate1
    SWAY_MODEL=Qwen/Qwen2.5-14B-Instruct SWAY_GPU=A100-80GB modal run modal_app.py::gate1

NOTE: A100/H100/H200/A10G all require a payment method on the Modal account,
regardless of credit balance.
"""

from __future__ import annotations

import os

import modal

MODEL = os.environ.get("SWAY_MODEL", "Qwen/Qwen2.5-7B-Instruct")
GPU = os.environ.get("SWAY_GPU", "A10G")

app = modal.App("swayambhu")
cache = modal.Volume.from_name("swayambhu-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # transformers pinned below 5.x: v5 renames load kwargs, and we are not
    # spending sprint hours discovering its other breaking changes.
    # scipy is needed in the container too: step() parses prices, which imports
    # game.py for the grid, which solves the Nash/cartel FOCs.
    .pip_install(
        "torch", "transformers>=4.45,<5", "accelerate", "safetensors", "numpy", "scipy"
    )
    .env(
        {
            "HF_HOME": "/cache/hf",
            "TOKENIZERS_PARALLELISM": "false",
            # Baked in so the container reports the same choice as the launcher.
            # Without SWAY_GPU here the container falls back to the default and
            # info() misreports the hardware -- which would then be wrong in the
            # paper's methods section.
            "SWAY_MODEL": MODEL,
            "SWAY_GPU": GPU,
            "SWAY_DEVICE_MAP": os.environ.get("SWAY_DEVICE_MAP", "cuda"),
        }
    )
    .add_local_python_source("swayambhu")
)


@app.cls(
    gpu=GPU,
    image=image,
    volumes={"/cache": cache},
    timeout=60 * 60,
    scaledown_window=900,
    # >1 so a scale-ladder run at another model size can proceed alongside the
    # main capture run; each SWAY_MODEL bakes a distinct image, so they land in
    # separate containers and separate volume paths.
    max_containers=3,
)
class Runner:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from swayambhu.capture import find_block_list

        self.torch = torch
        self.model_name = os.environ.get("SWAY_MODEL", MODEL)
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # transformers 5.x renamed `torch_dtype` -> `dtype`. Try both, then ASSERT
        # the dtype took: an ignored kwarg would silently load fp32 and quietly
        # halve our effective batch size.
        # "cuda" pins everything to device 0; "auto" shards across GPUs, which a
        # 72B in bf16 (~145GB) needs. Hooks are unaffected: captured tensors are
        # moved to CPU, and steering vectors follow h.device per block.
        kw = dict(
            device_map=os.environ.get("SWAY_DEVICE_MAP", "cuda"),
            attn_implementation="sdpa",
        )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, dtype=torch.bfloat16, **kw
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, torch_dtype=torch.bfloat16, **kw
            )
        self.model.eval()
        got = next(self.model.parameters()).dtype
        assert got == torch.bfloat16, f"expected bfloat16, loaded {got}"

        self.block_path, self.blocks = find_block_list(self.model)
        self.n_layers = len(self.blocks)
        self.d_model = int(self.model.config.hidden_size)
        print(
            "[load] {}: blocks at {!r}, n_layers={}, d_model={}".format(
                self.model_name, self.block_path, self.n_layers, self.d_model
            )
        )

    # ---------- internals ----------

    def _texts(self, chats, suffix=""):
        """`suffix` may be a single string or one string per chat."""
        if isinstance(suffix, str):
            suffix = [suffix] * len(chats)
        return [
            self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) + s
            for c, s in zip(chats, suffix)
        ]

    def _encode(self, chats, suffix=""):
        batch = self.tok(self._texts(chats, suffix), return_tensors="pt", padding=True)
        return {k: v.to("cuda") for k, v in batch.items()}

    def _intervention(self, spec):
        """Build the hook context manager described by `spec` (or a no-op)."""
        import contextlib

        from swayambhu.capture import ablate, steer

        if not spec:
            return contextlib.nullcontext()
        v = self.torch.tensor(spec["vector"], dtype=self.torch.float32, device="cuda")
        if spec.get("mode", "add") == "ablate":
            return ablate(self.blocks, int(spec["layer"]), v)
        mask = spec.get("row_mask")
        if mask is not None:
            mask = self.torch.tensor(mask, dtype=self.torch.float32, device="cuda")
        return steer(
            self.blocks,
            int(spec["layer"]),
            v,
            float(spec["alpha"]),
            spec.get("positions", "all"),
            spec.get("relative", True),
            mask,
        )

    # ---------- public API ----------

    @modal.method()
    def info(self) -> dict:
        return {
            "model": self.model_name,
            "gpu": os.environ.get("SWAY_GPU", GPU),
            "block_path": self.block_path,
            "n_layers": self.n_layers,
            "d_model": self.d_model,
            "block_type": type(self.blocks[0]).__name__,
        }

    @modal.method()
    def capture(self, chats: list, layers: list | None = None, suffix: str = ""):
        """Residual stream at the final prompt position. -> [batch, len(layers), d_model]."""
        import numpy as np

        from swayambhu.capture import capture_last_token

        layers = list(range(self.n_layers)) if layers is None else [int(x) for x in layers]
        enc = self._encode(chats, suffix)
        with self.torch.no_grad(), capture_last_token(self.blocks, layers) as store:
            self.model(**enc)
        return np.stack([store[i].float().numpy() for i in layers], axis=1).astype("float16")

    @modal.method()
    def generate(
        self,
        chats: list,
        max_new_tokens: int = 96,
        temperature: float = 0.0,
        seed: int | None = None,
        steering: dict | None = None,
        suffix: str = "",
    ) -> list:
        if seed is not None:
            self.torch.manual_seed(seed)
        enc = self._encode(chats, suffix)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with self.torch.no_grad(), self._intervention(steering):
            out = self.model.generate(**enc, **kwargs)
        gen = out[:, enc["input_ids"].shape[1] :]
        return self.tok.batch_decode(gen, skip_special_tokens=True)

    @modal.method()
    def step(
        self,
        chats: list,
        comms: str,
        cfg: dict,
        layers: list | None = None,
        max_new_tokens: int = 110,
        temperature: float = 0.7,
        seed: int | None = None,
        run_id: str = "main",
        tag: str = "",
        steering: dict | None = None,
        do_capture: bool = True,
    ) -> dict:
        """One round for a whole batch of (episode x firm) prompts.

        capture -> generate -> parse -> second capture, all server-side, so the
        wire carries only text. Activations land on the Modal volume, because a
        full run would otherwise ship ~14GB of fp16 across the network.
        """
        import os as _os

        import numpy as np

        from swayambhu.agents import parse_response
        from swayambhu.capture import capture_last_token
        from swayambhu.game import BertrandConfig

        torch = self.torch
        bcfg = BertrandConfig(**cfg)
        layers = list(range(self.n_layers)) if layers is None else [int(x) for x in layers]

        def cap(suffix):
            enc = self._encode(chats, suffix)
            with torch.no_grad(), capture_last_token(self.blocks, layers) as store:
                self.model(**enc)
            return np.stack([store[i].float().numpy() for i in layers], axis=1).astype("float16")

        # (1) decision state before anything has been generated
        pre_reasoning = cap("") if do_capture else None

        # (2) the turn itself
        if seed is not None:
            torch.manual_seed(seed)
        enc = self._encode(chats)
        kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kwargs.update(do_sample=False)
        with torch.no_grad(), self._intervention(steering):
            out = self.model.generate(**enc, **kwargs)
        texts = self.tok.batch_decode(
            out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )

        # (3) state after reasoning, immediately before the price digits
        saved = None
        if do_capture:
            prefixes = [parse_response(t, bcfg, comms).prefix_to_price for t in texts]
            pre_price = cap(prefixes)
            d = f"/cache/acts/{run_id}"
            _os.makedirs(d, exist_ok=True)
            saved = f"{d}/{tag.replace('/', '__')}.npz"
            np.savez(
                saved,
                pre_reasoning=pre_reasoning,
                pre_price=pre_price,
                layers=np.array(layers, dtype="int32"),
            )
            cache.commit()

        return {"texts": texts, "saved": saved, "layers": layers, "n": len(texts)}


@app.local_entrypoint()
def gate1():
    """Shared Gate 1 experiment, run against the Modal backend."""
    import json
    from pathlib import Path

    from swayambhu.gate1 import run_gate1

    result = run_gate1(Runner(), out_dir=Path("data"))
    Path("data").mkdir(exist_ok=True)
    Path("data/gate1_modal.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("\nwrote data/gate1_modal.json")
