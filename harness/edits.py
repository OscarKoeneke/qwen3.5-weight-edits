"""Turning edit scripts into weights, measuring them, and making the random controls (SPEC.md sections 6 and 8).

An edit is a Python file with edit(tensors), which gets a private copy of every checkpoint tensor and returns a dict of
only the tensors it changed. Everything is measured on the values vLLM will hold, which can be coarser than the
checkpoint: an edit that rounds away there is refused. vllm_param_dtypes.json records what vLLM holds, read from a real load.
"""
import importlib.util
import json
import re
import shutil

import torch
from safetensors.torch import load_file, save_file

from common import HARNESS, MODEL_DIR, SEED, seed_for, sha256

WEIGHTS_FILE = "model.safetensors-00001-of-00001.safetensors"
EDITABLE = "model.language_model."
CONTROL_TOLERANCE = 0.02
GROUP = re.compile(r"layers\.(\d+)\.(self_attn|linear_attn|mlp|input_layernorm|post_attention_layernorm)")
HELD = json.loads((HARNESS / "vllm_param_dtypes.json").read_text(encoding="utf-8"))


class BadEdit(Exception):
    pass


def held_dtype(name):
    for suffix in HELD["float32"]:
        if name.endswith(suffix):
            return torch.float32
    return torch.bfloat16


def held(name, tensor):
    return tensor.to(held_dtype(name)).float()


def load_original():
    return load_file(MODEL_DIR / WEIGHTS_FILE)


def run_script(script, original):
    spec = importlib.util.spec_from_file_location("edit_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.manual_seed(SEED)
    return module.edit({k: v.clone() for k, v in original.items()})


def footprint(original, changed):
    """Check an edit and measure it, per tensor. Refuses anything vLLM wouldn't load as written."""
    if not isinstance(changed, dict) or not changed:
        raise BadEdit("edit() must return a dict of the tensors it changed")
    fp = {}
    for name, new in sorted(changed.items()):
        if name not in original or not name.startswith(EDITABLE):
            raise BadEdit(f"{name}: not a language model tensor in this checkpoint")
        old = original[name]
        if new.shape != old.shape or new.dtype != old.dtype:
            raise BadEdit(f"{name}: shape or dtype changed")
        if not torch.isfinite(new.float()).all():
            raise BadEdit(f"{name}: has NaN or infinite values")
        delta = held(name, new) - held(name, old)
        count = int((delta != 0).sum())
        if count == 0:
            raise BadEdit(f"{name}: the change rounds away in {held_dtype(name)}, so vLLM would never see it")
        fp[name] = {"changed": count, "changed_in_file": int((new != old).sum()), "l2": float(delta.norm()),
                    "numel": old.numel(), "file_dtype": str(old.dtype), "held_dtype": str(held_dtype(name))}
    return fp


def random_control(original, fp, seed):
    """The control for an edit: same tensors, same number of changed entries per tensor at random positions, random
    Gaussian values scaled so the change has the same norm per tensor, all measured on the values vLLM holds."""
    changed = {}
    for name, f in sorted(fp.items()):
        old = original[name]
        gen = torch.Generator().manual_seed(seed_for(name, seed, 0))
        picks = torch.randperm(old.numel(), generator=gen)[:f["changed"]]
        direction = torch.randn(f["changed"], generator=gen)
        direction = direction / direction.norm()
        before = old.flatten()[picks]
        scale, best = f["l2"], None
        for _ in range(40):
            values = (before.float() + scale * direction).to(old.dtype)
            got = float((held(name, values) - held(name, before)).norm())
            error = abs(got - f["l2"]) / f["l2"]
            if best is None or error < best[0]:
                best = (error, values)
            if error < 0.001:
                break
            scale *= f["l2"] / got if got else 2
        new = old.flatten().clone()
        new[picks] = best[1]
        changed[name] = new.reshape(old.shape)
    made = footprint(original, changed)
    for name, f in fp.items():
        if abs(made[name]["l2"] - f["l2"]) > CONTROL_TOLERANCE * f["l2"] or \
                abs(made[name]["changed"] - f["changed"]) > CONTROL_TOLERANCE * f["changed"]:
            raise BadEdit(f"{name}: random control can't match the edit's size in {held_dtype(name)}")
    return changed, made


def write_model(original, changed, out_dir):
    """A full model folder: the shared files as links, and one weights file with the changes merged in."""
    out_dir.mkdir(parents=True)
    for p in MODEL_DIR.iterdir():
        if p.is_file() and p.name != WEIGHTS_FILE:
            (out_dir / p.name).symlink_to(p)
    merged = {**original, **{k: v.contiguous() for k, v in changed.items()}}
    save_file(merged, out_dir / WEIGHTS_FILE, metadata={"format": "pt"})
    return sha256(out_dir / WEIGHTS_FILE)


def remove_model(out_dir):
    shutil.rmtree(out_dir, ignore_errors=True)


def group_key(name):
    """One name for a block, across checkpoint names and vLLM's fused names. Embeddings are tied to the output head."""
    m = GROUP.search(name)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    if "embed_tokens" in name or "lm_head" in name:
        return "embed_tokens"
    return "final_norm" if name.endswith("norm.weight") else name


def expected_groups(fp):
    return sorted({group_key(n) for n in fp}) if fp else []


def readback_problem(original_table, table, groups):
    """Compare what vLLM holds with the original. Returns a problem, or None if the edit landed where it should."""
    if set(original_table) != set(table):
        return "vLLM's parameter names differ from the original load"
    changed = {group_key(n) for n in table if table[n]["sha256"] != original_table[n]["sha256"]}
    missing, extra = sorted(set(groups) - changed), sorted(changed - set(groups))
    if missing:
        return f"edit didn't reach these blocks: {missing}"
    if extra:
        return f"blocks changed that shouldn't have: {extra}"
    return None
