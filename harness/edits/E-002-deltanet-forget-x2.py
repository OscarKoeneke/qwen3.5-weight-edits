"""Exploring edit 2: make the Gated DeltaNet state forget twice as fast in layers 4 to 22.

The loop map found the running state changes about 35 to 55% less inside a loop in layers 4 to 22 (STUDY_LOG, 6 Nov
12:45 UTC). That is a weak lead, correlation only, and repeated input alone could cause it. The per-step decay is
exp(-A * softplus(a + dt_bias)) with A = exp(A_log), so adding ln 2 to A_log squares every head's decay at every token,
whatever the input: each head's memory half-life halves (a head at 0.999 per token goes from about 693 tokens to 346).
A_log is held in float32 by vLLM, so the change can't round away.
"""
import math

L = "model.language_model.layers."
FULL_ATTENTION = {3, 7, 11, 15, 19, 23}
LAYERS = [i for i in range(4, 23) if i not in FULL_ATTENTION]
SHIFT = math.log(2.0)


def edit(t):
    return {f"{L}{i}.linear_attn.A_log": t[f"{L}{i}.linear_attn.A_log"] + SHIFT for i in LAYERS}
