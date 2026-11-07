"""Exploring edit 3: E-002 at a second strength, the Gated DeltaNet state in layers 4 to 22 forgets four times as fast.

E-002 (twice as fast) raised pass on MATH-500 exploring questions from 18.2% to 28.7% and kept finished answers on the
questions the default could pass 98.7% right (STUDY_LOG, 7 Nov). That leaves room to push further. Adding ln 4 to A_log
raises every head's per-token decay to the fourth power, so each head's memory half-life is a quarter of the original
(a head at 0.999 per token goes from about 693 tokens to 173). Same tensors as E-002.
"""
import math

L = "model.language_model.layers."
FULL_ATTENTION = {3, 7, 11, 15, 19, 23}
LAYERS = [i for i in range(4, 23) if i not in FULL_ATTENTION]
SHIFT = math.log(4.0)


def edit(t):
    return {f"{L}{i}.linear_attn.A_log": t[f"{L}{i}.linear_attn.A_log"] + SHIFT for i in LAYERS}
