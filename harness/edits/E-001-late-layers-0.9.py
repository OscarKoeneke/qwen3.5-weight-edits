"""Exploring edit 1: scale what layers 22 and 23 write into the residual stream by 0.9.

The loop map's logit lens found a loop's extra certainty in the next token appears only in layers 22 and 23 (STUDY_LOG,
6 Nov 12:45 UTC). That is a weak lead, correlation only. This edit shrinks the four tensors those layers write with
(the attention or DeltaNet output and the MLP output) by 10%, a mild first strength.
"""
L = "model.language_model.layers."
NAMES = ("22.linear_attn.out_proj.weight", "22.mlp.down_proj.weight", "23.self_attn.o_proj.weight",
         "23.mlp.down_proj.weight")
SCALE = 0.9


def edit(t):
    return {L + n: t[L + n] * SCALE for n in NAMES}
