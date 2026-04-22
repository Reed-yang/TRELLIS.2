"""flex_gemm submanifold_conv3d backward patch for frozen-weight compatibility.

Without this patch, training with `weight.requires_grad=False` crashes inside
the triton backward kernel (grad_weight returned as None → reshape on None).
Also fixes a secondary bias=None crash.

Call `_patch_flex_gemm_frozen_weight_bug()` once at process start (idempotent).
"""


def _patch_flex_gemm_frozen_weight_bug():
    try:
        from flex_gemm.ops.spconv.submanifold_conv3d import SubMConv3dFunction
    except Exception as e:
        print(f"[patch] flex_gemm not importable, skipping patch: {e}")
        return

    if getattr(SubMConv3dFunction, "_trellis_freeze_patch", False):
        return

    def _patched_backward(ctx, grad_output, _):
        feats, weight, bias = ctx.saved_tensors
        neighbor_cache = ctx.neighbor_cache
        want_input = feats.requires_grad
        want_weight = weight.requires_grad
        want_bias = bias is not None and bias.requires_grad

        w_prev = weight.requires_grad
        if not w_prev:
            weight.requires_grad_(True)
        try:
            grad_input, grad_weight, grad_bias = (
                SubMConv3dFunction._sparse_submanifold_conv_backward(
                    grad_output, feats, neighbor_cache, weight, bias
                )
            )
        finally:
            if not w_prev:
                weight.requires_grad_(False)

        if not want_input:
            grad_input = None
        if not want_weight:
            grad_weight = None
        if not want_bias:
            grad_bias = None
        return grad_input, None, None, None, grad_weight, grad_bias, None

    SubMConv3dFunction.backward = staticmethod(_patched_backward)
    SubMConv3dFunction._trellis_freeze_patch = True
    print("[patch] applied flex_gemm SubMConv3dFunction.backward frozen-weight fix")
