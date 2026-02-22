# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Benchmark RoPE backward and full pass performance."""

import torch
import triton
import triton.testing

import tilegym
from tilegym.backend import is_backend_available
from tilegym.backend import register_impl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor = None,  # Unused, kept for compatibility
    unsqueeze_dim: int = 1,
    use_tma: bool = False,  # Unused, kept for compatibility
):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


register_impl("apply_rope_base", "torch")(apply_rope_torch)


def create_rotary_embeddings(seq_len: int, head_dim: int, dtype: torch.dtype, device, base: float = 10000.0):
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)

    cos_half = torch.cos(freqs).to(dtype)
    sin_half = torch.sin(freqs).to(dtype)

    cos = torch.cat([cos_half, cos_half], dim=-1)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def get_providers():
    providers = [("torch", "PyTorch", ("green", "-"))]
    if is_backend_available("cutile"):
        providers.insert(0, ("cutile", "CuTile", ("orange", "-")))
    return providers


def create_benchmark_config(mode: str, dtype: torch.dtype, bsz: int, num_heads: int, head_dim: int):
    providers = get_providers()
    if not providers:
        return None

    backends, names, styles = zip(*providers)
    dtype_name = str(dtype).split(".")[-1]

    return triton.testing.Benchmark(
        x_names=["SEQ_LEN"],
        x_vals=[2**i for i in range(10, 15)],  # 1024 to 16384
        line_arg="backend",
        line_vals=list(backends),
        line_names=list(names),
        styles=list(styles),
        ylabel="GB/s",
        plot_name=f"rope-{mode}-bsz{bsz}-heads{num_heads}-d{head_dim}-{dtype_name}-GBps",
        args={
            "BSZ": bsz,
            "NUM_HEADS": num_heads,
            "HEAD_DIM": head_dim,
            "datatype": dtype,
            "mode": mode,
        },
    )


@triton.testing.perf_report(
    [
        create_benchmark_config(mode, dtype, bsz, num_heads, head_dim)
        for mode in ["backward", "full"]
        for dtype in [torch.float16]
        for bsz in [1]
        for num_heads in [16]
        for head_dim in [64]
    ]
)
def bench_rope_backward(
    BSZ,
    NUM_HEADS,
    SEQ_LEN,
    HEAD_DIM,
    backend,
    datatype,
    mode,
    device=DEVICE,
):
    dtype = datatype

    # Correctness check on a tiny config first.
    q_ref = torch.randn((1, NUM_HEADS, 128, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    k_ref = torch.randn((1, NUM_HEADS, 128, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
    q_opt = q_ref.detach().clone().requires_grad_(True)
    k_opt = k_ref.detach().clone().requires_grad_(True)

    pos_ids_ref = torch.arange(128, device=device, dtype=torch.long).unsqueeze(0)
    cos_ref, sin_ref = create_rotary_embeddings(128, HEAD_DIM, dtype, device)
    cos_ref = cos_ref.unsqueeze(0)
    sin_ref = sin_ref.unsqueeze(0)

    tilegym.set_backend("torch")
    oq_t, ok_t = tilegym.ops.apply_rope_base(q_ref, k_ref, cos_ref, sin_ref, pos_ids_ref)
    gq_t = torch.randn_like(oq_t)
    gk_t = torch.randn_like(ok_t)
    (oq_t * gq_t + ok_t * gk_t).sum().backward()

    tilegym.set_backend(backend)
    oq_c, ok_c = tilegym.ops.apply_rope_base(q_opt, k_opt, cos_ref, sin_ref, pos_ids_ref)
    (oq_c * gq_t + ok_c * gk_t).sum().backward()

    torch.testing.assert_close(oq_c, oq_t, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(ok_c, ok_t, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q_opt.grad, q_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k_opt.grad, k_ref.grad, atol=2e-2, rtol=2e-2)

    # Benchmark tensors.
    q_base = torch.randn((BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device)
    k_base = torch.randn((BSZ, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device=device)
    pos_ids = torch.arange(SEQ_LEN, device=device, dtype=torch.long).unsqueeze(0).expand(BSZ, -1)
    cos, sin = create_rotary_embeddings(SEQ_LEN, HEAD_DIM, dtype, device)
    cos = cos.unsqueeze(0).expand(BSZ, -1, -1)
    sin = sin.unsqueeze(0).expand(BSZ, -1, -1)

    bytes_per_element = q_base.element_size()
    qk_elems = BSZ * NUM_HEADS * SEQ_LEN * HEAD_DIM
    cos_sin_elems = 2 * BSZ * SEQ_LEN * HEAD_DIM

    if mode == "backward":
        q = q_base.detach().clone().requires_grad_(True)
        k = k_base.detach().clone().requires_grad_(True)
        tilegym.set_backend(backend)
        out_q, out_k = tilegym.ops.apply_rope_base(q, k, cos, sin, pos_ids)
        grad_q = torch.randn_like(out_q)
        grad_k = torch.randn_like(out_k)

        def bwd_only():
            torch.autograd.backward((out_q, out_k), (grad_q, grad_k), retain_graph=True)

        ms = triton.testing.do_bench(bwd_only, rep=10)
        # Approximate backward traffic: read dq/dk + read cos/sin + read/write q/k grads.
        total_bytes = (4 * qk_elems + cos_sin_elems) * bytes_per_element
    else:

        def full_pass():
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            tilegym.set_backend(backend)
            out_q, out_k = tilegym.ops.apply_rope_base(q, k, cos, sin, pos_ids)
            grad_q = torch.randn_like(out_q)
            grad_k = torch.randn_like(out_k)
            (out_q * grad_q + out_k * grad_k).sum().backward()

        ms = triton.testing.do_bench(full_pass, rep=10)
        # Forward + backward approximate total traffic.
        fwd_bytes = (4 * qk_elems + cos_sin_elems) * bytes_per_element
        bwd_bytes = (4 * qk_elems + cos_sin_elems) * bytes_per_element
        total_bytes = fwd_bytes + bwd_bytes

    gb_per_s = total_bytes * 1e-9 / (ms * 1e-3)
    return gb_per_s


if __name__ == "__main__":
    bench_rope_backward.run(print_data=True)
