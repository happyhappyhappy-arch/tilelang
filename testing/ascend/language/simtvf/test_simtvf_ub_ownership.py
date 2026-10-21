"""Source and numerical integration for automatic SIMT UB ownership."""

import pytest
import tilelang
import tilelang.testing
from tilelang.ascend import language as T


def elementwise_kernel(n=1048576, tile=8192, threads=2048, blocks=64, affine=False, dtype="float32"):
    @T.prim_func
    def main(A: T.Tensor((n,), dtype), B: T.Tensor((n,), dtype), C: T.Tensor((n,), dtype)):
        with T.Kernel(blocks) as block:
            a = T.alloc_shared((tile,), dtype)
            b = T.alloc_shared((tile,), dtype)
            c = T.alloc_shared((tile,), dtype)
            for k in T.serial(n // (blocks * tile)):
                offset = (k * blocks + block) * tile
                T.copy(A[offset : offset + tile], a)
                T.copy(B[offset : offset + tile], b)
                with T.SimtVF(threads=threads):
                    for i in T.Parallel(tile):
                        if affine:
                            c[i] = a[i] * T.float32(2.0) + b[i]
                        else:
                            c[i] = a[i] + b[i]
                T.copy(c, C[offset : offset + tile])

    return main


def test_ownership_full_lowering():
    with tilelang.tvm.target.Target("ascend"):
        lowered = tilelang.lower(elementwise_kernel(), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "int32"])
@pytest.mark.parametrize("tile,threads", [(258, 128), (1024, 33)])
def test_ownership_types_and_tails_full_lowering(dtype, tile, threads):
    snapshots = {}

    @tilelang.tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLayoutInference":
                snapshots["inferred"] = mod

    with (
        tilelang.tvm.target.Target("ascend"),
        tilelang.transform.PassContext(instruments=[Capture()]),
    ):
        lowered = tilelang.lower(elementwise_kernel(n=tile, tile=tile, threads=threads, blocks=1, dtype=dtype), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source
    assert "if (" in lowered.kernel_source
    from testing.ascend._ir import nodes
    from tilelang import tvm
    from tvm import tirx

    layouts = [
        loop.annotations["parallel_loop_layout"]
        for loop in nodes(snapshots["inferred"], tirx.For)
        if "parallel_loop_layout" in loop.annotations
    ]
    assert len(layouts) == 1
    analyzer = tvm.arith.Analyzer()
    # Width is inferred, including packed f16/bf16 tails. Every logical
    # element must still have a distinct valid (thread, slot) assignment.
    assignments = set()
    for index in range(tile):
        thread = int(analyzer.simplify(layouts[0].map_forward_thread([index])[0]))
        slots = tuple(int(analyzer.simplify(x)) for x in layouts[0].map_forward_index([index]))
        assert 0 <= thread < threads
        assignments.add((thread, slots))
    assert len(assignments) == tile


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("threads", [128, 2048])
def test_ownership_numerical(threads):
    import torch

    n = 8192
    compiled = tilelang.compile(
        elementwise_kernel(n=n, blocks=1, threads=threads),
        target="ascend",
        out_idx=[2],
    )
    a = (torch.arange(n, dtype=torch.float32) % 251 - 125).npu()
    b = (torch.arange(n, dtype=torch.float32) % 67 - 33).npu()
    torch.testing.assert_close(compiled(a, b).cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)
