"""Source and numerical integration for manually selected UB layouts."""

import pytest
import tilelang
import tilelang.testing
from tilelang.ascend import language as T

STRATEGY = "tl.ascend.ub_layout_strategy"


def transpose_kernel(rows=32, cols=32, tiles=16, threads=128, blocks=1):
    @T.prim_func
    def main(A: T.Tensor((blocks, tiles, rows, cols), "float32"), B: T.Tensor((blocks, tiles, cols, rows), "float32")):
        with T.Kernel(blocks) as block:
            src = T.alloc_shared((rows, cols), "float32")
            stage = T.alloc_shared((rows, cols), "float32")
            dst = T.alloc_shared((cols, rows), "float32")
            for tile in T.serial(tiles):
                T.copy(A[block, tile, :, :], src)
                with T.SimtVF(threads=threads):
                    for i, j in T.Parallel(rows, cols):
                        stage[i, j] = src[i, j] * T.float32(2.0) + T.float32(1.0)
                with T.SimtVF(threads=threads):
                    for i, j in T.Parallel(cols, rows):
                        dst[i, j] = stage[j, i]
                T.copy(dst, B[block, tile, :, :])

    return main


def elementwise_kernel(n=1048576, tile=8192, threads=2048, blocks=64, affine=False):
    @T.prim_func
    def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(blocks) as block:
            a = T.alloc_shared((tile,), "float32")
            b = T.alloc_shared((tile,), "float32")
            c = T.alloc_shared((tile,), "float32")
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


@pytest.mark.parametrize("strategy", ["off", "padding", "swizzle"])
@pytest.mark.parametrize("auto_schedule", [True, False])
def test_storage_full_lowering(strategy, auto_schedule):
    with (
        tilelang.tvm.target.Target("ascend"),
        tilelang.transform.PassContext(config={STRATEGY: strategy, "tl.enable_auto_schedule": auto_schedule}),
    ):
        lowered = tilelang.lower(transpose_kernel(), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source
    if strategy == "swizzle":
        assert "^" in lowered.kernel_source


@pytest.mark.parametrize("ownership", [False, True])
def test_ownership_full_lowering(ownership):
    with tilelang.tvm.target.Target("ascend"), tilelang.transform.PassContext(config={"tl.ascend.ub_layout_ownership": ownership}):
        lowered = tilelang.lower(elementwise_kernel(), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("strategy", ["off", "padding", "swizzle"])
@pytest.mark.parametrize("shape", [(32, 32), (8, 64)])
def test_storage_numerical(strategy, shape):
    import torch

    rows, cols = shape
    func = transpose_kernel(rows, cols, tiles=2)
    compiled = tilelang.compile(func, target="ascend", out_idx=[1], pass_configs={STRATEGY: strategy})
    a = (torch.arange(2 * rows * cols, dtype=torch.float32).reshape(1, 2, rows, cols) % 257 - 128).npu()
    torch.testing.assert_close(compiled(a).cpu(), (a.cpu() * 2 + 1).transpose(-1, -2), rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("threads", [128, 2048])
@pytest.mark.parametrize("ownership", [False, True])
def test_ownership_numerical(threads, ownership):
    import torch

    n = 8192
    compiled = tilelang.compile(
        elementwise_kernel(n=n, blocks=1, threads=threads),
        target="ascend",
        out_idx=[2],
        pass_configs={"tl.ascend.ub_layout_ownership": ownership},
    )
    a = (torch.arange(n, dtype=torch.float32) % 251 - 125).npu()
    b = (torch.arange(n, dtype=torch.float32) % 67 - 33).npu()
    torch.testing.assert_close(compiled(a, b).cpu(), a.cpu() + b.cpu(), rtol=0, atol=0)


def versioned_transpose_kernel():
    @T.prim_func
    def main(A: T.Tensor((4, 32, 32), "float32"), B: T.Tensor((4, 32, 32), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((32, 32), "float32")
            stage = T.alloc_shared((32, 32), "float32")
            dst = T.alloc_shared((32, 32), "float32")
            T.annotate_buffer_versions({stage: (2, "iteration")})
            for tile in T.Pipelined(4, num_stages=2):
                T.copy(A[tile, :, :], src)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(32, 32):
                        stage[i, j] = src[i, j] * T.float32(2) + T.float32(1)
                with T.SimtVF(threads=128):
                    for i, j in T.Parallel(32, 32):
                        dst[i, j] = stage[j, i]
                T.copy(dst, B[tile, :, :])

    return main


@pytest.mark.parametrize("strategy", ["padding", "swizzle"])
def test_storage_physical_pitch_survives_version_expansion(strategy):
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import allocated_buffer, nodes

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AscendLowerTileOp", "tl.RewriteAscendBufferVersionLayout"):
                snapshots[info.name] = mod

    with tvm.target.Target("ascend"), tvm.transform.PassContext(config={STRATEGY: strategy}, instruments=[Capture()]):
        tilelang.lower(versioned_transpose_kernel(), target="ascend")
    buffer = allocated_buffer(snapshots["tl.AscendLowerTileOp"], "stage")
    assert tuple(int(x) for x in buffer.shape) == ((2, 1026) if strategy == "padding" else (2, 32, 32))
    # Padding holes enlarge each version's physical span, requiring alignment.
    # Swizzle preserves the already aligned span, so its shape stays intact.
    aligned = snapshots["tl.RewriteAscendBufferVersionLayout"]
    physical = next(n.buffer for n in nodes(aligned, tirx.AllocBuffer) if n.buffer.name == "stage")
    assert tuple(int(x) for x in physical.shape) == ((2 * 1032,) if strategy == "padding" else (2, 32, 32))


def component_kernel(rows=256, cols=32):
    @T.prim_func
    def main(A: T.Tensor((rows, cols), "float32"), B: T.Tensor((rows, cols), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((rows, cols), "float32")
            first = T.alloc_shared((rows, cols), "float32")
            stage = T.alloc_shared((rows, cols), "float32")
            dst = T.alloc_shared((rows, cols), "float32")
            T.copy(A, src)
            with T.SimtVF(threads=2048):
                for r, c in T.Parallel(rows, cols):
                    first[r, c] = src[r, c] + T.float32(1)
            with T.SimtVF(threads=2048):
                for r, c in T.Parallel(rows, cols):
                    stage[r, c] = first[r, c] * T.float32(2)
            with T.SimtVF(threads=2048):
                for r, c in T.Parallel(rows, cols):
                    dst[r, c] = stage[r, c] + T.float32(3)
            T.copy(dst, B)

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("strategy", ["padding", "swizzle"])
def test_combined_ownership_storage_numerical(strategy):
    import torch

    compiled = tilelang.compile(
        component_kernel(),
        target="ascend",
        out_idx=[1],
        pass_configs={
            STRATEGY: strategy,
            "tl.ascend.ub_layout_ownership": True,
        },
    )
    a = (torch.arange(8192, dtype=torch.float32).reshape(256, 32) % 251 - 125).npu()
    torch.testing.assert_close(compiled(a).cpu(), (a.cpu() + 1) * 2 + 3, rtol=0, atol=0)


@pytest.mark.parametrize("strategy,axis", [("padding", 0), ("padding", 1), ("swizzle", 0)])
@pytest.mark.parametrize("auto_schedule", [True, False])
def test_combined_ownership_storage_full_lowering(strategy, axis, auto_schedule):
    from tilelang import tvm
    from testing.ascend._ir import allocated_buffer

    snapshots = {}

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name in ("tl.AscendLayoutInference", "tl.AscendLowerTileOp"):
                snapshots[info.name] = mod

    config = {
        STRATEGY: strategy,
        "tl.ascend.ub_layout_ownership": True,
        "tl.ascend.ub_layout_padding_axis": axis,
        "tl.enable_auto_schedule": auto_schedule,
    }
    with tvm.target.Target("ascend"), tvm.transform.PassContext(config=config, instruments=[Capture()]):
        lowered = tilelang.lower(component_kernel(), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source
    assert snapshots
    physical = snapshots["tl.AscendLowerTileOp"]
    stage = allocated_buffer(physical, "stage")
    expected = (256, 32) if strategy == "swizzle" else ((8222,) if axis == 0 else (8704,))
    assert tuple(int(x) for x in stage.shape) == expected
    assert tuple(int(x) for x in allocated_buffer(physical, "first").shape) == expected
    if strategy == "swizzle":
        assert "^" in lowered.kernel_source

    from testing.ascend._ir import nodes
    from tvm import tirx

    def layouts(mod):
        return [n.annotations["parallel_loop_layout"] for n in nodes(mod, tirx.For) if "parallel_loop_layout" in n.annotations]

    combined_layouts = layouts(snapshots["tl.AscendLayoutInference"])
    for ownership in (False, True):
        baseline = {STRATEGY: "off", "tl.ascend.ub_layout_ownership": ownership, "tl.enable_auto_schedule": auto_schedule}
        with tvm.target.Target("ascend"), tvm.transform.PassContext(config=baseline, instruments=[Capture()]):
            tilelang.lower(component_kernel(), target="ascend")
        baseline_layouts = layouts(snapshots["tl.AscendLayoutInference"])
        assert len(combined_layouts) == len(baseline_layouts) == 3
        # Storage is applied manually and preserves the selected ownership.
        assert all(a.is_equal(b) == ownership for a, b in zip(combined_layouts, baseline_layouts))


def strided_consumer_kernel(stride, rows=8):
    cols = 32 * stride

    @T.prim_func
    def main(A: T.Tensor((rows, cols), "float32"), B: T.Tensor((rows, 32), "float32")):
        with T.Kernel(1):
            src = T.alloc_shared((rows, cols), "float32")
            stage = T.alloc_shared((rows, cols), "float32")
            dst = T.alloc_shared((rows, 32), "float32")
            T.copy(A, src)
            with T.SimtVF(threads=128):
                for r, c in T.Parallel(rows, cols):
                    stage[r, c] = src[r, c] * T.float32(2) + T.float32(1)
            with T.SimtVF(threads=128):
                for r, c in T.Parallel(rows, 32):
                    dst[r, c] = stage[r, stride * c]
            T.copy(dst, B)

    return main


@pytest.mark.parametrize("stride", [4, 8, 16])
def test_column_padding_fixed_stride_consumer_full_lowering(stride):
    from tilelang import tvm
    from tvm import tirx
    from testing.ascend._ir import allocated_buffer, nodes

    snapshots = []

    @tvm.ir.instrument.pass_instrument
    class Capture:
        def run_after_pass(self, mod, info):
            if info.name == "tl.AscendLayoutInference":
                snapshots.append(mod)

    config = {
        STRATEGY: "padding",
        "tl.ascend.ub_layout_padding_axis": 1,
        "tl.ascend.ub_layout_padding_period": 64,
        "tl.ascend.ub_layout_padding_delta": 2,
    }
    with tvm.target.Target("ascend"), tvm.transform.PassContext(config=config, instruments=[Capture()]):
        lowered = tilelang.lower(strided_consumer_kernel(stride), target="ascend")
    assert "__simt_vf__" in lowered.kernel_source
    after = snapshots[-1]
    pitch = 32 * stride + 2 * ((32 * stride - 1) // 64)
    stage = allocated_buffer(after, "stage")
    assert tuple(int(x) for x in stage.shape) == (8, 32 * stride)
    layout = next(block.annotations["layout_map"][stage] for block in nodes(after, tirx.SBlock) if "layout_map" in block.annotations)
    assert tuple(int(x) for x in layout.get_output_shape()) == (8 * pitch,)
    consumer = next(n for n in nodes(after, tirx.BufferLoad) if str(n.buffer.name) == "stage")
    variables = {v for index in consumer.indices for v in nodes(index, tirx.Var)}
    assert len(variables) == 1
    fused = next(iter(variables))
    loop = next(n for n in nodes(after, tirx.For) if n.loop_var.same_as(fused))
    assert int(loop.extent) == 8 * 32
    analyzer = tvm.arith.Analyzer()
    for r, c in [(0, 0), (0, 31), (1, 16), (7, 31)]:
        actual = analyzer.simplify(tirx.stmt_functor.substitute(layout.map_forward_index(list(consumer.indices))[0], {fused: r * 32 + c}))
        assert int(actual) == r * pitch + stride * c + 2 * ((stride * c) // 64)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("stride", [4, 8, 16])
def test_column_padding_fixed_stride_consumer_numerical(stride):
    import torch

    compiled = tilelang.compile(
        strided_consumer_kernel(stride),
        target="ascend",
        out_idx=[1],
        pass_configs={
            STRATEGY: "padding",
            "tl.ascend.ub_layout_padding_axis": 1,
            "tl.ascend.ub_layout_padding_period": 64,
            "tl.ascend.ub_layout_padding_delta": 2,
        },
    )
    a = (torch.arange(8 * 32 * stride, dtype=torch.float32).reshape(8, 32 * stride) % 251 - 125).npu()
    torch.testing.assert_close(compiled(a).cpu(), a.cpu()[:, ::stride] * 2 + 1, rtol=0, atol=0)
