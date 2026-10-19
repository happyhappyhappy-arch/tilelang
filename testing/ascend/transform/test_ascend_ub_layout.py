"""Manual UB strategies at the layout-inference boundary, without a device."""

import operator
import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes, kernel, seq, copy, region

KEY = "tl.ascend.ub_layout_strategy"
OWNERSHIP = "tl.ascend.ub_layout_ownership"


def _infer(mod, strategy="off", **options):
    with mod["main"].attrs["target"], tvm.transform.PassContext(config={KEY: strategy, **options}):
        return transform.AscendLayoutInference()(mod)


def _vf(body, threads=128, name="SIMT_VF"):
    tx = tirx.IterVar(tvm.ir.Range(0, threads), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    return tirx.SBlockRealize([], True, tirx.SBlock([], [], [], name, tirx.AttrStmt(tx, "thread_extent", threads, body)))


def _nest(rows, cols, body):
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    return tirx.For(i, 0, rows, tirx.ForKind.PARALLEL, tirx.For(j, 0, cols, tirx.ForKind.PARALLEL, body(i, j)))


def _storage(rows=32, cols=32, *, guard=None, dtype="float32", two=False):
    src = tirx.decl_buffer((rows, cols), dtype, name="src", scope="shared.dyn")
    stage = tirx.decl_buffer((rows, cols), dtype, name="stage", scope="shared.dyn")
    dst = tirx.decl_buffer((cols, rows), dtype, name="dst", scope="shared.dyn")
    gm = tirx.decl_buffer((rows, cols), dtype, name="gm")
    out = tirx.decl_buffer((cols, rows), dtype, name="out")
    stages = [stage]
    if two:
        stages.append(tirx.decl_buffer((rows, cols), dtype, name="stage2", scope="shared.dyn"))
    statements = [copy(gm, src)]
    for k, current in enumerate(stages):
        previous = src if k == 0 else stages[k - 1]
        statements.append(
            _vf(_nest(rows, cols, lambda i, j, current=current, previous=previous: tirx.BufferStore(current, previous[i, j] + 1, [i, j])))
        )
    statements.append(_vf(_nest(cols, rows, lambda i, j: tirx.BufferStore(dst, stages[-1][j, i], [i, j]))))
    annotations = {}
    if guard == "layout":
        annotations["layout_map"] = {stage.data: tilelang.layout.Layout((rows, cols), lambda i, j: i * cols + j)}
    elif guard == "dma":
        statements.append(copy(stage, gm))
    elif guard == "pointer":
        statements.append(tirx.Evaluate(tirx.call_extern("int32", "escape", stage.data)))
    elif guard == "annotation":
        annotations["private_metadata"] = {"buffer": stage}
    elif guard == "alias":
        view = tirx.decl_buffer((rows * cols,), dtype, data=stage.data, name="view", scope="shared.dyn")
        statements.extend([tirx.DeclBuffer(view), tirx.Evaluate(view[0])])
    elif guard == "outside":
        statements.append(tirx.Evaluate(stage[0, 0]))
    statements.append(copy(dst, out))
    return kernel(seq(*statements), buffers=[src, *stages, dst], params=[gm, out], annotations=annotations)


def _constant(expr):
    # TVM's arithmetic simplifier does not constant-fold every bitwise call.
    # Interpret the actual rewritten integer expression for exhaustive checks.
    if isinstance(expr, tirx.IntImm):
        return int(expr)
    if isinstance(expr, tirx.Call):
        op = {"tirx.bitwise_xor": operator.xor, "tirx.bitwise_and": operator.and_}[expr.op.name]
        return op(*[_constant(x) for x in expr.args])
    op = {
        tirx.Add: operator.add,
        tirx.Sub: operator.sub,
        tirx.Mul: operator.mul,
        tirx.FloorDiv: operator.floordiv,
        tirx.FloorMod: operator.mod,
    }[type(expr)]
    return op(_constant(expr.a), _constant(expr.b))


def _buffers(mod):
    return {str(b.name): b for block in nodes(mod, tirx.SBlock) for b in block.alloc_buffers}


def _layouts(mod):
    return [n.annotations["parallel_loop_layout"] for n in nodes(mod, tirx.For) if "parallel_loop_layout" in n.annotations]


def _storage_layout(mod, name="stage"):
    return next(
        layout
        for block in nodes(mod, tirx.SBlock)
        for buffer, layout in block.annotations.get("layout_map", {}).items()
        if str(buffer.name) == name
    )


def _physical_indices(mod, access):
    return _storage_layout(mod, str(access.buffer.name)).map_forward_index(list(access.indices))


@pytest.mark.parametrize("strategy", ["padding", "swizzle"])
@pytest.mark.parametrize("shape", [(32, 32), (64, 64), (32, 16), (32, 4)])
def test_storage_all_uses_and_logical_domain(strategy, shape):
    before = _storage(*shape, two=True)
    off, after = _infer(before), _infer(before, strategy)
    old, new = _buffers(off), _buffers(after)
    for name in ("src", "dst"):
        assert new[name].same_as(old[name])  # DMA endpoints
    for name in ("stage", "stage2"):
        expected = (shape[0] * shape[1] + 2 * ((shape[0] - 1) // 16),) if strategy == "padding" else shape
        assert new[name].same_as(old[name])
        assert tuple(int(x) for x in _storage_layout(after, name).get_output_shape()) == expected
        for node in nodes(after, tirx.BufferLoad) + nodes(after, tirx.BufferStore):
            if str(node.buffer.name) == name:
                assert node.buffer.same_as(new[name])
    for a, b in zip(_layouts(off), _layouts(after)):
        assert a.is_equal(b)
    assert len(_layouts(off)) == len(_layouts(after)) == 3
    for a, b in zip(nodes(off, tirx.For), nodes(after, tirx.For)):
        tvm.ir.assert_structural_equal(a.extent, b.extent)
    # Evaluate the selected address maps across every logical element.
    a = tvm.arith.Analyzer()
    for name in ("stage", "stage2"):
        for old_access, new_access in zip(
            [n for n in nodes(off, tirx.BufferLoad) + nodes(off, tirx.BufferStore) if str(n.buffer.name) == name],
            [n for n in nodes(after, tirx.BufferLoad) + nodes(after, tirx.BufferStore) if str(n.buffer.name) == name],
        ):
            variables = {v for expr in old_access.indices for v in nodes(expr, tirx.Var)}
            assert len(variables) == 1  # fused parallel domain
            var = next(iter(variables))
            new_var = next(iter({v for e in new_access.indices for v in nodes(e, tirx.Var)}))
            tvm.ir.assert_structural_equal(old_access.indices, new_access.indices, map_free_vars=True)
            physical_indices = _physical_indices(after, new_access)
            seen = set()
            for index in range(shape[0] * shape[1]):
                r, c = [int(a.simplify(tirx.stmt_functor.substitute(e, {var: index}))) for e in old_access.indices]
                mapped = tuple(_constant(a.simplify(tirx.stmt_functor.substitute(e, {new_var: index}))) for e in physical_indices)
                expected = (
                    (r * shape[1] + c + 2 * (r // 16),)
                    if strategy == "padding"
                    else (r, 2 * ((c // 2) ^ ((r ^ (r // 16)) & (min(shape[1] // 2, 16) - 1))) + c % 2)
                )
                assert mapped == expected
                seen.add(mapped)
            assert len(seen) == shape[0] * shape[1]


@pytest.mark.parametrize("strategy", ["padding", "swizzle"])
@pytest.mark.parametrize("guard", ["layout", "dma", "pointer", "annotation", "alias", "outside"])
def test_storage_protects_unsupported_uses(strategy, guard):
    before = _storage(guard=guard)
    off, after = _infer(before), _infer(before, strategy)
    tvm.ir.assert_structural_equal(off, after)


@pytest.mark.parametrize(
    "strategy,shape,dtype",
    [("swizzle", (8, 12), "float32"), ("swizzle", (8, 2), "float32"), ("padding", (8, 31), "float32"), ("padding", (8, 32), "float16")],
)
def test_storage_rejects_unsupported_geometry(strategy, shape, dtype):
    mod = _storage(*shape, dtype=dtype)
    tvm.ir.assert_structural_equal(_infer(mod), _infer(mod, strategy))


@pytest.mark.parametrize(
    "config",
    [
        {KEY: "ownership"},
        {KEY: "pair_xor"},
        {KEY: "unknown"},
        {"tl.ascend.ub_layout_ownership_width": 0},
        {"tl.ascend.ub_layout_ownership_width": 3},
        {"tl.ascend.ub_layout_padding_delta": -2},
        {"tl.ascend.ub_layout_padding_delta": 0},
        {"tl.ascend.ub_layout_padding_period": 0},
        {"tl.ascend.ub_layout_padding_period": 3},
        {"tl.ascend.ub_layout_padding_axis": 2},
    ],
)
def test_invalid_configuration(config):
    with pytest.raises(ValueError, match="UB|ub_layout_strategy"):
        _infer(_storage(), **config)


def test_off_matches_unconfigured():
    before = _storage()
    with before["main"].attrs["target"]:
        tvm.ir.assert_structural_equal(_infer(before), transform.AscendLayoutInference()(before))


def _elementwise(n=8192, threads=2048, *, annotate=False, broadcast=False, dtype="float32", user_layout=False):
    a = tirx.decl_buffer((n,), dtype, name="a", scope="shared.dyn")
    b = tirx.decl_buffer((n,), dtype, name="b", scope="shared.dyn")
    c = tirx.decl_buffer((n,), dtype, name="c", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    loop = tirx.For(
        i,
        0,
        n,
        tirx.ForKind.PARALLEL,
        tirx.BufferStore(c, a[0 if broadcast else i] + b[i], [i]),
        annotations={"coalesced_width": tirx.IntImm("int32", 2)} if annotate else {},
    )
    annotations = {"layout_map": {a.data: tilelang.layout.Layout((n,), lambda i: i)}} if user_layout else {}
    return kernel(_vf(loop, threads), buffers=[a, b, c], annotations=annotations)


def test_ownership_width_changes_partition():
    mod = _elementwise()
    off, on = _infer(mod), _infer(mod, **{OWNERSHIP: True})
    assert not _layouts(off)[0].is_equal(_layouts(on)[0])
    assert _layouts(_infer(mod, **{OWNERSHIP: True, "tl.ascend.ub_layout_ownership_width": 2}))[0].is_equal(_layouts(off)[0])


@pytest.mark.parametrize(
    "options", [{"annotate": True}, {"user_layout": True}, {"broadcast": True}, {"dtype": "float16"}, {"n": 8191}, {"threads": 33}]
)
def test_ownership_preserves_constraints(options):
    mod = _elementwise(**options)
    tvm.ir.assert_structural_equal(_infer(mod), _infer(mod, **{OWNERSHIP: True}))


@pytest.mark.parametrize(
    "target", [{"kind": "ascend", "keys": ["pto"]}, {"kind": "ascend", "arch": "dav-2201"}, {"kind": "ascend", "mcpu": "dav-2201"}]
)
@pytest.mark.parametrize("strategy", ["padding", "swizzle", "ownership"])
def test_non_plain_or_other_arch_is_unchanged(target, strategy):
    mod = _elementwise() if strategy == "ownership" else _storage()
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target(target))})
    tvm.ir.assert_structural_equal(
        _infer(mod), _infer(mod, "off" if strategy == "ownership" else strategy, **{OWNERSHIP: strategy == "ownership"})
    )


def test_padding_overflow_keeps_storage():
    mod = _storage()
    off = _infer(mod)
    after = _infer(mod, "padding", **{"tl.ascend.ub_layout_padding_delta": 2147483646})
    tvm.ir.assert_structural_equal(off, after)


def test_ownership_collective_boundary_and_independent_component():
    # The first add shares UB with a collective through a fragment copy. The
    # independent add remains eligible; UB edges must not swallow the function.
    left, right = _elementwise(), _elementwise()
    a = _buffers(left)
    values = tirx.decl_buffer((8192,), "float32", name="values", scope="local.fragment")
    sums = tirx.decl_buffer((1,), "float32", name="sums", scope="local.fragment")
    collective = _vf(
        seq(
            copy(a["c"], values),
            tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [region(values, 1), region(sums, 2), "sum", 0, True])),
        ),
        threads=2048,
    )
    left_body = left["main"].body.block.body
    right_body = right["main"].body.block.body
    mod = kernel(seq(left_body, collective, right_body), buffers=[*a.values(), values, sums, *_buffers(right).values()])
    off, on = _infer(mod), _infer(mod, **{OWNERSHIP: True})
    assert _layouts(off)[0].is_equal(_layouts(on)[0])
    assert not _layouts(off)[-1].is_equal(_layouts(on)[-1])


def test_ownership_unknown_access_protects_component():
    mod = _elementwise()
    root = mod["main"].body.block
    a = _buffers(mod)["a"]
    root = tirx.SBlock(
        [],
        [],
        [],
        "root",
        seq(root.body, tirx.Evaluate(tirx.call_extern("int32", "escape", a.data))),
        alloc_buffers=list(_buffers(mod).values()),
    )
    mod = tvm.IRModule({"main": mod["main"].with_body(tirx.SBlockRealize([], True, root))})
    tvm.ir.assert_structural_equal(_infer(mod), _infer(mod, **{OWNERSHIP: True}))


def test_padding_preserves_pair_alignment():
    mod = _storage()
    tvm.ir.assert_structural_equal(_infer(mod), _infer(mod, "padding", **{"tl.ascend.ub_layout_padding_delta": 1}))


def test_ownership_thread_dependent_value_is_unchanged():
    mod = _elementwise()
    body = mod["main"].body
    tx = nodes(mod, tirx.AttrStmt)[0].node.var

    def add_thread(node):
        if isinstance(node, tirx.BufferStore):
            return tirx.BufferStore(node.buffer, node.value + tirx.Cast("float32", tx), node.indices)

    body = tirx.stmt_functor.ir_transform(body, None, add_thread)
    mod = tvm.IRModule({"main": mod["main"].with_body(body)})
    tvm.ir.assert_structural_equal(_infer(mod), _infer(mod, **{OWNERSHIP: True}))


@pytest.mark.parametrize("strategy", ["padding", "swizzle"])
def test_storage_regions_follow_physical_buffer(strategy):
    mod = _storage()
    stage = _buffers(mod)["stage"]

    def attach_region(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "SIMT_VF":
            region = tirx.BufferRegion(stage, [tvm.ir.Range(0, 32), tvm.ir.Range(0, 32)])
            return tirx.SBlock(
                node.iter_vars,
                [region],
                [region],
                node.name_hint,
                node.body,
                alloc_buffers=node.alloc_buffers,
                annotations=node.annotations,
            )

    mod = tvm.IRModule({"main": mod["main"].with_body(tirx.stmt_functor.ir_transform(mod["main"].body, None, attach_region))})
    after = _infer(mod, strategy)
    # Inference keeps logical regions; lowering materializes both the Buffer
    # and a conservative cover of its mapped regions.
    assert _buffers(after)["stage"].same_as(stage)
    with after["main"].attrs["target"]:
        after = transform.AscendLowerTileOp()(after)
    physical = _buffers(after)["stage"]
    assert not physical.same_as(stage)
    for access in nodes(after, tirx.BufferLoad) + nodes(after, tirx.BufferStore):
        if str(access.buffer.name) == "stage":
            assert access.buffer.same_as(physical)
    for block in nodes(after, tirx.SBlock):
        for buffer_region in list(block.reads) + list(block.writes):
            if str(buffer_region.buffer.name) == "stage":
                assert buffer_region.buffer.same_as(physical)
                assert len(buffer_region.region) == len(physical.shape)
                for extent, bound in zip(physical.shape, buffer_region.region):
                    assert int(bound.min) == 0
                    assert int(bound.extent) == int(extent)


@pytest.mark.parametrize("strategy", ["padding", "swizzle"])
def test_storage_simd_use_is_unchanged(strategy):
    mod = _storage()

    def simd(node):
        if isinstance(node, tirx.SBlock) and node.name_hint == "SIMT_VF":
            return tirx.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                "SIMD_VF",
                node.body,
                alloc_buffers=node.alloc_buffers,
                annotations=node.annotations,
            )

    mod = tvm.IRModule({"main": mod["main"].with_body(tirx.stmt_functor.ir_transform(mod["main"].body, None, simd))})
    tvm.ir.assert_structural_equal(_infer(mod), _infer(mod, strategy))


@pytest.mark.parametrize("axis,period,delta", [(0, 4, 6), (0, 1, 2), (1, 4, 6), (1, 16, 2)])
def test_padding_configured_mapping(axis, period, delta):
    rows, cols = 32, 32
    mod = _storage(rows, cols)
    off = _infer(mod)
    after = _infer(
        mod,
        "padding",
        **{
            "tl.ascend.ub_layout_padding_axis": axis,
            "tl.ascend.ub_layout_padding_period": period,
            "tl.ascend.ub_layout_padding_delta": delta,
        },
    )
    pitch = cols + delta * ((cols - 1) // period)
    span = rows * cols + delta * ((rows - 1) // period) if axis == 0 else rows * pitch
    assert tuple(int(x) for x in _storage_layout(after).get_output_shape()) == (span,)
    analyzer = tvm.arith.Analyzer()
    accesses = []
    for result in (off, after):
        accesses.append([n for n in nodes(result, tirx.BufferLoad) + nodes(result, tirx.BufferStore) if str(n.buffer.name) == "stage"])
    images = []
    for old, new in zip(*accesses):
        ov = next(iter({v for e in old.indices for v in nodes(e, tirx.Var)}))
        nv = next(iter({v for e in new.indices for v in nodes(e, tirx.Var)}))
        image = {}
        for i in range(rows * cols):
            r, c = [_constant(analyzer.simplify(tirx.stmt_functor.substitute(e, {ov: i}))) for e in old.indices]
            physical = _constant(analyzer.simplify(tirx.stmt_functor.substitute(_physical_indices(after, new)[0], {nv: i})))
            expected = r * cols + c + delta * (r // period) if axis == 0 else r * pitch + c + delta * (c // period)
            assert physical == expected
            image[r, c] = physical
        assert len(set(image.values())) == rows * cols
        assert max(image.values()) == span - 1
        images.append(image)
    assert all(image == images[0] for image in images)  # producer/consumer round trip


@pytest.mark.parametrize("axis,period,delta", [(0, 32, 2), (1, 32, 2), (1, 1, 2)])
def test_padding_identity_and_unaligned_period_keep_storage(axis, period, delta):
    mod = _storage()
    tvm.ir.assert_structural_equal(
        _infer(mod),
        _infer(
            mod,
            "padding",
            **{
                "tl.ascend.ub_layout_padding_axis": axis,
                "tl.ascend.ub_layout_padding_period": period,
                "tl.ascend.ub_layout_padding_delta": delta,
            },
        ),
    )


def _elementwise_2d():
    rows, cols = 256, 32
    buffers = [tirx.decl_buffer((rows, cols), "float32", name=name, scope="shared.dyn") for name in ("a", "b", "stage", "dst")]
    a, b, stage, dst = buffers
    return kernel(
        seq(
            _vf(_nest(rows, cols, lambda r, c: tirx.BufferStore(stage, a[r, c] + b[r, c], [r, c])), 2048),
            _vf(_nest(rows, cols, lambda r, c: tirx.BufferStore(dst, stage[r, c] * 2, [r, c])), 2048),
        ),
        buffers=buffers,
    )


@pytest.mark.parametrize("strategy", ["padding", "swizzle"])
def test_ownership_composes_with_storage_for_whole_component(strategy):
    mod = _elementwise_2d()
    off = _infer(mod)
    ownership = _infer(mod, **{OWNERSHIP: True})
    storage = _infer(mod, strategy)
    combined = _infer(mod, strategy, **{OWNERSHIP: True})
    assert len(_layouts(off)) == len(_layouts(combined)) == 2
    for normal, owner, stored, both in zip(_layouts(off), _layouts(ownership), _layouts(storage), _layouts(combined)):
        assert not normal.is_equal(owner)
        assert normal.is_equal(stored)
        assert owner.is_equal(both)
    # Manual selection applies even to pointwise accesses without a modeled win.
    for result in (storage, combined):
        physical = _buffers(result)["stage"]
        assert physical.same_as(_buffers(off)["stage"])
        expected = (256 * 32 + 2 * 15,) if strategy == "padding" else (256, 32)
        assert tuple(int(x) for x in _storage_layout(result).get_output_shape()) == expected


def test_swizzle_distributes_stride16_rows_across_virtual_groups():
    mod = _infer(_storage(512, 32), "swizzle")
    store = next(n for n in nodes(mod, tirx.BufferStore) if str(n.buffer.name) == "stage")
    var = next(iter({v for e in store.indices for v in nodes(e, tirx.Var)}))
    analyzer = tvm.arith.Analyzer()
    # Rows 16 * lane previously all selected the same virtual group.
    groups = []
    for lane in range(32):
        row = 16 * lane
        col = _constant(analyzer.simplify(tirx.stmt_functor.substitute(_physical_indices(mod, store)[1], {var: row * 32})))
        groups.append(col // 2)
    assert len(set(groups)) == 16
    assert all(groups.count(group) == 2 for group in set(groups))
