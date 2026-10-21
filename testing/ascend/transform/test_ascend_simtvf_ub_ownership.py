"""Ascend layout inference: SIMT thread assignment and layout propagation."""

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes, kernel, seq, copy, region


def _infer(mod, **options):
    with mod["main"].attrs["target"], tvm.transform.PassContext(config=options):
        return transform.AscendLayoutInference()(mod)


def _vf(body, threads=128, name="SIMT_VF"):
    tx = tirx.IterVar(tvm.ir.Range(0, threads), tirx.Var("tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    return tirx.SBlockRealize([], True, tirx.SBlock([], [], [], name, tirx.AttrStmt(tx, "thread_extent", threads, body)))


def _buffers(mod):
    return {str(b.name): b for block in nodes(mod, tirx.SBlock) for b in block.alloc_buffers}


def _layouts(mod):
    return [n.annotations["parallel_loop_layout"] for n in nodes(mod, tirx.For) if "parallel_loop_layout" in n.annotations]


def _assert_ownership(layout, n, threads, width):
    analyzer = tvm.arith.Analyzer()
    for i in {0, 1, width - 1, width, threads - 1, threads, n - 1}:
        assert int(analyzer.simplify(layout.map_forward_thread([i])[0])) == (i // width) % threads


def _elementwise(n=8192, threads=2048, *, annotate=False, broadcast=False, dtype="float32", user_layout=False, alignments=(0, 0, 0)):
    a = tirx.decl_buffer((n,), dtype, name="a", scope="shared.dyn", data_alignment=alignments[0])
    b = tirx.decl_buffer((n,), dtype, name="b", scope="shared.dyn", data_alignment=alignments[1])
    c = tirx.decl_buffer((n,), dtype, name="c", scope="shared.dyn", data_alignment=alignments[2])
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


def test_ownership_is_always_enabled():
    mod = _elementwise()
    on = _infer(mod)
    _assert_ownership(_layouts(on)[0], 8192, 2048, 1)


@pytest.mark.parametrize("buffer_index", [0, 2], ids=["load", "store"])
@pytest.mark.parametrize("alignment", [1, 2, 4, 8, 32, 64])
def test_ownership_requires_known_base_alignment(buffer_index, alignment):
    alignments = [32, 32, 32]
    alignments[buffer_index] = alignment
    mod = _elementwise(alignments=alignments)
    # Weak alignment on even one UB access retains the native width=2 plan.
    # An explicitly guaranteed 8B base is sufficient; 32B is not required.
    _assert_ownership(_layouts(_infer(mod))[0], 8192, 2048, 1 if alignment >= 8 else 2)


@pytest.mark.parametrize("options", [{"annotate": True}, {"user_layout": True}])
def test_ownership_preserves_constraints(options):
    mod = _elementwise(**options)
    _assert_ownership(_layouts(_infer(mod))[0], 8192, 2048, 2)


@pytest.mark.parametrize(
    "dtype,width", [("int8", 4), ("float16", 2), ("bfloat16", 2), ("int16", 2), ("float32", 1), ("int32", 1), ("int64", 1)]
)
def test_ownership_scalar_types(dtype, width):
    mod = _elementwise(threads=128, dtype=dtype)
    on = _infer(mod)
    analyzer = tvm.arith.Analyzer()
    for i in range(256):
        assert int(analyzer.simplify(_layouts(on)[0].map_forward_thread([i])[0])) == (i // width) % 128
    with on["main"].attrs["target"]:
        transform.AscendLowerTileOp()(on)


def _executed_store_indices(mod, name):
    """Enumerate actual lowered control flow, including padded-lane guards."""
    analyzer = tvm.arith.Analyzer()
    found = []

    def value(expr, bindings):
        return int(analyzer.simplify(tirx.stmt_functor.substitute(expr, bindings)))

    def visit(stmt, bindings):
        if isinstance(stmt, tirx.SBlockRealize):
            if value(stmt.predicate, bindings):
                visit(stmt.block, bindings)
        elif isinstance(stmt, tirx.SBlock):
            visit(stmt.body, bindings)
        elif isinstance(stmt, tirx.AttrStmt):
            if stmt.attr_key == "thread_extent":
                for thread in range(value(stmt.value, bindings)):
                    visit(stmt.body, {**bindings, stmt.node.var: thread})
            else:
                visit(stmt.body, bindings)
        elif isinstance(stmt, tirx.For):
            start = value(stmt.min, bindings)
            for i in range(start, start + value(stmt.extent, bindings)):
                visit(stmt.body, {**bindings, stmt.loop_var: i})
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child, bindings)
        elif isinstance(stmt, tirx.IfThenElse):
            child = stmt.then_case if value(stmt.condition, bindings) else stmt.else_case
            if child is not None:
                visit(child, bindings)
        elif isinstance(stmt, tirx.BufferStore) and str(stmt.buffer.name) == name:
            found.append(tuple(value(index, bindings) for index in stmt.indices))

    visit(mod["main"].body, {})
    return found


@pytest.mark.parametrize("n,threads", [(258, 128), (8191, 2048), (256, 16), (130, 33)])
def test_ownership_partial_warps_and_tail_guards(n, threads):
    on = _infer(_elementwise(n=n, threads=threads))
    with on["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(on)
    # Every logical element executes once; inactive lanes produce no stores.
    assert sorted(_executed_store_indices(lowered, "c")) == [(i,) for i in range(n)]


@pytest.mark.parametrize("mode", ["broadcast", "offset", "stride", "transpose", "fill", "gm", "parameter", "predicate", "inplace"])
def test_ownership_access_maps(mode):
    n, threads = 256, 32
    a = tirx.decl_buffer((512,), "float32", name="a", scope="shared.dyn")
    shape = (16, 16) if mode == "transpose" else (512,)
    c = tirx.decl_buffer(shape, "float32", name="c", scope="global" if mode == "gm" else "shared.dyn")
    i = tirx.Var("i", "int32")
    scale = tirx.Var("scale", "float32")
    index = i * 2 if mode == "stride" else i + 16 if mode == "offset" else i
    indices = [i % 16, i // 16] if mode == "transpose" else [index]
    expr = tirx.FloatImm("float32", 1) if mode == "fill" else a[0 if mode == "broadcast" else index] + 1
    if mode == "parameter":
        expr = expr * scale
    if mode == "inplace":
        expr = c[index] + 1
    body = tirx.BufferStore(c, expr, indices)
    if mode == "predicate":
        body = tirx.IfThenElse(i < 250, body, None)
    params = [c] if mode == "gm" else [scale] if mode == "parameter" else []
    mod = kernel(_vf(tirx.For(i, 0, n, tirx.ForKind.PARALLEL, body), threads), buffers=[a] + ([] if mode == "gm" else [c]), params=params)
    on = _infer(mod)
    with on["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(on)
    if mode != "parameter":
        expected = [
            (x % 16, x // 16) if mode == "transpose" else (2 * x if mode == "stride" else x + 16 if mode == "offset" else x,)
            for x in range(250 if mode == "predicate" else n)
        ]
        assert sorted(_executed_store_indices(lowered, "c")) == sorted(expected)


@pytest.mark.parametrize("dependent", [False, True])
def test_ownership_does_not_change_conflicting_accesses(dependent):
    a = tirx.decl_buffer((512,), "float32", name="a", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    # Either overlapping stores or a loop-carried read/write dependence.
    body = tirx.BufferStore(a, a[i + 1] if dependent else tirx.Cast("float32", i), [i if dependent else i % 128])
    mod = kernel(_vf(tirx.For(i, 0, 256, tirx.ForKind.PARALLEL, body), 32), buffers=[a])
    _assert_ownership(_layouts(_infer(mod))[0], 256, 32, 1 if dependent else 2)


def test_pto_infers_ownership():
    mod = _elementwise()
    target = {"kind": "ascend", "keys": ["pto"]}
    mod = tvm.IRModule({"main": mod["main"].with_attr("target", tvm.target.Target(target))})
    _assert_ownership(_layouts(_infer(mod))[0], 8192, 2048, 1)


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
    on = _infer(mod)
    _assert_ownership(_layouts(on)[0], 8192, 2048, 2)
    _assert_ownership(_layouts(on)[-1], 8192, 2048, 1)


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
    _assert_ownership(_layouts(_infer(mod))[0], 8192, 2048, 2)


def test_ownership_thread_dependent_value_is_unchanged():
    mod = _elementwise()
    body = mod["main"].body
    tx = nodes(mod, tirx.AttrStmt)[0].node.var

    def add_thread(node):
        if isinstance(node, tirx.BufferStore):
            return tirx.BufferStore(node.buffer, node.value + tirx.Cast("float32", tx), node.indices)

    body = tirx.stmt_functor.ir_transform(body, None, add_thread)
    mod = tvm.IRModule({"main": mod["main"].with_body(body)})
    _assert_ownership(_layouts(_infer(mod))[0], 8192, 2048, 2)


def _nest(rows, cols, body):
    i, j = tirx.Var("i", "int32"), tirx.Var("j", "int32")
    return tirx.For(i, 0, rows, tirx.ForKind.PARALLEL, tirx.For(j, 0, cols, tirx.ForKind.PARALLEL, body(i, j)))


def _simt_copy_component(direction, copies, *, explicit=False, alignment=32):
    strided = tirx.decl_buffer((32, 32), "float16", name="strided", scope="shared.dyn", strides=(1, 32))
    buffers = [strided]
    fragments = []
    transfers = []
    for k in range(copies):
        ub = tirx.decl_buffer((32, 32), "float16", name=f"ub{k}", scope="shared.dyn", data_alignment=alignment)
        if direction == "load" or not fragments:
            fragments.append(tirx.decl_buffer((32, 32), "float16", name=f"fragment{k}", scope="local.fragment"))
        src, dst = (ub, fragments[-1]) if direction == "load" else (fragments[0], ub)
        transfers.append(
            _nest(32, 32, lambda i, j, src=src, dst=dst: tirx.BufferStore(dst, src[i, j], [i, j])) if explicit else copy(src, dst)
        )
        buffers.append(ub)
    if direction == "load":
        compute = _nest(32, 32, lambda i, j: tirx.BufferStore(strided, sum(f[i, j] for f in fragments), [i, j]))
        body = seq(*transfers, compute)
    else:
        compute = _nest(32, 32, lambda i, j: tirx.BufferStore(fragments[0], strided[i, j] + 1, [i, j]))
        body = seq(compute, *transfers)
    return kernel(_vf(body, 128), buffers=[*buffers, *fragments])


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("copies,width", [(1, 8), (4, 4)])
def test_ownership_scores_simt_copies(direction, copies, width):
    # Four contiguous transfers outweigh the strided compute's preference for
    # width=8. A single transfer still permits that ownership optimization.
    actual = _infer(_simt_copy_component(direction, copies))
    explicit = _infer(_simt_copy_component(direction, copies, explicit=True))
    actual_layout, explicit_layout = _layouts(actual)[0], _layouts(explicit)[0]
    analyzer = tvm.arith.Analyzer()
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    assert analyzer.can_prove_equal(actual_layout.map_forward_thread([i, j])[0], explicit_layout.map_forward_thread([i, j])[0])
    for index in [0, 1, 3, 4, 7, 8, 127, 128, 511, 1023]:
        assert int(analyzer.simplify(actual_layout.map_forward_thread([index // 32, index % 32])[0])) == (index // width) % 128
    with actual["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(actual)
    # Check the executed copy stores as well as the explicit compute loop.
    outputs = ["strided"] if direction == "load" else [f"ub{k}" for k in range(copies)]
    for name in outputs:
        assert sorted(_executed_store_indices(lowered, name)) == [(r, c) for r in range(32) for c in range(32)]


@pytest.mark.parametrize("direction", ["load", "store"])
def test_ownership_unknown_copy_alignment_keeps_native(direction):
    # Only the copy sees the weakly aligned UB. Its unknown cost must prevent
    # the otherwise profitable width=8 change in the explicit compute loop.
    actual = _infer(_simt_copy_component(direction, 1, alignment=1))
    explicit = _infer(_simt_copy_component(direction, 1, explicit=True, alignment=1))
    i, j = tirx.Var("row", "int32"), tirx.Var("col", "int32")
    assert tvm.arith.Analyzer().can_prove_equal(
        _layouts(actual)[0].map_forward_thread([i, j])[0], _layouts(explicit)[0].map_forward_thread([i, j])[0]
    )


@pytest.mark.parametrize("output_scope", ["global", "local.fragment"])
def test_ownership_scores_ub_loads_in_branch_conditions(output_scope):
    a = tirx.decl_buffer((32, 32), "float16", name="a", scope="shared.dyn", strides=(1, 32))
    c = tirx.decl_buffer((32, 32), "float16", name="c", scope="shared.dyn", strides=(1, 32))
    flags = tirx.decl_buffer((1024,), "int8", name="flags", scope="shared.dyn", data_alignment=1)
    out = tirx.decl_buffer((1024,), "int32", name="out", scope=output_scope)
    body = _nest(
        32,
        32,
        lambda i, j: seq(
            tirx.BufferStore(c, a[i, j] + tirx.const(1, "float16"), [i, j]),
            tirx.IfThenElse(
                flags[i * 32 + j] > 0,
                tirx.BufferStore(out, tirx.const(1, "int32"), [i * 32 + j]),
                tirx.BufferStore(out, tirx.const(0, "int32"), [i * 32 + j]),
            ),
        ),
    )
    mod = kernel(
        _vf(body, 128),
        buffers=[a, c, flags] + ([out] if output_scope == "local.fragment" else []),
        params=[out] if output_scope == "global" else [],
    )
    on = _infer(mod)
    layout = _layouts(on)[0]
    analyzer = tvm.arith.Analyzer()
    # The branches contain no UB accesses. Omitting the condition would score
    # only a/c and choose width=8, ignoring flags' unknown base low bits.
    # Visiting flags retains the native width=1 plan for the whole loop.
    for i in range(32):
        for j in range(32):
            indices = [i, j] if output_scope == "local.fragment" else [i * 32 + j]
            assert int(analyzer.simplify(layout.map_forward_thread(indices)[0])) == (i * 32 + j) % 128
    with on["main"].attrs["target"]:
        transform.AscendLowerTileOp()(on)


def _strided_ownership(shape, strides, dtype="float16", elem_offset=0):
    a = tirx.decl_buffer(shape, dtype, name="a", scope="shared.dyn", strides=strides, elem_offset=elem_offset)
    c = tirx.decl_buffer(shape, dtype, name="c", scope="shared.dyn", strides=strides, elem_offset=elem_offset)
    body = _nest(*shape, lambda i, j: tirx.BufferStore(c, a[i, j] + tirx.const(1, dtype), [i, j]))
    return kernel(_vf(body, 128), buffers=[a, c])


@pytest.mark.parametrize("strides", [(32, 1), (256, 1), (64, 2), (256, 2)])
@pytest.mark.parametrize("dtype", ["int8", "float16", "float32", "int64"])
def test_ownership_size_stride_dtype(strides, dtype):
    # All four patterns use the same physical-address analysis: compact,
    # blocks separated by gaps, strided elements, and strided blocks.
    mod = _strided_ownership((32, 32), strides, dtype)
    on = _infer(mod)
    with on["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(on)
    assert sorted(_executed_store_indices(lowered, "c")) == [(i, j) for i in range(32) for j in range(32)]


@pytest.mark.parametrize(
    "key,value",
    [("tl.ascend.ub_layout_ownership", False), ("tl.ascend.ub_layout_ownership", True), ("tl.ascend.ub_layout_ownership_width", 2)],
)
def test_ownership_has_no_configuration_switch(key, value):
    with pytest.raises(AttributeError, match="Invalid config option.*ub_layout_ownership"):
        _infer(_elementwise(), **{key: value})


def test_ownership_strided_group_can_exceed_vector_width():
    # The physical fast axis differs from logical traversal. No two adjacent
    # logical values are contiguous, but grouping eight in a thread exposes
    # contiguous addresses across lanes. This must not be capped at the
    # vectorizer's native width=1.
    mod = _strided_ownership((32, 32), (1, 32))
    on = _infer(mod)
    analyzer = tvm.arith.Analyzer()
    for i in range(1024):
        assert int(analyzer.simplify(_layouts(on)[0].map_forward_thread([i])[0])) == (i // 8) % 128
    with on["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(on)
    assert not any(loop.kind == tirx.ForKind.VECTORIZED for loop in nodes(lowered, tirx.For))
    assert sorted(_executed_store_indices(lowered, "c")) == [(i, j) for i in range(32) for j in range(32)]


@pytest.mark.parametrize("offset", [0, 1, 4, 256])
def test_ownership_block_access_eliminates_cross_rows(offset):
    mod = _strided_ownership((32, 32), (256, 1), elem_offset=offset)
    on = _infer(mod)
    analyzer = tvm.arith.Analyzer()
    # With width=1 each warp accesses a single 64B row. Width=2 packs f16,
    # but mixes rows 512B apart in each request. Their total entry counts
    # tie, so the policy chooses the candidate without same-key cross rows.
    for i in range(1024):
        assert int(analyzer.simplify(_layouts(on)[0].map_forward_thread([i])[0])) == i % 128


def test_ownership_periodic_large_byte_tile():
    # Three byte buffers still fit in UB. The request analysis must fold
    # proven repetition, not reject the loop above a 65536-point budget.
    n = 73728
    on = _infer(_elementwise(n=n, threads=128, dtype="int8"))
    analyzer = tvm.arith.Analyzer()
    for index in (0, 1, 3, 4, 511, 512, n - 1):
        assert int(analyzer.simplify(_layouts(on)[0].map_forward_thread([index])[0])) == (index // 4) % 128


@pytest.mark.parametrize("transpose", [False, True])
def test_ownership_candidate_propagates_through_fragments(transpose):
    a = tirx.decl_buffer((32, 32), "float16", name="a", scope="shared.dyn")
    f = tirx.decl_buffer((32, 32), "float16", name="f", scope="local.fragment")
    c = tirx.decl_buffer((32, 32), "float16", name="c", scope="shared.dyn")
    producer = _nest(32, 32, lambda i, j: tirx.BufferStore(f, a[i, j], [i, j]))
    consumer = _nest(32, 32, lambda i, j: tirx.BufferStore(c, f[j, i] if transpose else f[i, j], [i, j]))
    on = _infer(kernel(_vf(seq(producer, consumer)), buffers=[a, f, c]))
    layouts = _layouts(on)
    assert len(layouts) == 2
    analyzer = tvm.arith.Analyzer()
    for i in range(32):
        for j in range(32):
            producer_thread = layouts[0].map_forward_thread([j, i] if transpose else [i, j])[0]
            consumer_thread = layouts[1].map_forward_thread([i, j])[0]
            assert analyzer.can_prove_equal(producer_thread, consumer_thread)
    with on["main"].attrs["target"]:
        lowered = transform.AscendLowerTileOp()(on)
    assert sorted(_executed_store_indices(lowered, "c")) == [(i, j) for i in range(32) for j in range(32)]
