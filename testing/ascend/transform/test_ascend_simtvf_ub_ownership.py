"""Ascend layout inference: SIMT thread assignment and layout propagation."""

from itertools import product

import pytest
import tilelang
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import nodes, kernel, seq, copy, region


def _buffer(name, shape=(32, 32), dtype="float16", scope="shared.dyn", **kwargs):
    return tirx.decl_buffer(shape, dtype, name=name, scope=scope, **kwargs)


def _vf(body, threads=128, tx=None, annotations=None):
    if tx is None:
        tx = tirx.IterVar(tvm.ir.Range(0, threads), tirx.Var("simtvf_tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    block = tirx.SBlock([], [], [], "SIMT_VF", tirx.AttrStmt(tx, "thread_extent", threads, body), annotations=annotations or {})
    return tirx.SBlockRealize([], True, block)


def _parallel(shape, body, annotations=None):
    indices = [tirx.Var(f"i{k}", "int32") for k in range(len(shape))]
    stmt = body(*indices)
    for index, extent in reversed(list(zip(indices, shape))):
        stmt = tirx.For(index, 0, extent, tirx.ForKind.PARALLEL, stmt, annotations=annotations or {})
    return stmt


def _infer(mod):
    with mod["main"].attrs["target"]:
        return transform.AscendLayoutInference()(mod)


def _layouts(mod):
    return [n.annotations["parallel_loop_layout"] for n in nodes(mod, tirx.For) if "parallel_loop_layout" in n.annotations]


def _assert_ownership(layout, n, threads, width):
    analyzer = tvm.arith.Analyzer()
    i = tirx.Var("index", "int32")
    analyzer.bind(i, tvm.ir.Range(0, n))
    indices, remaining = [], i
    for extent in reversed(layout.get_input_shape()):
        indices.insert(0, remaining % extent)
        remaining = remaining // extent
    assert analyzer.can_prove_equal(layout.map_forward_thread(indices)[0], (i // width) % threads)


def _assert_partition(layout, threads):
    analyzer = tvm.arith.Analyzer()
    assignments = set()
    for indices in product(*(range(int(n)) for n in layout.get_input_shape())):
        thread = int(analyzer.simplify(layout.map_forward_thread(list(indices))[0]))
        slots = tuple(int(analyzer.simplify(x)) for x in layout.map_forward_index(list(indices)))
        assert 0 <= thread < threads
        assert (thread, slots) not in assignments
        assignments.add((thread, slots))


def _check(mod, width, n=1024, threads=128, *, lower=False):
    inferred = _infer(mod)
    layout = _layouts(inferred)[0]
    if width is None:
        _assert_partition(layout, threads)
    else:
        _assert_ownership(layout, n, threads, width)
    if lower:
        with inferred["main"].attrs["target"]:
            return transform.AscendLowerTileOp()(inferred)


def _elementwise(n=8192, threads=2048, *, constraint=None, dtype="float32", alignments=(0, 0, 0)):
    a, b, c = [_buffer(name, (n,), dtype, data_alignment=alignment) for name, alignment in zip("abc", alignments)]
    annotations = {"coalesced_width": tirx.IntImm("int32", 2)} if constraint == "width" else {}
    loop = _parallel((n,), lambda i: tirx.BufferStore(c, a[i] + b[i], [i]), annotations)
    layout = {"layout_map": {a.data: tilelang.layout.Layout((n,), lambda i: i)}} if constraint == "layout" else {}
    return kernel(_vf(loop, threads), buffers=[a, b, c], annotations=layout)


@pytest.mark.parametrize("buffer_index", [0, 2], ids=["load", "store"])
@pytest.mark.parametrize("alignment", [1, 2, 4, 8, 32, 64])
def test_ownership_base_alignment(buffer_index, alignment):
    alignments = [32, 32, 32]
    alignments[buffer_index] = alignment
    # One weakly aligned access retains native width=2; an 8B guarantee suffices.
    _check(_elementwise(alignments=alignments), 1 if alignment >= 8 else 2, 8192, 2048)


@pytest.mark.parametrize("constraint", ["width", "layout"])
def test_ownership_preserves_constraints(constraint):
    _check(_elementwise(constraint=constraint), 2, 8192, 2048)


@pytest.mark.parametrize(
    "dtype,n,threads,width",
    [
        (dtype, 8192, 128, width)
        for dtype, width in [("int8", 4), ("float16", 2), ("bfloat16", 2), ("int16", 2), ("float32", 1), ("int32", 1), ("int64", 1)]
    ]
    + [("int8", 73728, 128, 4)]  # Repetition must fold beyond the 65536-point analysis budget.
    + [("float32", n, threads, None) for n, threads in [(258, 128), (8191, 2048), (256, 16), (130, 33)]],
)
def test_ownership_dtype_and_partition(dtype, n, threads, width):
    _check(_elementwise(n, threads, dtype=dtype), width, n, threads, lower=True)


@pytest.mark.parametrize("dependent", [False, True])
def test_ownership_does_not_change_conflicting_accesses(dependent):
    a = tirx.decl_buffer((512,), "float32", name="a", scope="shared.dyn")
    i = tirx.Var("i", "int32")
    # Either overlapping stores or a loop-carried read/write dependence.
    body = tirx.BufferStore(a, a[i + 1] if dependent else tirx.Cast("float32", i), [i if dependent else i % 128])
    mod = kernel(_vf(tirx.For(i, 0, 256, tirx.ForKind.PARALLEL, body), 32), buffers=[a])
    _assert_ownership(_layouts(_infer(mod))[0], 256, 32, 1 if dependent else 2)


@pytest.mark.parametrize("boundary", ["collective", "escape", "thread_value", "thread_guard"])
def test_ownership_component_boundaries(boundary):
    left, right = _elementwise(), _elementwise()
    buffers = [b for mod in (left, right) for block in nodes(mod, tirx.SBlock) for b in block.alloc_buffers]
    a, b, c = buffers[:3]
    tx = nodes(left, tirx.AttrStmt)[0].node
    loop, sibling = nodes(left, tirx.For)[0], nodes(right, tirx.For)[0]
    suffix = []
    if boundary == "collective":
        values, sums = _buffer("values", (8192,), "float32", "local.fragment"), _buffer("sums", (1,), "float32", "local.fragment")
        buffers += [values, sums]
        reduce = tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tl.tileop.reduce"), [region(values, 1), region(sums, 2), "sum", 0, True]))
        suffix = [copy(c, values), reduce]
    elif boundary == "escape":
        suffix = [tirx.Evaluate(tirx.call_extern("int32", "escape", a.data))]
    elif boundary == "thread_value":
        loop = _parallel((8192,), lambda i: tirx.BufferStore(c, a[i] + b[i] + tirx.Cast("float32", tx.var), [i]))
    else:
        loop = tirx.IfThenElse(tx.var < 1024, loop, None)
    on = _infer(kernel(_vf(seq(loop, *suffix, sibling), 2048, tx), buffers=buffers))
    _assert_ownership(_layouts(on)[0], 8192, 2048, 2)
    _assert_ownership(_layouts(on)[-1], 8192, 2048, 1)


@pytest.mark.parametrize("direction", ["load", "store"])
@pytest.mark.parametrize("copies,alignment,width", [(1, 32, 8), (4, 32, 4), (1, 1, 1)])
def test_ownership_scores_simt_copies(direction, copies, alignment, width):
    strided = _buffer("strided", strides=(1, 32))
    buffers = [strided]
    fragments = []
    transfers = []
    for k in range(copies):
        ub = _buffer(f"ub{k}", data_alignment=alignment)
        if direction == "load" or not fragments:
            fragments.append(_buffer(f"fragment{k}", scope="local.fragment"))
        src, dst = (ub, fragments[-1]) if direction == "load" else (fragments[0], ub)
        transfers.append(copy(src, dst))
        buffers.append(ub)
    if direction == "load":
        compute = _parallel((32, 32), lambda i, j: tirx.BufferStore(strided, sum(f[i, j] for f in fragments), [i, j]))
        body = seq(*transfers, compute)
    else:
        compute = _parallel((32, 32), lambda i, j: tirx.BufferStore(fragments[0], strided[i, j] + 1, [i, j]))
        body = seq(compute, *transfers)
    mod = kernel(_vf(body), buffers=[*buffers, *fragments])
    # Four copies change the preferred width; unknown alignment retains native.
    # The native load-root plan has width=4; the store-root plan has width=1.
    if alignment == 1 and direction == "load":
        width = 4
    _check(mod, width)


@pytest.mark.parametrize(
    "case,copies,start,tail,width", [("serial", 1, 0, 0, 8), ("unrolled", 4, 0, 0, 4), ("serial", 4, 3, 512, 4), ("dependent", 6, 0, 0, 4)]
)
def test_ownership_enclosing_copy_loops(case, copies, start, tail, width):
    a, frag = _buffer("a", strides=(1, 32)), _buffer("frag", (1024,), scope="local.fragment")
    c = _buffer("c", (1024 if case == "dependent" else copies * 1024 - tail,))
    compute = _parallel((32, 32), lambda i, j: tirx.BufferStore(frag, a[i, j] + 1, [i * 32 + j]))
    q, r = tirx.Var("q", "int32"), tirx.Var("r", "int32")

    def transfer(index):
        return copy(frag, c, dst_indices=[0 if case == "dependent" else index * 1024], dst_shape=[1024])

    if case == "dependent":
        # Identical addresses still execute 1+2+3 copies, not 3*1 copies.
        repeated = tirx.For(q, 0, 3, tirx.ForKind.SERIAL, tirx.For(r, 0, q + 1, tirx.ForKind.UNROLLED, transfer(0)))
    else:
        kind = tirx.ForKind.UNROLLED if case == "unrolled" else tirx.ForKind.SERIAL
        repeated = tirx.For(q, start, copies, kind, transfer(q - start))
    actual = _infer(kernel(_vf(seq(compute, repeated)), buffers=[a, frag, c]))
    explicit = _infer(kernel(_vf(seq(compute, *(transfer(k) for k in range(copies)))), buffers=[a, frag, c]))
    assert tvm.ir.structural_equal(_layouts(actual)[0], _layouts(explicit)[0])
    if not tail:
        _assert_ownership(_layouts(actual)[0], 1024, 128, width)


@pytest.mark.parametrize("output_scope", ["global", "local.fragment"])
def test_ownership_scores_ub_condition_loads(output_scope):
    a, c = [_buffer(name, strides=(1, 32)) for name in "ac"]
    flags, out = _buffer("flags", (1024,), "int8", data_alignment=1), _buffer("out", (1024,), "int32", output_scope)

    def body(i, j):
        k = i * 32 + j
        branch = tirx.IfThenElse(flags[k] > 0, tirx.BufferStore(out, 1, [k]), tirx.BufferStore(out, 0, [k]))
        return seq(tirx.BufferStore(c, a[i, j] + 1, [i, j]), branch)

    mod = kernel(
        _vf(_parallel((32, 32), body)),
        buffers=[a, c, flags] + ([out] if output_scope == "local.fragment" else []),
        params=[out] if output_scope == "global" else [],
    )
    # The condition's weakly aligned UB access must retain the native plan.
    _check(mod, 1)


@pytest.mark.parametrize(
    "strides,dtype,offset,width",
    [(stride, dtype, 0, None) for stride in [(32, 1), (256, 1), (64, 2), (256, 2)] for dtype in ["int8", "float16", "float32", "int64"]]
    + [((256, 1), "float16", offset, 1) for offset in [0, 1, 4, 256]]
    + [((1, 32), "float16", 0, 8)],
)
def test_ownership_physical_accesses(strides, dtype, offset, width):
    a, c = [_buffer(name, dtype=dtype, strides=strides, elem_offset=offset) for name in "ac"]
    body = _parallel((32, 32), lambda i, j: tirx.BufferStore(c, a[i, j] + tirx.const(1, dtype), [i, j]))
    lowered = _check(kernel(_vf(body), buffers=[a, c]), width, lower=True)
    # Width=1 avoids mixing 512B-separated rows; width=8 groups transposed
    # accesses beyond the vectorizer's width=1 without inventing vector loads.
    if width == 8:
        assert not any(loop.kind == tirx.ForKind.VECTORIZED for loop in nodes(lowered, tirx.For))


@pytest.mark.parametrize("transpose", [False, True])
def test_ownership_candidate_propagates_through_fragments(transpose):
    a = _buffer("a")
    f = _buffer("f", scope="local.fragment")
    c = _buffer("c")
    producer = _parallel((32, 32), lambda i, j: tirx.BufferStore(f, a[i, j], [i, j]))
    consumer = _parallel((32, 32), lambda i, j: tirx.BufferStore(c, f[j, i] if transpose else f[i, j], [i, j]))
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
        transform.AscendLowerTileOp()(on)


@pytest.mark.parametrize(
    "control,thread_dependent",
    [(control, True) for control in ["then", "else", "nested", "bind", "block", "while", "serial_extent"]]
    + [(control, False) for control in ["then", "bind", "serial", "block_index"]],
)
def test_ownership_enclosing_control(control, thread_dependent):
    mod = _elementwise(n=256, threads=128)
    tx = nodes(mod, tirx.AttrStmt)[0].node
    flag, counter = tirx.Var("flag", "int32"), tirx.Var("counter", "int32")
    condition = tx.var < 64 if thread_dependent else flag > 0
    loop = nodes(mod, tirx.For)[0]
    body = tirx.IfThenElse(condition, loop, None)
    if control == "else":
        body = tirx.IfThenElse(tirx.Not(condition), tirx.Evaluate(0), loop)
    elif control == "nested":
        body = tirx.IfThenElse(condition, body, None)
    elif control == "bind":
        active = tirx.Var("active", "bool")
        body = seq(tirx.Bind(active, condition), tirx.IfThenElse(active, loop, None))
    elif control == "block":
        body = tirx.SBlockRealize([], condition, tirx.SBlock([], [], [], "conditional", loop))
    elif control == "while":
        body = tirx.While(condition, loop)
    elif control == "serial_extent":
        body = tirx.For(counter, 0, tirx.if_then_else(condition, 1, 0), tirx.ForKind.SERIAL, loop)
    elif control in ("serial", "block_index"):
        body = tirx.IfThenElse(condition if thread_dependent else counter == 0, loop, None)
        bx = tirx.IterVar(tvm.ir.Range(0, 2), counter, tirx.IterVar.ThreadIndex, "blockIdx.x")
        body = tirx.For(counter, 0, 2, tirx.ForKind.SERIAL, body) if control == "serial" else tirx.AttrStmt(bx, "thread_extent", 2, body)
    buffers = nodes(mod, tirx.SBlock)[-1].alloc_buffers
    _check(kernel(_vf(body, tx=tx), buffers=buffers, params=[flag]), 2 if thread_dependent else 1, 256)


@pytest.mark.parametrize(
    "direction,case",
    list(product(["load", "store"], ["uniform", "thread_offset", "thread_alias", "thread_guard", "private"]))
    + [("load", "uniform_fallback"), ("load", "thread_fallback")],
)
def test_ownership_copy_dependencies(direction, case):
    a, c = [_buffer(name, (256,), "float32") for name in "ac"]
    frag = _buffer("frag", (256,), "float32", "local.fragment")
    endpoint = _buffer("endpoint", (128 if "fallback" in case else 512,), "float32", "local" if case == "private" else "global")
    # The DSL's simtvf prefix selects parallel lowering for local copies.
    tx = tirx.IterVar(tvm.ir.Range(0, 128), tirx.Var("simtvf_tx", "int32"), tirx.IterVar.ThreadIndex, "threadIdx.x")
    offset = 2 * tx.var if case in ("thread_offset", "thread_alias") else 0
    prefix = []
    if case == "thread_alias":
        alias = tirx.Var("offset_alias", "int32")
        prefix = [tirx.Bind(alias, offset)]
        offset = alias
    transfer = (
        copy(endpoint, frag, src_indices=[offset], src_shape=[256])
        if direction == "load"
        else copy(frag, endpoint, dst_indices=[offset], dst_shape=[256])
    )
    if case == "thread_guard":
        transfer = tirx.IfThenElse(tx.var < 64, transfer, None)
    compute = _parallel(
        (256,), lambda i: tirx.BufferStore(c if direction == "load" else frag, frag[i] + a[i] if direction == "load" else a[i] + 1, [i])
    )
    body = seq(*prefix, transfer, compute) if direction == "load" else seq(*prefix, compute, transfer)
    fallback = tirx.Cast("float32", tx.var) if case == "thread_fallback" else tirx.const(3, "float32")
    vf = _vf(body, tx=tx, annotations={"safe_value_map": {endpoint.data: fallback}} if "fallback" in case else {})
    mod = kernel(vf, buffers=[a, c, frag] + ([endpoint] if case == "private" else []), params=[] if case == "private" else [endpoint])
    # Constant addresses are still owner-dependent for thread-private storage.
    _check(mod, 1 if case in ("uniform", "uniform_fallback") else 2, 256, lower=True)
