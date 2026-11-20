import pytest


@pytest.fixture
def flatten_buffer():
    import tilelang

    return tilelang.tvm, tilelang.transform.FlattenBuffer()


@pytest.mark.parametrize(
    ("extent", "stride", "expected_dtype"),
    [(1 << 30, 4, "int64"), (1 << 20, 2, "int32")],
)
def test_access_ptr_offset_width(flatten_buffer, extent, stride, expected_dtype):
    tvm, flatten = flatten_buffer
    tirx = tvm.tirx
    buf = tirx.decl_buffer((1,), "float16", name="buf")
    i = tirx.Var("i", "int32")
    access = tirx.tvm_access_ptr(tirx.type_annotation("float16"), buf.data, i * stride, 1, 1)
    loop = tirx.For(i, 0, extent, tirx.ForKind.SERIAL, tirx.Evaluate(access))
    func = tirx.PrimFunc([buf], loop)
    assert str(access.args[2].ty.dtype) == "int32"

    decl, loop = flatten(tvm.IRModule({"main": func}))["main"].body.seq
    assert decl.data.args[0].same_as(buf)
    offset = loop.body.value.args[2]
    assert str(offset.ty.dtype) == expected_dtype


def test_bool_backing_and_initializer(flatten_buffer):
    tvm, flatten = flatten_buffer
    tirx = tvm.tirx
    buf = tirx.decl_buffer((2, 3), "bool", name="flags")
    value = tirx.Var("value", "bool")
    body = tirx.SeqStmt(
        [
            tirx.AllocBuffer(buf),
            tirx.BufferStore(buf, value, [1, 2]),
            tirx.Evaluate(buf[1, 2]),
            tirx.Evaluate(tirx.address_of(buf[0, 1])),
        ]
    )
    func = tirx.PrimFunc([value], body).with_attr("tl.local_var_init", {buf: tirx.IntImm("int32", 1)})
    alloc, store, read, address = flatten(tvm.IRModule({"main": func}))["main"].body.seq
    assert int(alloc.annotations["tl.local_var_init"]) == 1
    assert alloc.buffer.dtype == "int8"
    assert store.buffer.same_as(alloc.buffer) and int(store.indices[0]) == 5
    assert str(store.value.ty.dtype) == "int8"
    assert str(read.value.ty.dtype) == "bool" and read.value.value.source.same_as(alloc.buffer)
    assert address.value.args[0].source.same_as(alloc.buffer)


def test_masked_boolean_access(flatten_buffer):
    tvm, flatten = flatten_buffer
    tirx = tvm.tirx
    buf = tirx.decl_buffer((2, 3), "bool", name="flags")
    value = tirx.Var("value", "bool")
    mask = tirx.IntImm("bool", 1)
    body = tirx.SeqStmt(
        [
            tirx.Evaluate(tirx.op.masked_store(buf, value, 1, 2, mask)),
            tirx.Evaluate(tirx.op.masked_load("bool", buf, 1, 0, mask)),
        ]
    )
    func = tirx.PrimFunc([buf, value], body)
    decl, store, read = flatten(tvm.IRModule({"main": func}))["main"].body.seq
    assert store.value.args[0].same_as(decl.buffer)
    assert str(store.value.args[1].ty.dtype) == "int8" and int(store.value.args[2]) == 5
    assert str(read.value.ty.dtype) == "bool" and str(read.value.value.ty.dtype) == "int8"
    assert int(read.value.value.args[1]) == 3
