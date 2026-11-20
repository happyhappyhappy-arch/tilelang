import ctypes
import shutil
import subprocess

import pytest
import tvm_ffi
from tvm_ffi import libinfo as ffi_libinfo

from tilelang import tvm
from tvm import tirx


def test_return_with_intermediate_values(tmp_path):
    """Return expressions may emit declarations before the C return statement."""
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("A C compiler is required to execute the generated host code")

    x = tirx.Var("x", "int32")
    y = tirx.Var("y", "int32")
    func = tirx.PrimFunc([x, y], tirx.Return(tirx.Min(2 * x + 1, y - 3)), tvm.ir.PrimType("int32"))
    func = func.with_attr("global_symbol", "tl_min")
    mod = tvm.IRModule({"tl_min": func})
    build = tvm_ffi.get_global_func("target.build.tilelang_c_host")
    source = tmp_path / "host.c"
    source.write_text(build(mod, tvm.target.Target("c")).inspect_source())
    library = tmp_path / "host.so"
    includes = [tvm.libinfo.find_include_path(), ffi_libinfo.find_include_path(), ffi_libinfo.find_dlpack_include_path()]
    subprocess.run([cc, "-std=c11", "-shared", "-fPIC", *(f"-I{path}" for path in includes), source, "-o", library], check=True)

    generated_min = ctypes.CDLL(str(library)).tl_min
    generated_min.argtypes = [ctypes.c_int32, ctypes.c_int32]
    generated_min.restype = ctypes.c_int32
    for a, b in [(-9, -8), (2, 14), (8, 1), (0, 0)]:
        assert generated_min(a, b) == min(2 * a + 1, b - 3)
