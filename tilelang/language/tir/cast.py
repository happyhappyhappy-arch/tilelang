"""TileLang casts with optional backend rounding and saturation attributes."""

from tvm import tirx
from tvm.tirx.script.builder import ir as builder_ir


def Cast(dtype, value, annotations=None, span=None):
    """Construct a cast and preserve explicit attributes even for equal types."""
    if not annotations:
        return tirx.Cast(dtype, value, span=span)
    attrs = dict(annotations)
    operands = [tirx.convert(value)]
    if "rbits" in attrs:
        # Random bits are an input expression and must follow IR substitutions.
        operands.append(tirx.convert(attrs.pop("rbits")))
    return tirx.call_intrin(dtype, "tl.annotated_cast", *operands, attrs=attrs, span=span)


_VALID_CAST_ROUNDING_MODES = {"", "rn", "rz", "rp", "rm", "rs"}


def cast(value, dtype, round: str = "", sat: bool = True, rbits=None, span=None):
    """Cast ``value`` to ``dtype`` with optional PTX-style rounding hints.

    Parameters
    ----------
    value : object
        The source operand.
    dtype : str
        The target data type.
    round : str, optional
        PTX rounding modifier (e.g. ``"rn"``, ``"rz"``, ``"rp"``, ``"rm"``,
        ``"rs"``). Empty string means use the backend default. Currently
        the CUDA backend supports ``"rn"`` for float32 to tfloat32 and
        ``"rs"`` for supported low-precision conversions. Other explicit
        modes are reserved. TF32 ``"rn"`` requires SM90+ with ``sat=False``
        or SM100+ with ``sat=True``.
    sat : bool, optional
        Saturate to finite (``True`` = PTX ``.satfinite``, default).
    rbits : PrimExpr, optional
        Random bits operand for stochastic rounding (``round="rs"``).
    span : Optional[Span]
        The location of this operator in the source.

    Returns
    -------
    op : tvm.ir.Expr
        The cast expression.
    """
    if round not in _VALID_CAST_ROUNDING_MODES:
        raise ValueError(f"Invalid round '{round}'. Must be one of: {sorted(_VALID_CAST_ROUNDING_MODES)}")
    if not isinstance(sat, bool):
        raise ValueError(f"Invalid sat '{sat}'. Must be a bool (True for satfinite, False for no saturation)")
    if round == "rs" and rbits is None:
        raise ValueError("rbits is required when round='rs' (stochastic rounding)")
    if round != "rs" and rbits is not None:
        raise ValueError("rbits is only valid with round='rs' (stochastic rounding)")

    annotations: dict = {}
    if round:
        annotations["round"] = tirx.StringImm(round)
    if not sat:
        annotations["sat"] = tirx.IntImm("bool", 0)
    if rbits is not None:
        annotations["rbits"] = rbits

    if annotations:
        return Cast(dtype, value, annotations, span)
    return builder_ir.cast(value, dtype, span)
