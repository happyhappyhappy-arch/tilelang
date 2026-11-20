#ifndef TVM_TL_OP_ANNOTATED_CAST_H_
#define TVM_TL_OP_ANNOTATED_CAST_H_

#include "builtin.h"
#include <optional>
#include <tvm/ir/attrs.h>
#include <tvm/ir/prim/expr.h>

namespace tvm::tl {

// Present an ordinary upstream cast and a TileLang cast to the same lowering
// rules. Random bits are an expression operand so IR rewrites visit them.
struct CastOperands {
  PrimExpr value;
  PrimType dtype;
  ffi::Map<ffi::String, ffi::Any> attributes;
  std::optional<PrimExpr> random_bits;
};

inline std::optional<CastOperands> GetCastOperands(const PrimExpr &expr) {
  if (const auto *cast = expr.as<prim::CastNode>()) {
    return CastOperands{cast->value, expr.ty(), {}, std::nullopt};
  }
  const auto *call = expr.as<CallNode>();
  if (!call || !call->op.same_as(annotated_cast())) {
    return std::nullopt;
  }
  TVM_FFI_ICHECK(call->args.size() == 1 || call->args.size() == 2);
  CastOperands cast{
      call->args[0].as_or_throw<PrimExpr>(), expr.ty(), {}, std::nullopt};
  if (call->attrs.defined()) {
    cast.attributes = call->attrs.as_or_throw<DictAttrs>()->dict;
  }
  if (call->args.size() == 2) {
    cast.random_bits = call->args[1].as_or_throw<PrimExpr>();
  }
  return cast;
}
} // namespace tvm::tl
#endif
