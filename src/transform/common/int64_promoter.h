/*!
 * \file int64_promoter.h
 * \brief Helper rewriter that promotes sub-64-bit integer expressions to int64.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_
#define TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_

#include "support/cast.h"
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include "../../../3rdparty/tvm/src/tirx/ir/data_type_rewriter.h"

namespace tvm {
namespace tl {

/*!
 * \brief Promote integer variables, immediates, and casts to int64.
 *
 * Used by passes that need to widen index expressions to avoid overflow.
 */
class Int64Promoter : public tirx::IndexDataTypeRewriter {
public:
  using Parent = tirx::IndexDataTypeRewriter;

  Expr VisitExpr_(const VarNode *op) final {
    Expr var = Parent::VisitExpr_(op);
    if (auto scalar = var.as<PrimExpr>();
        scalar && IsNarrowSignedInt(scalar->ty())) {
      return cast(PrimType::Int(64), *scalar);
    }
    return var;
  }

  Expr VisitExpr_(const IntImmNode *op) final {
    auto value = ffi::GetRef<IntImm>(op);
    if (IsNarrowSignedInt(value.ty()))
      return IntImm(PrimType::Int(64), op->value);
    return value;
  }

  Expr VisitExpr_(const prim::CastNode *op) final {
    auto value = ffi::GetRef<PrimExpr>(op);
    if (IsNarrowSignedInt(value.ty())) {
      // Keep the narrowing conversion before extending its result.
      return cast(value.ty().WithBits(64), value);
    }
    return value;
  }

private:
  static bool IsNarrowSignedInt(PrimType type) {
    return type.MatchesCode(kDLInt) && type.bits() < 64;
  }
};

inline PrimExpr PromoteIndexIfNeeded(const PrimExpr &index,
                                     arith::AnalyzerObj *analyzer) {
  auto type = index.ty();
  if (type.MatchesCode(kDLInt) && type.bits() < 64) {
    const auto bound = analyzer->const_int_bound(index);
    const int64_t limit = int64_t{1} << (type.bits() - 1);
    if (bound->max_value >= limit - 1 || bound->min_value < -limit) {
      return Int64Promoter()(index).as_or_throw<PrimExpr>();
    }
  }
  return index;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_INT64_PROMOTER_H_
