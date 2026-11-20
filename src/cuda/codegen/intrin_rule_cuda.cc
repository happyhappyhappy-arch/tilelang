/*!
 * \file intrin_rule_cuda.cc
 * \brief CUDA intrinsic rules.
 */
#include "support/check.h"
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, CUDA fast math.
using tirx::FLowerIntrinsic;
using namespace ffi;

struct TileLangCUDAMath {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLFloat)) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if ((t.MatchesCode(kDLBfloat) && t.bits() == 16)) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.MatchesCode(kDLInt) || t.MatchesCode(kDLUInt)) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

static PrimExpr DispatchCUDAPopcount(const PrimExpr &e) {
  const auto *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1U);
  PrimType dtype = call->args[0].as_or_throw<PrimExpr>().ty();
  const int bits = dtype.bits();
  if ((!dtype.MatchesCode(kDLInt) && !dtype.MatchesCode(kDLUInt)) ||
      (bits != 8 && bits != 16 && bits != 32 && bits != 64)) {
    return e;
  }

  // Preserve the original bit width before widening: int8(-1) has 8 set bits.
  PrimExpr arg = cast(PrimType::UInt(bits, dtype.lanes()),
                      call->args[0].as_or_throw<PrimExpr>());
  if (bits < 32) {
    arg = cast(PrimType::UInt(32, dtype.lanes()), arg);
  }
  PrimExpr count =
      Call(PrimType::Int(32, dtype.lanes()), builtin::call_pure_extern(),
           {prim::StringImm(bits == 64 ? "__popcll" : "__popc"), arg},
           call->attrs)
          .as_or_throw<PrimExpr>();
  return cast(e.ty(), count);
}

static PrimExpr DispatchCUDAIsFinite(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1U);

  PrimType arg_dtype = call->args[0].as_or_throw<PrimExpr>().ty();
  if (arg_dtype.MatchesCode(kDLFloat) &&
      (arg_dtype.bits() == 32 || arg_dtype.bits() == 64)) {
    Array<Expr> new_args = {prim::StringImm("isfinite"), call->args[0]};
    return Call(e.ty(), builtin::call_pure_extern(), new_args, call->attrs)
        .as_or_throw<PrimExpr>();
  }

  return e;
}

TVM_REGISTER_OP("tirx.popcount")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAPopcount,
                               11);

TVM_REGISTER_OP("tirx.asin")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.acos")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.asinh")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.acosh")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.atanh")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.log1p")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.atan2")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.hypot")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.copysign")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.ldexp")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.nextafter")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               DispatchPureExtern<TileLangCUDAMath>);

TVM_REGISTER_OP("tirx.isfinite")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", DispatchCUDAIsFinite);

} // namespace intrin
} // namespace codegen
} // namespace tvm
