/*!
 * \file tl/op/math.cc
 * \brief Math operations.
 *
 */

#include "support/check.h"
#include "utils.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

PrimExpr pow_of_int_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const auto &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr base = arg[0].as_or_throw<PrimExpr>();
  PrimExpr exp = arg[1].as_or_throw<PrimExpr>();
  ffi::String pow_of_int_name =
      "tl::pow_of_int<" + std::to_string(exp.as_or_throw<IntImm>()->value) +
      ">";
  return Call(base.ty(), tirx::builtin::call_extern(),
              {prim::StringImm(pow_of_int_name), base})
      .as_or_throw<PrimExpr>();
}

TVM_REGISTER_OP("tl.pow_of_int")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "pow_of_int")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", pow_of_int_op)
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", pow_of_int_op);

PrimExpr infinity_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  PrimType dtype = args.ty();
  ICHECK_EQ(dtype.lanes(), 1);

  // NOTE(wt): Codegen for PrintConst:Inf will handle this based on dtype
  if (dtype.MatchesCode(kDLFloat)) {
    if (dtype.bits() == 64 || dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(dtype, std::numeric_limits<float>::infinity(),
                      call->span);
    }
  } else if (dtype.MatchesElementType(kDLBfloat, 16)) {
    return FloatImm(dtype, std::numeric_limits<float>::infinity(), call->span);
  } else if (dtype.MatchesElementType(kTileLangTensorFloat32, 32)) {
    return FloatImm(dtype, std::numeric_limits<float>::infinity(), call->span);
  } else if (dtype.MatchesElementType(kDLFloat8_e5m2, 8)) {
    // e5m2 is the only fp8 format with a representable inf; the rest keep
    // the fatal below.
    return FloatImm(dtype, std::numeric_limits<float>::infinity(), call->span);
  }
  LOG(FATAL) << "Cannot decide infinity for type " << dtype;
  throw; // Unreachable, keeps compiler happy
}

TVM_REGISTER_OP("tl.infinity")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "infinity")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", infinity_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", infinity_op)
    .set_attr<FLowerIntrinsic>("default.FLowerIntrinsic", infinity_op);

PrimExpr round_ties_away_from_zero_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1);
  PrimType dtype = args.ty();
  if (dtype.MatchesCode(kDLInt, kDLUInt, kDLBool)) {
    return call->args[0].as_or_throw<PrimExpr>();
  }
  return Call(dtype, tirx::builtin::call_pure_extern(),
              {prim::StringImm("tl::RoundTiesAwayFromZero"), call->args[0]},
              call->attrs)
      .as_or_throw<PrimExpr>();
}

TVM_REGISTER_OP("tl.round_ties_away_from_zero")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               static_cast<int64_t>(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                  "round_ties_away_from_zero")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               round_ties_away_from_zero_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               round_ties_away_from_zero_op);

} // namespace tl
} // namespace tvm
