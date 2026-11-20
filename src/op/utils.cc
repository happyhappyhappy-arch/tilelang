/*!
 * \file tl/op/utils.cc
 * \brief Common utilities implementation for TL ops.
 */

#include "utils.h"

#include <cmath>

#include "builtin.h"
#include "support/check.h"
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/logging.h>

#include <tvm/tirx/builtin.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

// Decoded form of a tl.region(...) bridge call.
struct DecodedRegionCall {
  BufferVar buffer;
  ffi::Array<Range> ranges;
  int access_mask;
};

// Decode a tl.region call's args:
//   args[0]: BufferLoad whose indices are per-axis minima.
//   args[1]: constant int access mask (1=read, 2=write, 3=read-write).
//   args[2 + i]: extent of axis i (supports dynamic PrimExpr).
// A min encoded as a stride-1 Ramp carries its extent in the Ramp lanes; the
// lanes must agree with the provided extent when both are constant.
DecodedRegionCall DecodeRegionCall(const ffi::Array<Expr> &args) {
  size_t n = args.size();
  ICHECK_GE(n, 2U);
  size_t ndim = n - 2;
  const auto *load = args[0].as<TensorLoadNode>();
  ICHECK(load);
  ICHECK(load->indices.size() == ndim)
      << "load->indices.size() = " << load->indices << " ndim = " << ndim;
  ffi::Array<Range> ranges;
  for (size_t i = 0; i < ndim; i++) {
    PrimExpr index = load->indices[i];
    PrimExpr extent = args[2 + i].as_or_throw<PrimExpr>();
    if (const auto *ramp = index.as<prim::RampNode>()) {
      const auto *stride_imm = ramp->stride.as<IntImmNode>();
      ICHECK(stride_imm && stride_imm->value == 1)
          << "tl.region expects stride-1 Ramp for index";
      if (const auto *lanes_imm = ramp->lanes.as<IntImmNode>()) {
        if (const auto *ext_imm = extent.as<IntImmNode>()) {
          ICHECK_EQ(lanes_imm->value, ext_imm->value)
              << "Ramp lanes and provided extent must match";
        }
      }
      ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
    } else {
      ranges.push_back(Range::FromMinExtent(index, extent));
    }
  }
  const int64_t *mask = as_const_int(args[1].as_or_throw<PrimExpr>());
  ICHECK(mask) << "tl.region access mask must be a constant int, but got "
               << args[1];
  return {load->source.as_or_throw<BufferVar>(), std::move(ranges),
          static_cast<int>(*mask)};
}

} // namespace

bool IsZeroBitPattern(const PrimExpr &expr) {
  if (const auto *broadcast = expr.as<prim::BroadcastNode>()) {
    return IsZeroBitPattern(broadcast->value);
  }
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value == 0;
  }
  if (const auto *imm = expr.as<FloatImmNode>()) {
    return imm->value == 0.0 && !std::signbit(imm->value);
  }
  return false;
}

bool IsBufferLikeExpr(const Expr &expr) {
  if (auto *load = expr.as<TensorLoadNode>()) {
    return load->source.as<BufferVar>().has_value();
  }
  if (expr.as<BufferRegionNode>()) {
    return true;
  }
  if (const auto *call = expr.as<CallNode>()) {
    return (call->op.same_as(region()));
  }
  return false;
}

BufferRegion NormalizeToBufferRegion(const Expr &arg) {
  // Case 1: Already a BufferRegion
  if (arg->IsInstance<BufferRegionNode>()) {
    return arg.as_or_throw<BufferRegion>();
  }

  // Case 2: BufferLoad — convert indices to ranges (Ramp -> lanes, else
  // extent=1)
  if (const auto *load = arg.as<TensorLoadNode>()) {
    ffi::Array<Range> ranges;
    for (const PrimExpr &index : load->indices) {
      if (const auto *ramp = index.as<prim::RampNode>()) {
        ICHECK(ramp->stride.as<IntImmNode>()) << "Ramp stride must be IntImm";
        ICHECK_EQ(ramp->stride.as<IntImmNode>()->value, 1)
            << "Only stride-1 Ramp is supported in region conversion";
        ICHECK(ramp->lanes.as<IntImmNode>())
            << "Scalable vector lanes not supported in region conversion";
        ranges.push_back(Range::FromMinExtent(ramp->base, ramp->lanes));
      } else {
        ranges.push_back(Range::FromMinExtent(index, 1));
      }
    }
    return BufferRegion(load->source.as_or_throw<BufferVar>(), ranges);
  }

  // Case 3: tl.region(...) — decode the transport bridge
  if (const auto *call = arg.as<CallNode>()) {
    if (call->op.same_as(region())) {
      DecodedRegionCall decoded = DecodeRegionCall(call->args);
      return BufferRegion(decoded.buffer, decoded.ranges);
    }
    LOG(FATAL) << "Unsupported argument for BufferRegion (expect "
                  "BufferLoad/BufferRegion/tl.region): "
               << arg;
  }

  LOG(FATAL) << "Unsupported argument for BufferRegion: " << arg;
  throw; // Unreachable
}

AccessRegion NormalizeToAccessRegion(const Expr &arg, int default_access_mask) {
  if (const auto *call = arg.as<CallNode>()) {
    if (call->op.same_as(region())) {
      DecodedRegionCall decoded = DecodeRegionCall(call->args);
      return {BufferRegion(decoded.buffer, decoded.ranges),
              decoded.access_mask};
    }
  }
  return {NormalizeToBufferRegion(arg), default_access_mask};
}

Expr MakeAccessPtrFromRegion(const BufferRegion &region, int rw_mask,
                             bool require_2d) {
  BufferVar buf = region->buffer;
  int ndim = static_cast<int>(buf->shape.size());
  if (require_2d) {
    ICHECK(ndim >= 2) << "Expect buffers with at least 2 dims";
  }

  PrimExpr offset, extent;
  if (ndim == 0) {
    offset = 0;
    extent = 1;
  } else if (ndim == 1) {
    // 1D: straightforward
    auto axis = region->region[0];
    offset = axis->min;
    extent = axis->extent;
  } else {
    // Compute row-major strides
    std::vector<PrimExpr> strides(ndim);
    PrimExpr one = MakeConst(buf->shape[0].ty(), 1);
    PrimExpr cur = one;
    for (int i = ndim - 1; i >= 0; --i) {
      strides[i] = cur;
      cur = cur * buf->shape[i];
    }
    // Offset: sum_{i in [0..ndim-1]} min_i * stride_i
    offset = MakeConst(buf->shape[0].ty(), 0);
    for (int i = 0; i < ndim; ++i) {
      offset = offset + region->region[i]->min * strides[i];
    }
    // Extent: last two extents product (elements)
    extent =
        region->region[ndim - 2]->extent * region->region[ndim - 1]->extent;
  }

  // ptype and return handle
  Expr ptype = tirx::TypeAnnotation(buf->dtype);
  ffi::Array<Expr> acc_args{ptype, buf.data(), offset, extent,
                            IntImm(PrimType::Int(32), rw_mask)};
  return Call(PointerType(PrimType::Void(), buf.scope()),
              builtin::tvm_access_ptr(), acc_args);
}

Expr MakeAccessPtrFromBufferLoad(const TensorLoad &load, int rw_mask) {
  BufferVar buf = load->source.as_or_throw<BufferVar>();
  int ndim = static_cast<int>(buf->shape.size());

  // Compute offset using row-major layout (iterate in reverse)
  PrimExpr offset = 0;
  PrimExpr stride = 1;

  for (int i = ndim - 1; i >= 0; --i) {
    const PrimExpr &index = load->indices[i];
    if (const auto *ramp = index.as<prim::RampNode>()) {
      // For Ramp, use the base
      offset = offset + ramp->base * stride;
    } else {
      // For scalar index (IntImm or other PrimExpr)
      offset = offset + index * stride;
    }
    stride = stride * buf->shape[i];
  }

  // Extent is 1 element for a single BufferLoad access
  PrimExpr extent = MakeConst(PrimType::Int(32), 1);

  // Build access_ptr
  Expr ptype = tirx::TypeAnnotation(buf->dtype);
  ffi::Array<Expr> acc_args{ptype, buf.data(), offset, extent,
                            IntImm(PrimType::Int(32), rw_mask)};
  return Call(PointerType(PrimType::Void(), buf.scope()),
              builtin::tvm_access_ptr(), acc_args);
}

// Maps TVM PrimType to CUDA's CUtensorMapDataType enum value.
int to_CUtensorMapDataType(PrimType dtype) {
  // CUDA 13 adds packed U4 TensorMap formats. The vendored CUDA stub may lag
  // the installed toolkit, so keep the enum value by CUDA's documented order.
  constexpr int kTensorMapDataType16U4Align8B = 13;
  constexpr int kTensorMapDataType16U4Align16B = 14;
  if (dtype.MatchesElementType(kTileLangUnpackedFP4, 8)) {
    return kTensorMapDataType16U4Align16B;
  }
  if (dtype.MatchesElementType(kDLFloat4_e2m1fn, 4)) {
    return kTensorMapDataType16U4Align8B;
  }

  CUtensorMapDataType tp;
  if (dtype.MatchesCode(kDLFloat)) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.MatchesElementType(kDLBfloat, 16)) {
    tp = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if (dtype.MatchesElementType(kTileLangTensorFloat32, 32)) {
    // tfloat32 uses same memory layout as float32
    tp = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else if (dtype.bits() == 8 &&
             dtype.MatchesCode(kDLFloat8_e3m4, kDLFloat8_e4m3,
                               kDLFloat8_e4m3b11fnuz, kDLFloat8_e4m3fn,
                               kDLFloat8_e4m3fnuz, kDLFloat8_e5m2,
                               kDLFloat8_e5m2fnuz, kDLFloat8_e8m0fnu)) {
    tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype.MatchesCode(kDLInt)) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_INT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.MatchesCode(kDLUInt)) {
    switch (dtype.bits()) {
    case 64:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT64;
      break;
    case 32:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT32;
      break;
    case 16:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = CU_TENSOR_MAP_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else {
    ICHECK(0) << dtype;
  }
  return static_cast<int>(tp);
}

} // namespace tl
} // namespace tvm
