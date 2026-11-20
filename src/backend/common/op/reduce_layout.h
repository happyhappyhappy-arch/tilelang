#ifndef TILELANG_BACKEND_COMMON_OP_REDUCE_LAYOUT_H_
#define TILELANG_BACKEND_COMMON_OP_REDUCE_LAYOUT_H_

#include "layout/layout.h"
#include "support/check.h"
#include <limits>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tirx/op.h>
#include <vector>

namespace tvm::tl::backend::reduce {
using namespace tirx;

inline ffi::Array<PrimExpr> InputPlaceholders(size_t n) {
  ffi::Array<PrimExpr> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(InputPlaceholder(i));
  }
  return result;
}

struct ThreadReduceStep {
  int extent;
  int scale;
  // Position of this split inside the reduce var: the split covers
  // floormod(floordiv(rv, lower_factor), extent).
  int64_t lower_factor;

  int ReducingThreads() const {
    ICHECK_LE(extent, std::numeric_limits<int>::max() / scale)
        << "Reduce thread count overflow: extent=" << extent
        << ", scale=" << scale;
    return extent * scale;
  }
};

inline std::vector<ThreadReduceStep>
CollectThreadReduceSteps(const arith::IterSumExpr &thread_iter_sum,
                         const Var &reduce_var) {
  std::vector<ThreadReduceStep> steps;
  for (const auto &iter_split : thread_iter_sum->args) {
    auto mark = iter_split->source->source.as<Var>();
    if (!mark || !mark.value().same_as(reduce_var)) {
      continue;
    }

    auto scale = as_const_int(iter_split->scale);
    auto extent = as_const_int(iter_split->extent);
    auto lower_factor = as_const_int(iter_split->lower_factor);
    ICHECK(scale != nullptr && extent != nullptr && lower_factor != nullptr);
    if (*extent == 1) {
      continue;
    }
    ICHECK_LE(*scale, std::numeric_limits<int>::max());
    ICHECK_LE(*extent, std::numeric_limits<int>::max());
    steps.push_back(ThreadReduceStep{static_cast<int>(*extent),
                                     static_cast<int>(*scale), *lower_factor});
  }
  return steps;
}

} // namespace tvm::tl::backend::reduce

#endif // TILELANG_BACKEND_COMMON_OP_REDUCE_LAYOUT_H_
