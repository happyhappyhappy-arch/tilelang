/*!
 * \file tl/cpu/op/builtin.cc
 * \brief Registration of CPU-specific TileLang pass configurations.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

TVM_REGISTER_PASS_CONFIG_OPTION(kCPUParallel, bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kCPUParallelMinTrip, int64_t);

} // namespace tl
} // namespace tvm
