/*!
 * \file canonicalize_legacy_reducer.cc
 * \brief Rewrite legacy (v1) reducer syntax into first-class reducer v2 ops.
 *
 * Legacy form (deprecated):
 *   acc = T.alloc_reducer(shape, dtype, op, replication="all"/"none")
 *   T.clear(acc)                      # or T.fill(acc, value)
 *   acc[i] += v                       # sum; T.max/T.min RMW for max/min
 *   T.finalize_reducer(acc)           # in-place
 *   ... reads of acc ...
 *
 * The same legacy syntax is also accepted on a v2 allocation
 * (T.alloc_reducer without `replication`, the pre-v2 default): a
 * `local.reducer` buffer that is never touched by any first-class v2 op
 * is canonicalized under the same whitelist, keeping code written against
 * the old default compiling. Mixing v2 ops and legacy syntax on one buffer
 * is not canonicalized and fails in VerifyReducerEpoch.
 *
 * Canonical v2 form produced by this pass:
 *   local.reducer allocation + reducer_info_v2 annotation
 *   tl.reducer_init(acc)
 *   tl.reducer_update(acc[i], v)
 *   tl.finalize_reducer_v2(acc, dst)  # fresh dst fragment
 *   ... reads redirected to dst ...
 *
 * Mapping rules (strict whitelist — anything else is a compile error, never
 * silently accepted):
 *   - the fill value must be the combine identity (plain init), except for
 *     idempotent combines (max/min) where a non-identity fill is equivalent
 *     to a one-time seed and is forwarded as such. A non-zero sum fill is
 *     rejected: its v1 behavior was participant-count-dependent.
 *   - an update store must be exactly `combine(acc[idx], value)` with the
 *     declared combine op and identical load/store indices.
 *   - `T.finalize_reducer(acc)`'s batch annotation is forwarded.
 *   - accesses to the reducer after finalize read the new destination
 *     fragment; writes after finalize are rejected.
 *
 * This pass runs before VerifyReducerEpoch, so converted programs are then
 * verified under the same rules as native v2 code. It is a deprecation shim:
 * once legacy syntax is removed, this file is deleted with it.
 */

#include "support/cast.h"
#include "support/check.h"
#include <tvm/ir/prim/expr.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "../op/builtin.h"
#include "../op/fill.h"
#include "../op/reducer.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

// Reducer calls accept a buffer region, a single load, or the legacy
// tl.region descriptor. All three identify storage by the buffer variable.
const VarNode *RegionBuffer(const Expr &arg) {
  if (auto region = arg.as<BufferRegionNode>())
    return region->buffer.get();
  if (auto load = arg.as<TensorLoadNode>())
    return load->source.as<VarNode>();
  if (auto call = arg.as<CallNode>();
      call && call->op.same_as(region()) && !call->args.empty()) {
    if (auto load = call->args[0].as<TensorLoadNode>())
      return load->source.as<VarNode>();
  }
  return arg.as<BufferVar>() ? arg.as<VarNode>() : nullptr;
}

BufferVar WithStorageScope(const BufferVar &buffer, String name, String scope) {
  auto type = ffi::make_object<BufferTypeNode>(*buffer.operator->());
  type->storage_scope = std::move(scope);
  return RebuildBufferVar(buffer, std::move(type), std::move(name));
}

/*! \brief Evaluate a (possibly cast-wrapped) constant expression as double. */
std::optional<double> TryConstValue(PrimExpr e) {
  while (const auto *cast = e.as<prim::CastNode>()) {
    e = cast->value;
  }
  if (const auto *i = e.as<IntImmNode>()) {
    return static_cast<double>(i->value);
  }
  if (const auto *f = e.as<FloatImmNode>()) {
    return f->value;
  }
  return std::nullopt;
}

/*! \brief Collect the buffer vars referenced by any first-class v2 reducer
 *  op. A v2-allocated reducer outside this set only ever appears in legacy
 *  syntax and is eligible for canonicalization. */
class ReducerV2OpUseCollector : public StmtExprVisitor {
public:
  static std::unordered_set<const VarNode *> Collect(const Stmt &body) {
    ReducerV2OpUseCollector collector;
    collector.VisitStmt(body);
    return std::move(collector.used_);
  }

private:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(ReducerInitOp::Get()) ||
        op->op.same_as(FinalizeReducerV2Op::Get()) ||
        op->op.same_as(reducer_update())) {
      if (auto var = RegionBuffer(op->args[0]))
        used_.insert(var);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  std::unordered_set<const VarNode *> used_;
};

class LegacyReducerCanonicalizer : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    LegacyReducerCanonicalizer canonicalizer;
    canonicalizer.v2_op_used_ = ReducerV2OpUseCollector::Collect(f->body);
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = canonicalizer.VisitStmt(f->body);
    return f;
  }

private:
  enum class Phase : int { kAllocated, kActive, kFinalized };

  struct LegacyReducer {
    BufferVar old_buffer;
    BufferVar acc; // new local.reducer handle
    BufferVar dst; // out-of-place finalize destination
    ReducerV2OpType op;
    Optional<PrimExpr> seed;
    Phase phase{Phase::kAllocated};
    bool was_v2{false}; // v2 allocation used with legacy syntax only
  };

  // ---- allocation & annotation rewrite ------------------------------------

  Stmt VisitStmt_(const SBlockNode *op) final {
    if (auto anno = op->annotations.Get(tl::attr::kReducerInfo)) {
      auto map = anno.value().as<Map<Var, Map<String, String>>>();
      ICHECK(map) << "malformed reducer_info annotation";
      for (const auto &[var, info] : map.value()) {
        legacy_op_.emplace(var.get(),
                           ParseReducerV2OpType(info.Get("op").value()));
      }
    }
    // A v2 allocation never touched by a first-class v2 op carries the old
    // default of pre-v2 T.alloc_reducer: canonicalize its legacy syntax too.
    if (auto anno = op->annotations.Get(attr::kReducerInfoV2)) {
      auto map = anno.value().as<Map<Var, Map<String, Any>>>();
      ICHECK(map) << "malformed reducer_info_v2 annotation";
      for (const auto &[var, info] : map.value()) {
        if (v2_op_used_.count(var.get())) {
          continue;
        }
        auto op_str = info.Get("op");
        ICHECK(op_str) << "reducer_info_v2 for `" << var
                       << "` is missing the combine op";
        legacy_v2_op_.emplace(
            var.get(), ParseReducerV2OpType(op_str.value().cast<String>()));
      }
    }
    for (const auto &buffer : op->alloc_buffers) {
      LegacyReducer entry;
      auto it = legacy_op_.find(buffer.get());
      if (it != legacy_op_.end()) {
        // v1 allocation: retype the handle into the local.reducer scope.
        entry.op = it->second;
        entry.acc = WithStorageScope(buffer, buffer.name(), "local.reducer");
      } else {
        auto v2_it = legacy_v2_op_.find(buffer.get());
        if (v2_it == legacy_v2_op_.end()) {
          continue;
        }
        // v2 allocation used with legacy syntax only: keep the buffer.
        entry.op = v2_it->second;
        entry.acc = buffer;
        entry.was_v2 = true;
      }
      entry.old_buffer = buffer;
      entry.dst =
          WithStorageScope(buffer, buffer.name() + "_result", "local.fragment");
      reducers_.emplace(buffer.get(), std::move(entry));
    }

    auto result = StmtExprMutator::VisitStmt_(op).as<SBlock>().value();
    auto *p_result = result.CopyOnWrite();

    bool changed = false;
    Array<BufferVar> new_allocs;
    Map<Var, Map<String, Any>> v2_info;
    if (auto anno = p_result->annotations.Get(attr::kReducerInfoV2)) {
      if (auto as_map = anno.value().as<Map<Var, Map<String, Any>>>()) {
        v2_info = as_map.value();
      }
    }
    for (const auto &buffer : p_result->alloc_buffers) {
      auto it = reducers_.find(buffer.get());
      if (it == reducers_.end()) {
        new_allocs.push_back(buffer);
        continue;
      }
      const LegacyReducer &entry = it->second;
      ICHECK(entry.phase == Phase::kFinalized)
          << "legacy reducer `" << buffer
          << "` has no T.finalize_reducer; cannot canonicalize.";
      new_allocs.push_back(entry.acc);
      new_allocs.push_back(entry.dst);
      changed = true;
      if (entry.was_v2) {
        // The allocation and its reducer_info_v2 entry are already correct;
        // only the finalize destination is new.
        continue;
      }
      Map<String, Any> info;
      switch (entry.op) {
      case ReducerV2OpType::kSum:
        info.Set("op", String("sum"));
        break;
      case ReducerV2OpType::kMax:
        info.Set("op", String("max"));
        break;
      case ReducerV2OpType::kMin:
        info.Set("op", String("min"));
        break;
      default:
        LOG(FATAL) << "legacy (v1) reducers only support sum/max/min";
      }
      v2_info.Set(entry.acc.var(), info);
    }
    if (changed) {
      p_result->alloc_buffers = new_allocs;
      p_result->annotations.Set(attr::kReducerInfoV2, v2_info);
      p_result->annotations.erase(tl::attr::kReducerInfo);
    }
    return result;
  }

  // ---- epoch statement rewrites --------------------------------------------

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // T.clear / T.fill on a legacy reducer opens the epoch.
    if (call->op.same_as(Fill::Get())) {
      if (LegacyReducer *entry = FindReducerInRegionArg(call->args[0])) {
        ICHECK(entry->phase == Phase::kAllocated)
            << "second T.clear/T.fill on legacy reducer `" << entry->old_buffer
            << "`: one allocation supports exactly one reduction epoch.";
        PrimType dtype = entry->old_buffer->dtype;
        PrimExpr identity = ReducerV2Identity(entry->op, dtype);
        auto fill_const = TryConstValue(call->args[1].as_or_throw<PrimExpr>());
        auto identity_const = TryConstValue(identity);
        bool is_identity = fill_const.has_value() &&
                           identity_const.has_value() &&
                           *fill_const == *identity_const;
        if (!is_identity) {
          if (entry->op == ReducerV2OpType::kSum) {
            LOG(FATAL) << "legacy reducer `" << entry->old_buffer
                       << "` (op=sum) is filled with a non-zero value; its v1 "
                          "behavior depended on the physical thread count. Use "
                          "T.reducer_init(acc, value) with the v2 API "
                          "instead.";
          }
          // max/min are idempotent: a per-partial clamp equals a one-time
          // seed, so the v1 fill value is forwarded as the epoch seed
          // (normalized to the reducer dtype).
          if (fill_const.has_value()) {
            entry->seed = MakeConst(dtype, *fill_const);
          } else if (call->args[1].as_or_throw<PrimExpr>().ty() == dtype) {
            entry->seed = call->args[1].as_or_throw<PrimExpr>();
          } else {
            entry->seed =
                prim::Cast(dtype, call->args[1].as_or_throw<PrimExpr>());
          }
        }
        entry->phase = Phase::kActive;
        // The non-identity fill value (idempotent combines only) rides as
        // the epoch's logical init value on the synthesized reducer_init.
        Array<Expr> init_args = {BufferRegion::FullRegion(entry->acc)};
        if (entry->seed.defined()) {
          init_args.push_back(entry->seed.value());
        }
        return Evaluate(
            Call(PrimType::Void(), ReducerInitOp::Get(), init_args));
      }
    }
    // In-place T.finalize_reducer(acc) becomes out-of-place v2 finalize.
    if (call->op.same_as(FinalizeReducerOp::Get()) && call->args.size() == 1) {
      if (LegacyReducer *entry = FindReducerInRegionArg(call->args[0])) {
        ICHECK(entry->phase == Phase::kActive)
            << "T.finalize_reducer on legacy reducer `" << entry->old_buffer
            << "` without a preceding T.clear/T.fill.";
        entry->phase = Phase::kFinalized;
        return Evaluate(Call(PrimType::Void(), FinalizeReducerV2Op::Get(),
                             {BufferRegion::FullRegion(entry->acc),
                              BufferRegion::FullRegion(entry->dst)},
                             call->attrs)); // forwards batch
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto it = reducers_.find(op->buffer.get());
    if (it == reducers_.end()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    LegacyReducer &entry = it->second;
    ICHECK(entry.phase == Phase::kActive)
        << "store to legacy reducer `" << entry.old_buffer
        << (entry.phase == Phase::kFinalized ? "` after T.finalize_reducer."
                                             : "` before T.clear/T.fill.");
    PrimExpr contribution = MatchUpdate(entry, op);
    ICHECK(contribution.defined())
        << "unsupported store to legacy reducer `" << entry.old_buffer
        << "`: expected `acc[i] += v` (sum) or "
           "`acc[i] = T.max/T.min(acc[i], v)` matching the declared op, got "
        << GetRef<Stmt>(op)
        << ". Rewrite with the v2 API (T.reducer_update) if this is a "
           "different access pattern.";
    contribution = VisitPrimExpr(contribution);
    return Evaluate(Call(PrimType::Void(), reducer_update(),
                         {BufferLoad(entry.acc, op->indices), contribution}));
  }

  // ---- post-finalize redirection & opaqueness ------------------------------

  Expr VisitExpr_(const TensorLoadNode *op) final {
    auto it = reducers_.find(op->source.as<VarNode>());
    if (it == reducers_.end()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    const LegacyReducer &entry = it->second;
    ICHECK(entry.phase == Phase::kFinalized)
        << "read of legacy reducer `" << entry.old_buffer
        << "` before T.finalize_reducer; partial values are not readable.";
    auto indices = op->indices.Map(
        [&](const PrimExpr &index) { return VisitPrimExpr(index); });
    return BufferLoad(entry.dst, indices, op->span);
  }

  Expr VisitExpr_(const VarNode *op) final {
    auto it = reducers_.find(op);
    if (it != reducers_.end()) {
      const LegacyReducer &entry = it->second;
      ICHECK(entry.phase == Phase::kFinalized)
          << "unsupported use of legacy reducer handle `" << op->name
          << "` inside its reduction epoch.";
      return entry.dst.var();
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  // ---- helpers --------------------------------------------------------------

  LegacyReducer *FindReducerInRegionArg(const Expr &arg) {
    auto it = reducers_.find(RegionBuffer(arg));
    return it == reducers_.end() ? nullptr : &it->second;
  }

  /*! \brief Match `combine(acc[idx], v)` against the declared combine op and
   *  return the contribution `v`, or an undefined PrimExpr on mismatch. */
  PrimExpr MatchUpdate(const LegacyReducer &entry, const BufferStoreNode *op) {
    auto is_self_load = [&](const PrimExpr &expr) {
      const auto *load = expr.as<TensorLoadNode>();
      return load != nullptr &&
             load->source.as<VarNode>() == entry.old_buffer.get() &&
             StructuralEqual()(load->indices, op->indices);
    };
    auto match_binary = [&](const PrimExpr &a, const PrimExpr &b) -> PrimExpr {
      if (is_self_load(a)) {
        return b;
      }
      if (is_self_load(b)) {
        return a;
      }
      return PrimExpr();
    };
    switch (entry.op) {
    case ReducerV2OpType::kSum:
      if (const auto *add = op->value.as<prim::AddNode>()) {
        return match_binary(add->a, add->b);
      }
      break;
    case ReducerV2OpType::kMax:
      if (const auto *max = op->value.as<prim::MaxNode>()) {
        return match_binary(max->a, max->b);
      }
      break;
    case ReducerV2OpType::kMin:
      if (const auto *min = op->value.as<prim::MinNode>()) {
        return match_binary(min->a, min->b);
      }
      break;
    default:
      break; // legacy (v1) reducers only support sum/max/min
    }
    return PrimExpr();
  }

  std::unordered_map<const VarNode *, ReducerV2OpType> legacy_op_;
  std::unordered_map<const VarNode *, ReducerV2OpType> legacy_v2_op_;
  std::unordered_set<const VarNode *> v2_op_used_;
  std::unordered_map<const VarNode *, LegacyReducer> reducers_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass CanonicalizeLegacyReducer() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LegacyReducerCanonicalizer::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.CanonicalizeLegacyReducer", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.CanonicalizeLegacyReducer",
                        CanonicalizeLegacyReducer);
}

} // namespace tl
} // namespace tvm
