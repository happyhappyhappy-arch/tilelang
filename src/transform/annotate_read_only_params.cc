/*!
 * \file annotate_read_only_params.cc
 * \brief Annotate PrimFunc parameters that are read-only (never written).
 */

#include "support/check.h"
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace tl {
using namespace tirx;
using namespace ffi;

// A buffer view can reach a parameter through a DeclBuffer, a pointer Bind,
// or an SBlock match buffer. Keep the original parameter identity across views.
class ReadWriteMarker : public StmtExprVisitor {
public:
  void VisitStmt_(const DeclBufferNode *op) final {
    BindAlias(op->buffer, op->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    BindAlias(op->var, op->value);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const SBlockNode *op) final {
    for (const auto &match : op->match_buffers) {
      BindAlias(match->buffer, match->source->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    MarkWritten(op->buffer);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_access_ptr()) && op->args.size() == 5) {
      const auto *mask = op->args[4].as<IntImmNode>();
      if (!mask || (mask->value & 2))
        MarkWritten(op->args[1]);
    } else {
      for (const Expr &arg : op->args) {
        const auto *call = arg.as<CallNode>();
        if (call && call->op.same_as(builtin::tvm_access_ptr()))
          continue;
        if (op->op.same_as(builtin::call_extern()) ||
            (call && call->op.same_as(builtin::address_of()))) {
          MarkWritten(arg);
        }
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  bool IsWritten(const Var &param) const { return written_.count(param.get()); }

private:
  const VarNode *StorageRoot(const Expr &expr) const {
    if (const auto *var = expr.as<VarNode>()) {
      auto it = aliases_.find(var);
      return it == aliases_.end() ? var : it->second;
    }
    const auto *call = expr.as<CallNode>();
    if (!call || call->args.empty())
      return nullptr;
    if (call->op.same_as(builtin::address_of())) {
      const auto *load = call->args[0].as<TensorLoadNode>();
      return load ? StorageRoot(load->source) : nullptr;
    }
    if (call->op.same_as(builtin::buffer_data()) ||
        call->op.same_as(builtin::handle_add_byte_offset()) ||
        call->op.same_as(builtin::reinterpret())) {
      return StorageRoot(call->args[0]);
    }
    if (call->op.same_as(builtin::tvm_access_ptr()) && call->args.size() == 5) {
      return StorageRoot(call->args[1]);
    }
    return nullptr;
  }

  void BindAlias(const Var &alias, const Expr &value) {
    if (const auto *root = StorageRoot(value))
      aliases_[alias.get()] = root;
  }

  void MarkWritten(const Expr &value) {
    if (const auto *root = StorageRoot(value))
      written_.insert(root);
  }

  std::unordered_map<const VarNode *, const VarNode *> aliases_;
  std::unordered_set<const VarNode *> written_;
};

static tirx::PrimFunc MarkReadOnlyParams(tirx::PrimFunc f) {
  ReadWriteMarker marker;
  marker(f->body);
  Array<IntImm> readonly_indices;
  for (size_t i = 0; i < f->params.size(); ++i) {
    const Var &param = f->params[i];
    auto scalar = param->ty.as<PrimType>();
    if ((param.as<BufferVar>() || param->ty.as<PointerTypeNode>() ||
         (scalar && scalar->MatchesCode(kDLOpaqueHandle))) &&
        !marker.IsWritten(param)) {
      readonly_indices.push_back(IntImm::Int32(static_cast<int>(i)));
    }
  }
  if (!readonly_indices.empty()) {
    f = WithAttr(std::move(f), "tl.readonly_param_indices", readonly_indices);
  }
  return f;
}

namespace transform {
using namespace tirx::transform;

Pass AnnotateReadOnlyParams() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    return MarkReadOnlyParams(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateReadOnlyParams", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.AnnotateReadOnlyParams",
                        AnnotateReadOnlyParams);
}

} // namespace transform
} // namespace tl
} // namespace tvm
