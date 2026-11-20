#ifndef TILELANG_SUPPORT_POST_ORDER_VISIT_H_
#define TILELANG_SUPPORT_POST_ORDER_VISIT_H_

#include <tvm/tirx/stmt_functor.h>

#include <functional>
#include <unordered_set>

namespace tvm::tl {

// Visit executable IR in post-order, once per node. Buffer declarations carry
// type metadata that is outside the expression/statement traversal.
class PostOrderIRVisitor : public tirx::StmtExprVisitor {
public:
  explicit PostOrderIRVisitor(
      const std::function<void(const ffi::ObjectRef &)> &callback)
      : callback_(callback) {}

private:
  void VisitExpr(const Expr &expr) final {
    if (visited_.insert(expr.get()).second) {
      tirx::StmtExprVisitor::VisitExpr(expr);
      callback_(expr);
    }
  }

  void VisitStmt(const tirx::Stmt &stmt) final {
    if (visited_.insert(stmt.get()).second) {
      tirx::StmtVisitor::VisitStmt(stmt);
      callback_(stmt);
    }
  }

  void VisitBufferDef(const tirx::BufferVar &, bool) final {}
  void VisitBufferUse(const tirx::BufferVar &) final {}

  const std::function<void(const ffi::ObjectRef &)> &callback_;
  std::unordered_set<const ffi::Object *> visited_;
};

template <typename IR>
void PostOrderVisit(
    const IR &root,
    const std::function<void(const ffi::ObjectRef &)> &callback) {
  PostOrderIRVisitor visitor(callback);
  visitor(root);
}

} // namespace tvm::tl
#endif // TILELANG_SUPPORT_POST_ORDER_VISIT_H_
