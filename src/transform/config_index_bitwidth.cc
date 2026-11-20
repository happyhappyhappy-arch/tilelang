#include "../../3rdparty/tvm/src/tirx/ir/data_type_rewriter.h"
#include "../op/builtin.h"
#include "common/int64_promoter.h"
#include "tirx/ir_mutator_with_analyzer.h"
#include <tvm/tirx/op.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;
using namespace arith;
class ConfigIndexBitwidthRewriter : public IndexDataTypeNormalizer {
public:
  using IndexDataTypeNormalizer::IndexDataTypeNormalizer;

private:
  bool CanRewriteDType(PrimType type) const final {
    return type.MatchesCode(kDLInt) && type.bits() < 64;
  }
};

class IndexLegalizer : public IRMutatorWithAnalyzer {
public:
  using IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;
  using IRMutatorWithAnalyzer::VisitStmt;

private:
  Array<PrimExpr> Legalize(const Array<PrimExpr> &indices) {
    return indices.Map([this](const PrimExpr &i) {
      return PromoteIndexIfNeeded(i, analyzer_);
    });
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store =
        IRMutatorWithAnalyzer::VisitStmt_(op).as_or_throw<BufferStore>();
    auto indices = Legalize(store->indices);
    if (!indices.same_as(store->indices))
      store.CopyOnWrite()->indices = std::move(indices);
    return store;
  }

  Expr VisitExpr_(const TensorLoadNode *op) final {
    auto load = IRMutatorWithAnalyzer::VisitExpr_(op).as_or_throw<TensorLoad>();
    auto indices = Legalize(load->indices);
    if (!indices.same_as(load->indices))
      load.CopyOnWrite()->indices = std::move(indices);
    return load;
  }
};

tvm::transform::Pass ConfigIndexBitwidth() {
  using namespace tirx::transform;
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    if (auto bits = ctx->GetConfig<int64_t>(kConfigIndexBitwidth)) {
      f = ConfigIndexBitwidthRewriter(PrimType::Int(bits.value()))
              .Rewrite(std::move(f));
    }
    Analyzer analyzer;
    IndexLegalizer legalizer(analyzer);
    auto body = legalizer.VisitStmt(f->body);
    if (!body.same_as(f->body))
      f.CopyOnWrite()->body = std::move(body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ConfigIndexBitwidth", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.ConfigIndexBitwidth",
                        ConfigIndexBitwidth);
}

} // namespace tl
} // namespace tvm
