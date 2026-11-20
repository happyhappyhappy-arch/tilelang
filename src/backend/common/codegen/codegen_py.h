/*!
 * \file codegen_py.h
 * \brief Common utilities to generate simple Python code.
 */
#ifndef TVM_TL_TARGET_CODEGEN_PY_H_
#define TVM_TL_TARGET_CODEGEN_PY_H_

#include "support/check.h"
#include <tvm/ir/op.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/target/codegen.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>

#include <string>
#include <unordered_map>

// from tvm/src/
#include "target/source/codegen_source_base.h"
#include "tirx/transform/ir_utils.h"

namespace tvm {
namespace codegen {

using namespace tirx;
using namespace ffi;
/*!
 * \brief A base class to generate simple Python code.
 */
class CodeGenTileLangPY
    : public ExprFunctor<void(const Expr &, std::ostream &)>,
      public StmtFunctor<void(const Stmt &)>,
      public CodeGenSourceBase {
public:
  /*!
   * \brief Add the function definition to the generated module.
   * \param gvar The GlobalVar representing the function.
   * \param func The function to be compiled.
   */
  virtual void AddFunction(const GlobalVar &gvar, const PrimFunc &func);

  /*!
   * \brief Finalize the compilation and return the code.
   * \return The code.
   */
  virtual std::string Finish();

protected:
  /*!
   * \brief Get the name of a declared function
   * \param gvar The GlobalVar of the function
   * \returns The string name of the function
   */
  String GetFunctionName_(const GlobalVar &gvar);

  /*!
   * \brief Reserve the function name in the generated module.
   *
   * \param gvar The GlobalVar representing the function.
   * \param func The function to be compiled.
   * \param whether to append return 0 in the end.
   */
  virtual void RegisterFunction_(const GlobalVar &gvar, const PrimFunc &func);

  /*!
   * \brief Initialize codegen state for generating f.
   * \param f The function to be compiled.
   */
  virtual void InitFuncState_(const PrimFunc &f);

  /*! \brief Print the function signature before ":"
   * \param function_name The name of the function
   * \param func The function whose signature should be printed
   * \param os The output stream
   */
  virtual void PrintFunctionSignature_(const String &function_name,
                                       const PrimFunc &func,
                                       std::ostream &os); // NOLINT(*)

  /*!
   * \brief Print the function decorator
   * \param os The output stream
   */
  virtual void PrintFuncDecorator_(std::ostream &os) {} // NOLINT(*)

  /*!
   * \brief Insert statement before function body.
   * \param f The function to be compiled.
   */
  virtual void PreFunctionBody_(const PrimFunc &f) {}

protected:
  /*! \brief reserves common Python keywords */
  void ReserveKeywordsAsUnique_();

  void PrintSSAAssign(const std::string &target, const std::string &src,
                      const Type &t) override;

protected:
  /*!
   * \brief Print the type representation.
   * \param t The type representation.
   * \param os The output stream
   */
  void PrintType(const PrimType &type, std::ostream &os) override; // NOLINT(*)

  /*!
   * \brief Print the Stmt n to CodeGenTileLangPY->stream
   * \param n The statement to be printed.
   */
  void PrintStmt_(const Stmt &n) { VisitStmt(n); }
  /*!
   * \brief Print the expression n into os
   * \param n The expression to be printed.
   * \param os The output stream
   */
  void PrintExpr_(const Expr &n, std::ostream &os) { // NOLINT(*)
    VisitExpr(n, os);
  }
  /*!
   * \brief Same as PrintExpr_, but simply returns result string
   * \param n The expression to be printed.
   */
  std::string PrintExpr_(const Expr &n) {
    std::ostringstream os;
    PrintExpr_(n, os);
    return os.str();
  }

  // expression
  void VisitExpr_(const VarNode *op, std::ostream &os) override;    // NOLINT(*)
  void VisitExpr_(const IntImmNode *op, std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::StringImmNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::CastNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::AddNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::SubNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::MulNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::DivNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::ModNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::MinNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::MaxNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::EQNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::NENode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::LTNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::LENode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::GTNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::GENode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::AndNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::OrNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::NotNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::SelectNode *op,
                  std::ostream &os) override; // NOLINT(*)
  void VisitExpr_(const prim::RampNode *op,
                  std::ostream &os) override;                       // NOLINT(*)
  void VisitExpr_(const CallNode *op, std::ostream &os) override;   // NOLINT(*)
  void VisitExpr_(const TensorLoadNode *op,
                  std::ostream &os) override; // NOLINT(*)

  // statement
  void VisitStmt_(const BufferStoreNode *op) override;
  void VisitStmt_(const DeclBufferNode *op) override;
  void VisitStmt_(const BindNode *op) override;
  void VisitStmt_(const AllocBufferNode *op) override;
  void VisitStmt_(const AttrStmtNode *op) override;
  void VisitStmt_(const ForNode *op) override;
  void VisitStmt_(const WhileNode *op) override;
  void VisitStmt_(const IfThenElseNode *op) override;
  void VisitStmt_(const SeqStmtNode *op) override;
  void VisitStmt_(const EvaluateNode *op) override;
  void VisitStmt_(const ReturnNode *op) override;
  void VisitStmt_(const BreakNode *op) override;
  void VisitStmt_(const ContinueNode *op) override;
  void VisitStmt_(const AssertStmtNode *op) override;

protected:
  // Get a string of type casting
  virtual std::string CastFromTo_(const std::string &value, PrimType from,
                                  PrimType target);

  virtual void PrintBinaryExpr_(const std::string &opstr, PrimType dtype,
                                PrimExpr lhs, PrimExpr rhs,
                                std::ostream &os); // NOLINT(*)
  virtual void PrintBinaryIntrinsic_(const CallNode *op, const char *opstr,
                                     std::ostream &os); // NOLINT(*)

  /*!
   * \brief Print external function call.
   * \param ret_type The return type.
   * \param global_symbol The symbolic name of the target function.
   * \param args The arguments to the function.
   * \param skip_first_arg Whether to skip the first arguments.
   * \param os The output stream.
   */
  virtual void PrintCallExtern_(Type ret_type, String global_symbol,
                                const Array<Expr> &args, bool skip_first_arg,
                                std::ostream &os); // NOLINT(*)

  // Print reference to a buffer as type t in index.
  virtual std::string GetBufferRef_(PrimType t, const VarNode *buffer,
                                    PrimExpr index);

  /*!
   * \brief Register the data type of buf_var
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  void RegisterHandleType_(const VarNode *buf_var, PrimType t);

  /*!
   * \brief If buffer is allocated as type t.
   * \param buf_var The buffer variable.
   * \param t The type to be checked.
   */
  bool HandleTypeMatch_(const VarNode *buf_var, PrimType t) const;

protected:
  /*! \brief the storage scope of allocation */
  std::unordered_map<const VarNode *, std::string> alloc_storage_scope_;

  /*! \brief Record of ops that have pre-defined global symbol. */
  OpAttrMap<TGlobalSymbol> op_attr_global_symbol_ =
      Op::GetAttrMap<TGlobalSymbol>("TGlobalSymbol");

  // cache commonly used ops
  const Op &builtin_call_extern_ = builtin::call_extern();
  const Op &builtin_call_pure_extern_ = builtin::call_pure_extern();

private:
  /*! \brief the data type of allocated buffers */
  std::unordered_map<const VarNode *, PrimType> handle_data_type_;

  /* \brief Map of GlobalVar to their symbol.
   *
   * For externally-exposed functions, this is given by the
   * tvm::attr::kTarget attribute of the PrimFunc.  For internal
   * functions, this is the name of the function's GlobalVar, possibly
   * altered to prevent duplicate names.
   */
  std::unordered_map<GlobalVar, String> internal_functions_;

  /* \brief Name supply to generate unique function names */
  UniqueNameSupply func_name_supply_;

  /*!
   * \brief Escape a string to be a valid Python double-quoted string literal.
   * \param s The input string to escape.
   * \param os The output stream to write the escaped string to.
   */
  void EscapeStringLiteral_(const std::string &s, std::ostream &os);
};

} // namespace codegen
} // namespace tvm
#endif // TVM_TL_TARGET_CODEGEN_PY_H_
