/*!
 * \file codegen_py.cc
 */
#include "codegen_py.h"
#include "backend/common/codegen/codegen_utils.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include <tvm/arith/analyzer.h>
#include <tvm/ir/unique_name_supply.h>

#include <cctype>

namespace tvm {
namespace codegen {

using namespace ffi;

void CodeGenTileLangPY::AddFunction(const GlobalVar &gvar, const PrimFunc &f) {
  RegisterFunction_(gvar, f);
  auto function_name = GetFunctionName_(gvar);

  // clear previous generated state.
  InitFuncState_(f);

  PrintFuncDecorator_(stream);
  PrintFunctionSignature_(function_name, f, stream);
  stream << ":\n";

  int func_scope = BeginScope();
  PreFunctionBody_(f);
  PrintStmt_(f->body);
  EndScope(func_scope);
}

std::string CodeGenTileLangPY::Finish() {
  std::ostringstream code;
  code << decl_stream.str();
  code << stream.str();
  return code.str();
}

String CodeGenTileLangPY::GetFunctionName_(const GlobalVar &gvar) {
  auto it = internal_functions_.find(gvar);
  ICHECK(it != internal_functions_.end())
      << "Attempted to find name of " << gvar
      << ", but no function with this GlobalVar has been declared";
  return it->second;
}

void CodeGenTileLangPY::RegisterFunction_(const GlobalVar &gvar,
                                          const PrimFunc &func) {
  if (internal_functions_.count(gvar)) {
    return;
  }

  auto function_name = [&]() -> String {
    if (auto global_symbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol)) {
      auto name = global_symbol.value();
      ICHECK(!func_name_supply_->ContainsName(name))
          << "Function " << gvar << " must use global symbol " << name
          << ", but this name has already been used.";
      func_name_supply_->ReserveName(name);
      return name;
    } else {
      ICHECK(!func_name_supply_->ContainsName(gvar->name_hint))
          << "Function " << gvar << " must use name hint " << gvar->name_hint
          << ", but this name has already been used.";
      func_name_supply_->ReserveName(gvar->name_hint);
      return gvar->name_hint;
    }
  }();
  internal_functions_.insert({gvar, function_name});
}

void CodeGenTileLangPY::InitFuncState_(const PrimFunc &f) {
  alloc_storage_scope_.clear();
  handle_data_type_.clear();
  CodeGenSourceBase::ClearFuncState();
  ReserveKeywordsAsUnique_();
}

void CodeGenTileLangPY::PrintFunctionSignature_(const String &function_name,
                                                const PrimFunc &func,
                                                std::ostream &os) { // NOLINT(*)
  os << "def " << function_name << "(";
  for (size_t i = 0; i < func->params.size(); ++i) {
    tirx::Var v = func->params[i];
    if (i > 0) {
      os << ", ";
    }
    os << AllocVarID(v.get());
  }
  os << ")";

  // Register handle data type
  for (const auto &param : func->params) {
    if (auto buffer = param.as<BufferVar>()) {
      RegisterHandleType_(param.get(), (*buffer)->dtype);
    } else if (auto *ptr = param->ty.as<PointerTypeNode>()) {
      if (auto *prim = ptr->element_type.as<PrimTypeNode>()) {
        RegisterHandleType_(param.get(), GetRef<PrimType>(prim));
      }
    }
  }
}

void CodeGenTileLangPY::ReserveKeywordsAsUnique_() {
  // skip the first underscore, so SSA variable starts from _1
  name_supply_->ReserveName("_");
  name_supply_->ReserveName("False");
  name_supply_->ReserveName("None");
  name_supply_->ReserveName("True");
  name_supply_->ReserveName("and");
  name_supply_->ReserveName("as");
  name_supply_->ReserveName("assert");
  name_supply_->ReserveName("async");
  name_supply_->ReserveName("await");
  name_supply_->ReserveName("break");
  name_supply_->ReserveName("class");
  name_supply_->ReserveName("continue");
  name_supply_->ReserveName("def");
  name_supply_->ReserveName("del");
  name_supply_->ReserveName("elif");
  name_supply_->ReserveName("else");
  name_supply_->ReserveName("except");
  name_supply_->ReserveName("finally");
  name_supply_->ReserveName("for");
  name_supply_->ReserveName("from");
  name_supply_->ReserveName("global");
  name_supply_->ReserveName("if");
  name_supply_->ReserveName("import");
  name_supply_->ReserveName("in");
  name_supply_->ReserveName("is");
  name_supply_->ReserveName("lambda");
  name_supply_->ReserveName("nonlocal");
  name_supply_->ReserveName("not");
  name_supply_->ReserveName("or");
  name_supply_->ReserveName("pass");
  name_supply_->ReserveName("raise");
  name_supply_->ReserveName("return");
  name_supply_->ReserveName("try");
  name_supply_->ReserveName("while");
  name_supply_->ReserveName("with");
  name_supply_->ReserveName("yield");

  name_supply_->ReserveName("void");
  name_supply_->ReserveName("int");
  name_supply_->ReserveName("float");
  name_supply_->ReserveName("double");
  name_supply_->ReserveName("char");
  name_supply_->ReserveName("unsigned");
  name_supply_->ReserveName("short");
  name_supply_->ReserveName("long");

  name_supply_->ReserveName("cutlass");
  name_supply_->ReserveName("cute");
  name_supply_->ReserveName("tl");
}

void CodeGenTileLangPY::PrintSSAAssign(const std::string &target,
                                       const std::string &src, const Type &t) {
  PrintIndent();
  stream << target << " = " << RemoveOutermostParentheses(src) << "\n";
}

void CodeGenTileLangPY::PrintType(const PrimType &type,
                                  std::ostream &os) { // NOLINT(*)
  if (type.MatchesCode(kDLFloat)) {
    if (type.bits() == 16 || type.bits() == 32 || type.bits() == 64) {
      os << "float";
    } else {
      LOG(FATAL) << "Cannot convert float" << type.bits() << " to Python type";
    }
  } else if (type.MatchesCode(kDLUInt)) {
    switch (type.bits()) {
    case 8:
    case 16:
    case 32:
    case 64: {
      os << "int";
      break;
    }
    case 1:
      os << "bool";
      break;
    default:
      LOG(FATAL) << "Cannot convert uint" << type.bits() << " to Python type";
    }
  } else if (type.MatchesCode(kDLInt)) {
    switch (type.bits()) {
    case 8:
    case 16:
    case 32:
    case 64: {
      os << "int";
      break;
    }
    case 1:
      os << "bool";
      break;
    default:
      LOG(FATAL) << "Cannot convert int" << type.bits() << " to Python type";
    }
  } else {
    LOG(FATAL) << "Cannot convert type " << type << " to Python type";
  }
}

void CodeGenTileLangPY::VisitExpr_(const VarNode *op,
                                   std::ostream &os) { // NOLINT(*)
  os << GetVarID(op);
}

void CodeGenTileLangPY::VisitExpr_(const IntImmNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (op->ty.as_or_throw<PrimType>() == PrimType::Bool()) {
    os << (op->value ? "True" : "False");
  } else {
    std::ostringstream temp;
    temp << op->value;
    MarkConst(temp.str());
    os << temp.str();
  }
}

void CodeGenTileLangPY::VisitExpr_(const FloatImmNode *op,
                                   std::ostream &os) { // NOLINT(*)
  switch (op->ty.as_or_throw<PrimType>().bits()) {
  case 64:
  case 32: {
    std::ostringstream temp;
    temp << "float.fromhex('" << FlexibleHexFormat(op->value) << "')";
    MarkConst(temp.str());
    os << temp.str();
    break;
  }
  case 16: {
    PrintType(op->ty.as_or_throw<PrimType>(), os);
    os << "(float.fromhex('" << FlexibleHexFormat(op->value) << "'))";
    break;
  }
  default:
    LOG(FATAL) << "Bad bit-width for float: " << op->ty.as_or_throw<PrimType>()
               << "\n";
  }
}

void CodeGenTileLangPY::VisitExpr_(const prim::StringImmNode *op,
                                   std::ostream &os) { // NOLINT(*)
  EscapeStringLiteral_(op->value, os);
}

void CodeGenTileLangPY::VisitExpr_(const prim::CastNode *op,
                                   std::ostream &os) { // NOLINT(*)
  std::stringstream value;
  PrintExpr_(op->value, value);
  os << CastFromTo_(value.str(), op->value.ty(),
                    op->ty.as_or_throw<PrimType>());
}

void CodeGenTileLangPY::VisitExpr_(const prim::AddNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("+", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::SubNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("-", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::MulNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("*", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::DivNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (op->ty.as_or_throw<PrimType>().MatchesCode(kDLInt) ||
      op->ty.as_or_throw<PrimType>().MatchesCode(kDLUInt)) {
    PrintBinaryExpr_("//", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
  } else {
    PrintBinaryExpr_("/", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
  }
}
void CodeGenTileLangPY::VisitExpr_(const prim::ModNode *op,
                                   std::ostream &os) { // NOLINT(*)
  ICHECK(op->ty.as_or_throw<PrimType>().MatchesCode(kDLInt) ||
         op->ty.as_or_throw<PrimType>().MatchesCode(kDLUInt) ||
         op->ty.as_or_throw<PrimType>().MatchesCode(kDLFloat))
      << "Expected floating point or integer dtype in Mod, but got "
      << op->ty.as_or_throw<PrimType>();
  PrintBinaryExpr_("%", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}

void CodeGenTileLangPY::VisitExpr_(const prim::MinNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("min", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::MaxNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("max", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::EQNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("==", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::NENode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("!=", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::LTNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("<", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::LENode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("<=", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::GTNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_(">", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::GENode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_(">=", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::AndNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("and", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::OrNode *op,
                                   std::ostream &os) { // NOLINT(*)
  PrintBinaryExpr_("or", op->ty.as_or_throw<PrimType>(), op->a, op->b, os);
}
void CodeGenTileLangPY::VisitExpr_(const prim::NotNode *op,
                                   std::ostream &os) { // NOLINT(*)
  os << "(not ";
  PrintExpr_(op->a, os);
  os << ")";
}

void CodeGenTileLangPY::VisitExpr_(const prim::SelectNode *op,
                                   std::ostream &os) { // NOLINT(*)
  os << "(";
  PrintExpr_(op->true_value, os);
  os << " if ";
  PrintExpr_(op->condition, os);
  os << " else ";
  PrintExpr_(op->false_value, os);
  os << ")";
}

void CodeGenTileLangPY::VisitExpr_(const prim::RampNode *op,
                                   std::ostream &os) { // NOLINT(*)
  int lanes = op->ty.as_or_throw<PrimType>().lanes();
  os << "(";
  for (int i = 0; i < lanes; i++) {
    os << "(" << PrintExpr_(op->base) << ")"
       << "+(" << PrintExpr_(op->stride) << "*" << i << ")";
    if (i != lanes - 1)
      os << ", ";
  }
  os << ")";
}

void CodeGenTileLangPY::VisitExpr_(const CallNode *op,
                                   std::ostream &os) { // NOLINT(*)
  if (auto opt_call_op = op->op.as<Op>()) {
    const auto &call_op = opt_call_op.value();

    if (op->op.same_as(builtin::buffer_data())) {
      PrintExpr_(op->args[0], os);
    } else if (op->op.same_as(builtin::continue_loop())) {
      os << "continue";
    } else if (op->op.same_as(builtin::break_loop())) {
      os << "break";
    } else if (op->op.same_as(builtin_call_extern_) ||
               op->op.same_as(builtin_call_pure_extern_)) {
      ICHECK_GE(op->args.size(), 1U);
      auto func = op->args[0].as_or_throw<prim::StringImm>();
      PrintCallExtern_(op->ty, func->value, op->args, true, os);
    } else if (op_attr_global_symbol_.count(call_op)) {
      // call extern if the op itself have a global symbol.
      PrintCallExtern_(op->ty, op_attr_global_symbol_[call_op], op->args, false,
                       os);
    } else if (op->op.same_as(builtin::large_uint_imm())) {
      ICHECK_EQ(op->args.size(), 2U);
      uint64_t low =
          static_cast<uint64_t>(op->args[0].as_or_throw<IntImm>()->value);
      uint64_t high =
          static_cast<uint64_t>(op->args[1].as_or_throw<IntImm>()->value);
      uint64_t val = (high << 32U) | low;

      if (op->ty.as_or_throw<PrimType>() == PrimType::UInt(32)) {
        std::ostringstream temp;
        temp << val;
        MarkConst(temp.str());
        os << temp.str();
      } else {
        PrintType(op->ty.as_or_throw<PrimType>(), os);
        os << "(" << val << ")";
      }
    } else if (op->op.same_as(prim::builtin::bitwise_and())) {
      PrintBinaryIntrinsic_(op, "&", os);
    } else if (op->op.same_as(prim::builtin::bitwise_or())) {
      PrintBinaryIntrinsic_(op, "|", os);
    } else if (op->op.same_as(prim::builtin::bitwise_xor())) {
      PrintBinaryIntrinsic_(op, "^", os);
    } else if (op->op.same_as(prim::builtin::bitwise_not())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "~";
      PrintExpr_(op->args[0], os);
    } else if (op->op.same_as(prim::builtin::shift_left())) {
      PrintBinaryIntrinsic_(op, "<<", os);
    } else if (op->op.same_as(prim::builtin::shift_right())) {
      PrintBinaryIntrinsic_(op, ">>", os);
    } else if (op->op.same_as(prim::builtin::if_then_else())) {

      std::string cond = PrintExpr_(op->args[0]);
      std::string true_val = PrintExpr_(op->args[1]);
      std::string false_val = PrintExpr_(op->args[2]);
      os << "(" << true_val << " if " << cond << " else " << false_val << ")";
    } else if (op->op.same_as(builtin::isnullptr())) {
      ICHECK_EQ(op->args.size(), 1U);
      os << "(";
      PrintExpr_(op->args[0], os);
      os << " is None)";
    } else if (op->op.same_as(builtin::isnan())) {
      os << "(";
      PrintExpr_(op->args[0], os);
      os << " != ";
      PrintExpr_(op->args[0], os);
      os << ")";
    } else {
      LOG(FATAL) << "Unresolved call " << op->op;
    }
  } else if (auto opt = op->op.as<GlobalVar>()) {
    const auto &gvar = opt.value();
    auto callee_name = GetFunctionName_(gvar);
    PrintCallExtern_(op->ty, callee_name, op->args, false, os);
  } else {
    LOG(FATAL)
        << "CodeGenTileLangPY: Unknown operation " << op->op
        << " is neither a recognized built-in, "
        << "nor a GlobalVar reference to another function in the IRModule";
  }
}

void CodeGenTileLangPY::VisitExpr_(const TensorLoadNode *op,
                                   std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(op->indices.size(), 1)
      << "Load from non-flat memory not supported.";
  const BufferVar buffer = op->source.as_or_throw<BufferVar>();

  PrimType value_dtype = op->ty.as_or_throw<PrimType>();
  PrimExpr index = op->indices[0];
  PrimType element_dtype = buffer->dtype;

  ICHECK_EQ(value_dtype, element_dtype)
      << "value_dtype and element_dtype must be same for a TensorLoadNode";
  std::string ref =
      GetBufferRef_(op->ty.as_or_throw<PrimType>(), buffer.get(), index);
  os << ref;
}

void CodeGenTileLangPY::VisitStmt_(const BufferStoreNode *op) {
  ICHECK_EQ(op->indices.size(), 1) << "Store to non-flat memory not supported.";

  PrimType value_dtype = op->value.ty();
  PrimType element_dtype = op->buffer->dtype;
  PrimExpr index_expr = op->indices[0];

  ICHECK_EQ(value_dtype, element_dtype)
      << "value_dtype and element_dtype must be same for a BufferStoreNode";
  std::string value = PrintExpr_(op->value);
  std::string ref = GetBufferRef_(value_dtype, op->buffer.get(), index_expr);
  PrintIndent();
  stream << ref << " = " << RemoveOutermostParentheses(value) << "\n";
}

void CodeGenTileLangPY::VisitStmt_(const DeclBufferNode *op) {
  // The declaration binds a typed buffer view to its backing storage.
  std::string source = PrintExpr_(op->data);
  PrintIndent();
  stream << AllocVarID(op->buffer.get()) << " = " << source << "\n";
  alloc_storage_scope_[op->buffer.get()] = op->buffer.scope();
  RegisterHandleType_(op->buffer.get(), op->buffer->dtype);
}

void CodeGenTileLangPY::VisitStmt_(const BindNode *op) {
  std::string value = PrintExpr_(op->value);
  PrintIndent();
  stream << AllocVarID(op->var.get()) << " = " << value << "\n";
}

void CodeGenTileLangPY::VisitStmt_(const AllocBufferNode *op) {
  std::string vid = AllocVarID(op->buffer.get());

  PrintIndent();
  auto alloc_ref = GetRef<AllocBuffer>(op);
  auto opt_size = alloc_ref.ConstantAllocationSize();
  ICHECK(opt_size.has_value())
      << "Can only handle constant size stack allocation for now";
  size_t constant_size = static_cast<size_t>(opt_size.value());

  auto scope = op->buffer.scope();
  alloc_storage_scope_[op->buffer.get()] = scope;

  stream << vid << " = [None] * " << constant_size << "\n";

  RegisterHandleType_(op->buffer.get(), op->buffer->dtype);
}

void CodeGenTileLangPY::VisitStmt_(const AttrStmtNode *op) {
  PrintStmt_(op->body);
}

void CodeGenTileLangPY::VisitStmt_(const ForNode *op) {
  PrintIndent();
  std::string vid = AllocVarID(op->loop_var.get());
  stream << "for " << vid << " in range(";
  if (is_zero(op->min) && !op->step) {
    PrintExpr_(op->extent, stream);
  } else {
    PrintExpr_(op->min, stream);
    stream << ", ";
    PrimExpr upper_bound = arith::Analyzer()->Simplify(op->extent + op->min);
    PrintExpr_(upper_bound, stream);
  }
  if (op->step) {
    stream << ", ";
    PrintExpr_(*op->step, stream);
  }
  stream << "):\n";
  int for_scope = BeginScope();
  PrintStmt_(op->body);
  EndScope(for_scope);
}

void CodeGenTileLangPY::VisitStmt_(const WhileNode *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "while " << RemoveOutermostParentheses(cond) << ":\n";
  int while_scope = BeginScope();
  PrintStmt_(op->body);
  EndScope(while_scope);
}

void CodeGenTileLangPY::VisitStmt_(const IfThenElseNode *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  stream << "if " << RemoveOutermostParentheses(cond) << ":\n";
  int then_scope = BeginScope();
  PrintStmt_(op->then_case);
  EndScope(then_scope);

  if (op->else_case) {
    PrintIndent();
    stream << "else:\n";
    int else_scope = BeginScope();
    PrintStmt_(op->else_case.value());
    EndScope(else_scope);
  }
}

void CodeGenTileLangPY::VisitStmt_(const SeqStmtNode *op) {
  for (Stmt stmt : op->seq) {
    PrintStmt_(stmt);
  }
}

void CodeGenTileLangPY::VisitStmt_(const ReturnNode *op) {
  PrintIndent();
  stream << "return " << RemoveOutermostParentheses(PrintExpr_(op->value))
         << "\n";
}

void CodeGenTileLangPY::VisitStmt_(const BreakNode *) {
  PrintIndent();
  stream << "break\n";
}

void CodeGenTileLangPY::VisitStmt_(const ContinueNode *) {
  PrintIndent();
  stream << "continue\n";
}

void CodeGenTileLangPY::VisitStmt_(const EvaluateNode *op) {
  if (op->value.as<IntImmNode>())
    return;

  std::string vid = PrintExpr_(op->value);
  if (!vid.empty()) {
    PrintIndent();
    stream << vid << "\n";
  }
}

void CodeGenTileLangPY::VisitStmt_(const AssertStmtNode *op) {
  std::string cond = PrintExpr_(op->condition);
  PrintIndent();
  if (!op->message_parts.empty()) {
    std::string joined_msg;
    for (const auto &part : op->message_parts) {
      joined_msg += part->value;
    }
    stream << "assert " << cond << ", ";
    EscapeStringLiteral_(joined_msg, stream);
    stream << "\n";
  } else {
    stream << "assert " << cond << "\n";
  }
}

std::string CodeGenTileLangPY::CastFromTo_(const std::string &value,
                                           PrimType from, PrimType target) {
  if (from == target)
    return value;
  std::ostringstream os;
  PrintType(target, os);
  os << "(" << value << ")";
  return os.str();
}

void CodeGenTileLangPY::PrintBinaryExpr_(const std::string &opstr,
                                         PrimType dtype, PrimExpr lhs,
                                         PrimExpr rhs,
                                         std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(dtype.lanes(), 1);
  if (isalpha(opstr[0]) && opstr != "and" && opstr != "or") {
    os << opstr << '(';
    PrintExpr_(lhs, os);
    os << ", ";
    PrintExpr_(rhs, os);
    os << ')';
  } else {
    os << '(';
    PrintExpr_(lhs, os);
    os << ' ' << opstr << ' ';
    PrintExpr_(rhs, os);
    os << ')';
  }
}

void CodeGenTileLangPY::PrintBinaryIntrinsic_(const CallNode *op,
                                              const char *opstr,
                                              std::ostream &os) { // NOLINT(*)
  ICHECK_EQ(op->ty.as_or_throw<PrimType>().lanes(), 1);
  ICHECK_EQ(op->args.size(), 2U);
  os << '(';
  PrintExpr_(op->args[0], os);
  os << ' ' << opstr << ' ';
  PrintExpr_(op->args[1], os);
  os << ')';
}

void CodeGenTileLangPY::PrintCallExtern_(Type ret_type, String global_symbol,
                                         const Array<Expr> &args,
                                         bool skip_first_arg,
                                         std::ostream &os) { // NOLINT(*)
  os << global_symbol << "(";
  for (size_t i = static_cast<size_t>(skip_first_arg); i < args.size(); ++i) {
    PrintExpr_(args[i], os);
    if (i < args.size() - 1) {
      os << ", ";
    }
  }
  os << ")";
}

// Print a reference expression to a buffer.
std::string CodeGenTileLangPY::GetBufferRef_(PrimType t, const VarNode *buffer,
                                             PrimExpr index) {
  const VarNode *buffer_var = buffer;
  std::string vid = GetVarID(buffer_var);
  PrimType buffer_element_dtype = buffer->ty.as_or_throw<BufferType>()->dtype;

  ICHECK(HandleTypeMatch_(buffer_var, buffer_element_dtype));
  ICHECK_EQ(t, buffer_element_dtype);

  std::string index_str = PrintExpr_(index);
  return vid + "[" + index_str + "]";
}

void CodeGenTileLangPY::RegisterHandleType_(const VarNode *buf_var,
                                            PrimType t) {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end()) {
    handle_data_type_.emplace(buf_var, t);
  } else {
    ICHECK(it->second == t) << "conflicting buf var type";
  }
}

bool CodeGenTileLangPY::HandleTypeMatch_(const VarNode *buf_var,
                                         PrimType t) const {
  auto it = handle_data_type_.find(buf_var);
  if (it == handle_data_type_.end())
    return false;
  return it->second == t;
}

void CodeGenTileLangPY::EscapeStringLiteral_(const std::string &s,
                                             std::ostream &os) {
  os << '"';
  for (unsigned char c : s) {
    switch (c) {
    case '\\':
      os << "\\\\";
      break;
    case '"':
      os << "\\\"";
      break;
    case '\n':
      os << "\\n";
      break;
    case '\r':
      os << "\\r";
      break;
    case '\t':
      os << "\\t";
      break;
    case '\f':
      os << "\\f";
      break;
    case '\b':
      os << "\\b";
      break;
    default:
      // Handle non-printable and non-ASCII characters
      if (c < 32 || c == 127) {
        // Output as \xHH
        os << "\\x";
        const char hex[] = "0123456789abcdef";
        os << hex[(c >> 4) & 0xF];
        os << hex[c & 0xF];
      } else {
        os << c;
      }
      break;
    }
  }
  os << '"';
}

} // namespace codegen
} // namespace tvm
