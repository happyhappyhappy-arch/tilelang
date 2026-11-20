#ifndef TVM_TL_ASCEND_TRANSFORM_PTO_CAST_PLAN_H_
#define TVM_TL_ASCEND_TRANSFORM_PTO_CAST_PLAN_H_

#include "op/annotated_cast.h"

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace tvm::tl::pto {
/*! Check whether a dtype is supported by the unified element
 * addressing convention: the buffer element dtype itself must not carry
 * vector lanes and must not be a packed sub-byte type (FP4 etc.).
 *
 * E4M3FN and E5M2 are supported element types; the other FP8 variants
 * (float8_e4m3 with its own TIR dtype, E8M0, FNUZ, HiF8) are not, even
 * where the PTO codegen would map them to the same target type — only the
 * two dtypes listed in IsSupportedFloat8 take part in the FP8 conversion
 * matrix. */
bool IsSupportedElementDType(PrimType dtype);

/*! The two FP8 element types the conversion matrix supports: float8_e4m3fn and
 * float8_e5m2. Deliberately narrower than the codegen's FP8 name mapping:
 * admitting a type here means its conversion directions were validated, not
 * just that a target name exists. */
bool IsSupportedFloat8(PrimType dtype);

/*! Value category of one value-Cast position.  The kind is supplied by the
 * caller (Verify from AnalyzeLaneUse, Vectorize from the converted input
 * shape); the planner never infers it from lanes. */
enum class CastValueKind {
  kScalar,
  kVectorVmi,
};

/*! Per-Cast planning context.  `logical_lanes` is validation information
 * only; `value_kind` is the dispatch key. */
struct CastContext {
  CastValueKind value_kind = CastValueKind::kScalar;
  int64_t logical_lanes = 1;
};

/*! Scalar value-Cast plan: either keep the input (identity) or rebuild an
 * ordinary scalar Cast with the validated annotations and the original span.
 * No new scalar rounding/saturation mode is introduced. */
enum class ScalarCastAction {
  kKeepInput,
  kRebuildCast,
};

struct ScalarCastPlan {
  ScalarCastAction action = ScalarCastAction::kKeepInput;
};

/*! One ordered vector conversion action. */
enum class VectorCastOp {
  /*! Same-sign widening: one logical vcvt. */
  kIntegerExtend,
  /*! Same-width signedness change / source-sign re-view: one
   * vinterpret_cast. */
  kIntegerReinterpret,
  /*! Narrowing including cross-sign: one logical vcvt with NOSAT. */
  kIntegerTruncate,
  /*! Real non-integer conversion (f16/bf16/FP8 -> f32, f32 -> f16/bf16/FP8);
   * empty options still mean a conversion, never a no-op. */
  kNonIntegerConvert,
};

struct VectorCastStep {
  VectorCastOp op = VectorCastOp::kIntegerExtend;
  PrimType from; // element type consumed by this step
  PrimType to;   // element type produced by this step
  /*! PTODSL rounding annotation value, e.g. "R"; empty when N/A. */
  std::string rounding;
  /*! PTODSL saturate annotation value, e.g. "NOSAT"; empty when N/A. */
  std::string saturate;
};

struct VectorCastPlan {
  /*! Ordered actions; an empty list is a proven identity (no conversion).
   * A non-empty list still may be a single real conversion with empty
   * options (that is not an identity). */
  std::vector<VectorCastStep> steps;
};

using CastPlan = std::variant<ScalarCastPlan, VectorCastPlan>;

/*! Decide the conversion plan for a value Cast. This is the single admission
 * point for every Cast on the vectorize path; Verify and Vectorize both pass
 * an explicit CastContext and consume the same plan, so they cannot diverge.
 *
 * Dispatch is first scalar vs VMI, then integer vs non-integer:
 *  - scalar: identity KeepInput; integer widening/narrowing and the newly
 *    admitted same-width signedness change rebuild an ordinary scalar Cast;
 *    f16/bf16 <-> f32 keep their existing scalar lowering; non-identity FP8,
 *    float<->int and f16<->bf16 stay rejected;
 *  - VMI integer: identity 0 steps; same-width cross-sign one
 *    vinterpret_cast; same-sign widening one vcvt; cross-sign widening
 *    source-sign widen then same-width vinterpret_cast; narrowing one direct
 *    vcvt NOSAT. All 36 supported pairs are represented; no consumer-layout
 *    feasibility is guessed here;
 *  - VMI non-integer: a real conversion action for f16/bf16/FP8 -> f32
 *    (empty options are not a no-op) and for f32 -> f16/bf16/FP8 with the
 *    existing rounding/saturation policy.
 *
 * `cast` carries the backend annotations written by the frontend cast
 * helpers (rounding mode, saturation flag, stochastic rbits operand). They
 * are read here — never interpreted by the expression visitors — and an
 * attribute the selected conversion cannot honour is a rejection, not a
 * silently dropped hint. When `reason` is non-null it receives a diagnostic
 * naming the category, source/target types, lanes and the offending option. */
std::optional<CastPlan> PlanValueCast(const CastOperands &cast,
                                      const CastContext &context,
                                      std::string *reason = nullptr);

} // namespace tvm::tl::pto
#endif
