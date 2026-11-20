#include "pto_cast_plan.h"
#include <set>
#include <sstream>

namespace tvm::tl::pto {

bool IsSupportedElementDType(PrimType dtype) {
  // No packed sub-byte element types (FP4 etc.), no
  // buffer-element dtypes that already carry vector lanes, and no FP64
  // (the current PTO codegen has no FP64 scalar-type mapping).
  if (dtype.bits() < 8 || dtype.lanes() != 1) {
    // FP4 (e2m1) and every other sub-byte element type land here: an FP4
    // element is 4 bits wide whether packed or not, so a separate FP4 check
    // would be unreachable.
    return false;
  }
  // The PTO codegen only maps 8/16/32-bit scalars; unusual widths (int24) and
  // FP64 (no scalar-type mapping) are rejected for buffers and arithmetic
  // alike by the width rule below, not just inside Cast planning.
  if (!dtype.MatchesCode(kDLBool) && dtype.bits() != 8 && dtype.bits() != 16 &&
      dtype.bits() != 32) {
    return false;
  }
  return dtype.MatchesCode(kDLFloat, kDLInt, kDLUInt, kDLBool) ||
         dtype.MatchesElementType(kDLBfloat, 16) || IsSupportedFloat8(dtype);
}

bool IsSupportedFloat8(PrimType dtype) {
  // Only these two take part in the FP8 conversion matrix. float8_e4m3 is a
  // distinct TIR dtype from float8_e4m3fn and is not admitted by name
  // similarity; E8M0/HiF8/FNUZ are outside the supported conversion range,
  // while FP4 is a packed sub-byte type with no unified element addressing.
  return dtype.bits() == 8 &&
         dtype.MatchesCode(kDLFloat8_e4m3fn, kDLFloat8_e5m2);
}

namespace {

/*! Cast attributes read from a Cast node's backend annotations. The
 * frontend writes them in two shapes — TIR Imm objects (the `T.cast` helper
 * lowers `round`/`sat` to StringImm/IntImm) and plain Python values (a raw
 * `annotations={...}` map) — so both spellings are accepted here. */
struct CastAttributes {
  /*! The `round` annotation was present (even when its value is empty). */
  bool has_round_key = false;
  /*! A non-empty rounding mode was requested. An empty string is the
   * frontend's way of saying "use the backend default", which is not the same
   * as an unusable mode. */
  bool has_round_mode = false;
  std::string round; // raw spelling, e.g. "rn", "R", "A"
  bool has_sat = false;
  bool sat = true;
  bool has_rbits = false;
};

/*! Read the attribute hints of a Cast. Returns false with *why set when an
 * annotation uses a shape this version cannot read; an unknown key is
 * itself a rejection, because silently ignoring an attribute would accept
 * a conversion whose semantics were never checked. */
bool ReadCastAttributes(const ffi::Map<ffi::String, ffi::Any> &annotations,
                        CastAttributes *attrs, std::string *why) {
  for (const auto &[key_ref, value] : annotations) {
    const std::string key = key_ref;
    if (key == "round") {
      attrs->has_round_key = true;
      if (const auto *imm = value.as<prim::StringImmNode>()) {
        attrs->round = imm->value;
      } else if (auto str = value.as<ffi::String>()) {
        attrs->round = std::string(*str);
      } else {
        *why = "annotation `round` must be a string";
        return false;
      }
      attrs->has_round_mode = !attrs->round.empty();
    } else if (key == "sat") {
      attrs->has_sat = true;
      if (const auto *imm = value.as<IntImmNode>()) {
        // Only 0/1 are accepted, in the TIR Imm form as well.
        if (imm->value != 0 && imm->value != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(imm->value) + ")";
          return false;
        }
        attrs->sat = imm->value != 0;
      } else if (auto flag = value.as<bool>()) {
        attrs->sat = *flag;
      } else if (auto flag = value.as<int64_t>()) {
        // Only 0/1 are accepted: any other integer is a malformed flag, not a
        // saturating request.
        if (*flag != 0 && *flag != 1) {
          *why = "annotation `sat` must be a boolean (got " +
                 std::to_string(*flag) + ")";
          return false;
        }
        attrs->sat = *flag != 0;
      } else {
        *why = "annotation `sat` must be a boolean";
        return false;
      }
    } else if (key == "rbits") {
      attrs->has_rbits = true;
    } else {
      *why = "unknown cast annotation `" + key + "`";
      return false;
    }
  }
  return true;
}

/*! Normalize a rounding spelling to the canonical PTO token. The frontend
 * aliases are PTX-style; the canonical tokens are the ones the FP8
 * conversion policy reasons about. Returns false with *why set for
 * aliases with no PTO meaning (stochastic rounding) and for tokens no
 * conversion in this version accepts. */
bool NormalizeCastRounding(const std::string &raw, std::string *token,
                           std::string *why) {
  if (raw == "rn") {
    *token = "R";
    return true;
  }
  if (raw == "rz") {
    *token = "Z";
    return true;
  }
  if (raw == "rp") {
    *token = "C";
    return true;
  }
  if (raw == "rm") {
    *token = "F";
    return true;
  }
  if (raw == "rs") {
    *why = "stochastic rounding (round=\"rs\") is not supported";
    return false;
  }
  static const std::set<std::string> kCanonical = {"R", "A", "H",
                                                   "Z", "C", "F"};
  if (kCanonical.count(raw) != 0) {
    *token = raw;
    return true;
  }
  *why = "rounding mode \"" + raw + "\" has no PTO vcvt meaning";
  return false;
}

/*! Rounding tokens the FP8 narrowing policy accepts. The conversion
 * instructions the FP8 target exposes are FP32 <-> FP8; the matrix is the
 * intersection of the PTODSL vcvt interface with that support, so the source
 * is always float32 and only the rounding modes below are expressible. */
const std::set<std::string> &Fp8AllowedRoundings() {
  static const std::set<std::string> kAllowed = {"R", "A", "H", "Z"};
  return kAllowed;
}

bool IsVmiLogicalLaneCount(int64_t lanes) {
  return lanes == 64 || lanes == 128 || lanes == 256;
}

std::string CastContextName(const CastContext &context) {
  if (context.value_kind == CastValueKind::kScalar) {
    return "scalar";
  }
  std::ostringstream oss;
  oss << "VMI, lanes=" << context.logical_lanes;
  return oss.str();
}

bool IsHalfOrBfloat16(PrimType dtype) {
  return dtype.MatchesCode(kDLFloat, kDLBfloat) && dtype.bits() == 16;
}

/*! Identity attribute policy shared by the scalar and VMI branches: no
 * conversion exists, so sat=True (the default) is normalized away, while
 * sat=False or a non-empty rounding mode would claim a conversion semantic
 * that does not exist.  An explicit empty round key means "no request". */
bool ValidateIdentityCastAttributes(const CastAttributes &attrs,
                                    const std::string &pair, std::string *why) {
  if (attrs.has_sat && !attrs.sat) {
    *why = pair + ": an identity cast does not accept sat=False";
    return false;
  }
  if (attrs.has_round_mode) {
    *why = pair + ": an identity cast does not accept a rounding mode";
    return false;
  }
  return true;
}

/*! Integer attribute policy: rounding modes are never accepted; an explicit
 * saturation flag is accepted only for narrowing and only as sat=False, which
 * the direct NOSAT lowering already honours. */
bool ValidateIntegerCastAttributes(const CastAttributes &attrs, bool narrowing,
                                   const std::string &pair, std::string *why) {
  if (attrs.has_round_mode) {
    *why = pair + ": integer conversions do not accept a rounding mode";
    return false;
  }
  if (attrs.has_sat) {
    if (!narrowing) {
      *why = pair + ": this integer conversion does not accept an explicit "
                    "saturation flag";
      return false;
    }
    if (attrs.sat) {
      *why = pair + ": integer narrowing accepts only sat=False (NOSAT), not "
                    "sat=True";
      return false;
    }
  }
  return true;
}

std::optional<CastPlan> PlanScalarValueCast(PrimType from, PrimType to,
                                            const CastAttributes &attrs,
                                            const std::string &pair,
                                            std::string *reason) {
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return std::optional<CastPlan>();
  };
  ScalarCastPlan keep;
  keep.action = ScalarCastAction::kKeepInput;
  ScalarCastPlan rebuild;
  rebuild.action = ScalarCastAction::kRebuildCast;
  if (from == to) {
    std::string why;
    if (!ValidateIdentityCastAttributes(attrs, pair, &why)) {
      return reject(why);
    }
    return CastPlan{keep};
  }
  const bool from_int = from.MatchesCode(kDLInt, kDLUInt);
  const bool to_int = to.MatchesCode(kDLInt, kDLUInt);
  if (from_int && to_int) {
    std::string why;
    if (!ValidateIntegerCastAttributes(attrs, to.bits() < from.bits(), pair,
                                       &why)) {
      return reject(why);
    }
    // Scalar integer casts keep the ordinary pto.cast/trunc lowering; the
    // validated sat=False marker is already honoured and is normalized away.
    return CastPlan{rebuild};
  }
  if (IsSupportedFloat8(from) || IsSupportedFloat8(to)) {
    return reject(pair + ": non-identity FP8 conversions have no scalar "
                         "lowering inside SIMD_VF; use a lane-varying value "
                         "so the cast takes the VMI vcvt path");
  }
  const bool from_float = from.MatchesCode(kDLFloat, kDLBfloat);
  const bool to_float = to.MatchesCode(kDLFloat, kDLBfloat);
  if (from_float && to_float) {
    if (attrs.has_round_mode) {
      return reject(pair + ": this conversion does not accept a rounding mode");
    }
    const bool from_f32 = from.MatchesCode(kDLFloat) && from.bits() == 32;
    const bool to_f32 = to.MatchesCode(kDLFloat) && to.bits() == 32;
    if (!from_f32 && to_f32) {
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      return CastPlan{rebuild};
    }
    if (from_f32 && !to_f32) {
      if (attrs.has_sat && attrs.sat) {
        return reject(pair + ": this conversion accepts only sat=False "
                             "(non-saturating)");
      }
      return CastPlan{rebuild};
    }
    return reject(pair + ": cross-family f16 <-> bf16 narrowing has no "
                         "validated rounding semantics; convert through "
                         "float32 explicitly");
  }
  return reject(pair + ": no scalar conversion exists for this dtype pair");
}

std::optional<CastPlan>
PlanVectorValueCast(PrimType from, PrimType to, const CastContext &context,
                    const CastAttributes &attrs, const std::string &round_token,
                    const std::string &pair, std::string *reason) {
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return std::optional<CastPlan>();
  };
  if (from == to) {
    std::string why;
    if (!ValidateIdentityCastAttributes(attrs, pair, &why)) {
      return reject(why);
    }
    return CastPlan{VectorCastPlan{}};
  }
  const bool from_int = from.MatchesCode(kDLInt, kDLUInt);
  const bool to_int = to.MatchesCode(kDLInt, kDLUInt);
  if (from_int && to_int) {
    std::string why;
    if (!ValidateIntegerCastAttributes(attrs, to.bits() < from.bits(), pair,
                                       &why)) {
      return reject(why);
    }
    VectorCastPlan plan;
    if (to.bits() < from.bits()) {
      plan.steps.push_back(
          {VectorCastOp::kIntegerTruncate, from, to, "", "NOSAT"});
      return CastPlan{plan};
    }
    if (to.bits() == from.bits()) {
      plan.steps.push_back(
          {VectorCastOp::kIntegerReinterpret, from, to, "", ""});
      return CastPlan{plan};
    }
    if (from.MatchesCode(kDLInt) == to.MatchesCode(kDLInt)) {
      plan.steps.push_back({VectorCastOp::kIntegerExtend, from, to, "", ""});
      return CastPlan{plan};
    }
    // Cross-sign widening: extend with the source signedness to the target
    // width, then re-view the same-width result as the target signedness.
    const PrimType intermediate = from.MatchesCode(kDLInt)
                                      ? PrimType::Int(to.bits())
                                      : PrimType::UInt(to.bits());
    plan.steps.push_back(
        {VectorCastOp::kIntegerExtend, from, intermediate, "", ""});
    plan.steps.push_back(
        {VectorCastOp::kIntegerReinterpret, intermediate, to, "", ""});
    return CastPlan{plan};
  }
  const bool from_fp8 = IsSupportedFloat8(from);
  const bool to_fp8 = IsSupportedFloat8(to);
  if (from_fp8 || to_fp8) {
    if (from_fp8 && to_fp8) {
      return reject(pair + ": conversions between the two FP8 types are not "
                           "supported (they are not a reinterpretation)");
    }
    VectorCastPlan plan;
    if (from_fp8 && to.MatchesCode(kDLFloat) && to.bits() == 32) {
      // Exact widening: a real conversion with no options.  An explicit hint
      // here cannot be honoured, so it is rejected rather than dropped.
      if (attrs.has_round_key) {
        return reject(pair + ": this direction does not accept a rounding "
                             "mode");
      }
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "", ""});
      return CastPlan{plan};
    }
    if (to_fp8) {
      if (!(from.MatchesCode(kDLFloat) && from.bits() == 32)) {
        return reject(pair + ": the FP8 conversion instructions take a "
                             "float32 source; convert to float32 explicitly "
                             "first");
      }
      const std::set<std::string> &allowed = Fp8AllowedRoundings();
      const std::string token = attrs.has_round_mode ? round_token : "R";
      if (allowed.count(token) == 0) {
        std::ostringstream oss;
        oss << pair << ": rounding mode " << token;
        if (attrs.has_round_mode && attrs.round != token) {
          oss << " (from round=\"" << attrs.round << "\")";
        }
        oss << " is not supported by the FP8 conversion policy";
        return reject(oss.str());
      }
      plan.steps.push_back({VectorCastOp::kNonIntegerConvert, from, to, token,
                            (attrs.has_sat && !attrs.sat) ? "NOSAT" : "SAT"});
      return CastPlan{plan};
    }
    return reject(pair + ": FP8 -> FP16/BF16 and FP8 <-> integer are out of "
                         "scope; convert through float32 explicitly");
  }
  const bool from_float = from.MatchesCode(kDLFloat, kDLBfloat);
  const bool to_float = to.MatchesCode(kDLFloat, kDLBfloat);
  if (from_float && to_float) {
    if (attrs.has_round_mode) {
      return reject(pair + ": this conversion does not accept a rounding mode");
    }
    const bool from_f32 = from.MatchesCode(kDLFloat) && from.bits() == 32;
    const bool to_f32 = to.MatchesCode(kDLFloat) && to.bits() == 32;
    VectorCastPlan plan;
    if (!from_f32 && to_f32) {
      if (attrs.has_sat) {
        return reject(pair + ": this direction does not accept a saturation "
                             "mode");
      }
      // A real conversion: empty options are not a no-op.
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "", ""});
      return CastPlan{plan};
    }
    if (from_f32 && !to_f32) {
      if (attrs.has_sat && attrs.sat) {
        return reject(pair + ": this conversion accepts only sat=False "
                             "(NOSAT)");
      }
      // Pin R/NOSAT so the generated vcvt never depends on PTODSL defaults.
      plan.steps.push_back(
          {VectorCastOp::kNonIntegerConvert, from, to, "R", "NOSAT"});
      return CastPlan{plan};
    }
    return reject(pair + ": cross-family f16 <-> bf16 narrowing has no "
                         "validated rounding semantics; convert through "
                         "float32 explicitly");
  }
  return reject(pair + ": float <-> integer conversions need per-combination "
                       "rounding/overflow validation");
}

} // namespace

std::optional<CastPlan> PlanValueCast(const CastOperands &cast,
                                      const CastContext &context,
                                      std::string *reason) {
  PrimType from = cast.value.ty();
  PrimType to = cast.dtype;
  auto reject = [&](const std::string &why) {
    if (reason != nullptr) {
      *reason = why;
    }
    return std::optional<CastPlan>();
  };
  std::ostringstream pair_os;
  pair_os << "`" << from << " -> " << to << "` (" << CastContextName(context)
          << ")";
  const std::string pair = pair_os.str();
  // The explicit kind is authoritative; lanes only validate it.  Inferring
  // the kind from lanes would turn a lane-1 VMI value into a scalar and a
  // scalar request with vector lanes into an implicit VMI fallback.
  if (context.value_kind == CastValueKind::kScalar) {
    if (context.logical_lanes != 1) {
      return reject(pair +
                    ": a scalar cast context must have exactly one "
                    "logical lane, got " +
                    std::to_string(context.logical_lanes));
    }
  } else {
    if (!IsVmiLogicalLaneCount(context.logical_lanes)) {
      return reject(pair +
                    ": a VMI cast context requires logical lanes "
                    "64/128/256, got " +
                    std::to_string(context.logical_lanes));
    }
  }
  // Dtype legality is a precondition for every action plan,
  // including identity — unsupported types (int24/FP64/...) must not slip
  // through the same-type fast path.  The Parallel value path is
  // element-wise, so both sides must be lane-1 elements.
  if (from.lanes() != 1 || to.lanes() != 1) {
    return reject(pair + ": value casts require lane-1 element operands");
  }
  if (!IsSupportedElementDType(from) || !IsSupportedElementDType(to)) {
    return reject(pair + " is outside the supported element dtype set");
  }

  // Attributes are parsed for every Cast, not just the FP8 ones: the
  // expression whitelist deliberately does not look inside a cast's
  // annotations (attributes are not computation values), so an annotation the
  // selected conversion cannot honour has to be rejected here — for every
  // direction — or it would be dropped silently.
  CastAttributes attrs;
  std::string why;
  if (!ReadCastAttributes(cast.attributes, &attrs, &why)) {
    return reject(pair + ": " + why);
  }
  if (attrs.has_rbits || cast.random_bits.has_value()) {
    return reject(pair + ": the rbits operand (stochastic rounding) is not "
                         "supported");
  }
  std::string round_token;
  if (attrs.has_round_mode &&
      !NormalizeCastRounding(attrs.round, &round_token, &why)) {
    return reject(pair + ": " + why);
  }

  if (context.value_kind == CastValueKind::kScalar) {
    return PlanScalarValueCast(from, to, attrs, pair, reason);
  }
  return PlanVectorValueCast(from, to, context, attrs, round_token, pair,
                             reason);
}

} // namespace tvm::tl::pto
