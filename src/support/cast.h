#ifndef TILELANG_SUPPORT_CAST_H_
#define TILELANG_SUPPORT_CAST_H_

#include <tvm/ffi/any.h>

namespace tvm::tl {

// TileLang accepts both object references and dynamically typed FFI values.
template <typename T> T Downcast(ffi::AnyView value) { return value.cast<T>(); }

} // namespace tvm::tl

#endif // TILELANG_SUPPORT_CAST_H_
