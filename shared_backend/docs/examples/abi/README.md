# Why `npu_model.h` is a C ABI but `npu_api.h` is C++ — runnable proof

Four tiny experiments that show, with real compiler output, *why* a stable
cross-compiler / cross-language / cross-version boundary must be a **C ABI**
(`npu_model.h`), while an intra-build C++ convenience layer can stay **C++**
(`npu_api.h`). Background prose: [`../../design.md`](../../design.md) §2,
[`../../npu_api.md`](../../npu_api.md).

```bash
./run.sh        # needs g++ + nm (GCC/binutils); Linux/macOS
```

---

## 01 — Name mangling: C++ vs `extern "C"`
`g++ -c 01_mangling.cpp` then `nm`:

```
_Z7cpp_addii                          ← C++  cpp_add(int,int)     (compiler-specific)
c_add                                 ← extern "C"  -> plain "c_add"
_Z12cpp_take_strRK...basic_string...  ← std::string& arg, mangled + STL-coupled
_Z12cpp_make_strB5cxx11v              ← returns std::string -> [abi:cxx11] tag
c_take_str / c_make_str               ← C ABI: plain names
```

**Lesson:** a C symbol keeps its plain name (any compiler/language/FFI can find
it); a C++ symbol is mangled differently per compiler (Itanium `_Z...` vs MSVC
`?...`). Stable boundary ⇒ `extern "C"`.

---

## 02 — The decisive one: the *same* C++ source breaks across STL ABIs
The one file `02_stl_abi_break.cpp` compiled twice:

```
new ABI (_GLIBCXX_USE_CXX11_ABI=1): _Z12cpp_make_strB5cxx11v
old ABI (_GLIBCXX_USE_CXX11_ABI=0): _Z12cpp_make_strv
```
Linking a new-ABI caller against the old-ABI object:
```
undefined reference to `cpp_make_str[abi:cxx11]()'
```

**Lesson:** identical C++ source produces *different symbols* under different STL
ABIs (libstdc++ vs libc++ vs MSVC STL; GCC4 vs GCC5+). Exposing `std::string`
(or any STL type) at a binary boundary is unsafe unless **both sides use the
exact same compiler + STL + flags**. → `npu_model.h` uses `const char*`/POD,
never `std::string`. This is the single most important reason.

---

## 03 — Struct layout is fragile
```
InfoV1     size=16   off(a)=0 off(b)=8
InfoV2     size=24   off(a)=0 off(b)=8 off(c)=16     ← added a field
InfoPacked size=12   off(a)=0 off(b)=4               ← #pragma pack(1)
WithVtbl   size=16                                   ← one uint32, but a hidden vptr (virtual)
```

**Lesson:** adding a field, changing packing, or adding `virtual` changes the
binary layout/size. A peer built against an older layout misreads a newer struct
→ **silent corruption**.

---

## 04 — The C-ABI fix: `struct_size` negotiation (the `npu_model.h` pattern)
```
OLD caller calls NEW lib:  [lib] caller is OLD (size=12) -> b_new defaulted to 0; a=42
NEW caller calls NEW lib:  [lib] caller is NEW (size=16) -> a=42 b_new=7
```

**Lesson:** put `struct_size` (+ `api_version`) at the front of an extensible
POD. The library detects an old caller (smaller `struct_size`) and defaults the
new fields → forward/backward compatibility. Combined with **opaque handles**
(`npu_model_t*`), the internal layout is hidden entirely. This is exactly
`npu_model_load_info_t` in `npu_model.h`.

---

## Conclusion — the decision rule, proven

| Boundary property (shown by) | Choice |
|---|---|
| crossed by other compiler/language/separate build (01, 02), or version-sensitive POD crosses it (03) | **C ABI** + `struct_size`/`api_version` + opaque handle → `npu_model.h` |
| only same-build C++ adapters call it, passing **opaque pointers + scalars** (none of 01–03's hazards arise) | **C++** (namespace, `enum class`) → `npu_api.h` |

> `npu_api.h` is safe as C++ precisely because it never creates the situations in
> demos 02 (no `std::string` across the boundary) and 03 (no exposed struct) —
> and its callers share the core's toolchain. `npu_model.h` cannot assume that,
> so it is hardened as C.
