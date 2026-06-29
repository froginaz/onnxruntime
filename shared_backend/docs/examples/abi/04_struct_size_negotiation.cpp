// 04_struct_size_negotiation.cpp — the C-ABI fix for demo 03.
// Build: g++ -std=c++17 04_struct_size_negotiation.cpp -o 04 && ./04
//
// Point: put `struct_size` (+ api_version) at the front of an extensible POD.
// A library built with the new struct can detect an old caller (smaller
// struct_size) and default the new fields -> forward/backward compatibility.
// This is exactly the npu_model_load_info_t pattern in npu.h.

#include <cstdio>
#include <cstdint>
#include <cstddef>

typedef struct { uint32_t struct_size; uint32_t api_version; uint32_t a; }            LoadInfoV1; // old header
typedef struct { uint32_t struct_size; uint32_t api_version; uint32_t a; uint32_t b_new; } LoadInfoV2; // new header (field added)

// Library is built with the NEW struct; it negotiates via struct_size.
extern "C" int load(const void* info_ptr) {
  const LoadInfoV2* p = (const LoadInfoV2*)info_ptr;
  if (p->struct_size < offsetof(LoadInfoV2, b_new) + sizeof(p->b_new)) {
    printf("  [lib] caller is OLD (size=%u) -> b_new defaulted to 0; a=%u\n", p->struct_size, p->a);
    return 0;
  }
  printf("  [lib] caller is NEW (size=%u) -> a=%u b_new=%u\n", p->struct_size, p->a, p->b_new);
  return 0;
}

int main() {
  LoadInfoV1 v1{ sizeof(LoadInfoV1), 1, 42 };
  printf("OLD caller calls NEW lib:\n"); load(&v1);
  LoadInfoV2 v2{ sizeof(LoadInfoV2), 2, 42, 7 };
  printf("NEW caller calls NEW lib:\n"); load(&v2);
  return 0;
}
