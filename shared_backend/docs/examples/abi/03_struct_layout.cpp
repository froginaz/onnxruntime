// 03_struct_layout.cpp — struct layout is fragile across versions/flags.
// Build: g++ -std=c++17 03_struct_layout.cpp -o 03 && ./03
//
// Point: adding a field, changing packing, or adding `virtual` changes the
// binary layout/size. A peer built against an older layout misreads a newer
// struct -> silent corruption. Hence: opaque handles + struct_size (demo 04).

#include <cstdio>
#include <cstddef>
#include <cstdint>

struct InfoV1 { uint32_t a; uint64_t b; };              // v1
struct InfoV2 { uint32_t a; uint64_t b; uint32_t c; };  // v2: field added
#pragma pack(push, 1)
struct InfoPacked { uint32_t a; uint64_t b; };          // packing removed
#pragma pack(pop)
struct WithVtbl { virtual ~WithVtbl() {} uint32_t a; }; // virtual -> hidden vptr

int main() {
  printf("InfoV1     size=%zu  off(a)=%zu off(b)=%zu\n",
         sizeof(InfoV1), offsetof(InfoV1, a), offsetof(InfoV1, b));
  printf("InfoV2     size=%zu  off(a)=%zu off(b)=%zu off(c)=%zu\n",
         sizeof(InfoV2), offsetof(InfoV2, a), offsetof(InfoV2, b), offsetof(InfoV2, c));
  printf("InfoPacked size=%zu  off(a)=%zu off(b)=%zu  (alignment removed)\n",
         sizeof(InfoPacked), offsetof(InfoPacked, a), offsetof(InfoPacked, b));
  printf("WithVtbl   size=%zu  (one uint32 but larger due to the vptr)\n",
         sizeof(WithVtbl));
  return 0;
}
