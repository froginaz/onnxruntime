// 01_mangling.cpp — C++ name mangling vs extern "C"
// Build: g++ -std=c++17 -O2 -c 01_mangling.cpp -o 01.o && nm -C 01.o
//
// Point: a C++ symbol name is mangled (compiler-specific); an extern "C" symbol
// keeps its plain name. That is why a stable cross-compiler / cross-language
// boundary (npu_model.h) is extern "C".

#include <string>

// (A) same add() as C++ vs C
int  cpp_add(int x, int y) { return x + y; }     // -> _Z7cpp_addii
extern "C" int c_add(int x, int y) { return x + y; }  // -> c_add

// (B) std::string in a C++ signature: mangled AND coupled to the STL ABI
void cpp_take_str(const std::string& s) { (void)s; }
std::string cpp_make_str() { return "hi"; }      // return std::string -> [abi:cxx11] tag

// (C) the same intent expressed as C ABI (POD / pointer only): stable
extern "C" void c_take_str(const char* s) { (void)s; }
extern "C" const char* c_make_str() { return "hi"; }
