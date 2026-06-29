// 02_stl_abi_break.cpp — the SAME C++ source breaks across STL ABI versions.
//
// This single definition is compiled twice (see run.sh):
//   g++ -D_GLIBCXX_USE_CXX11_ABI=1 -c  -> symbol _Z12cpp_make_strB5cxx11v
//   g++ -D_GLIBCXX_USE_CXX11_ABI=0 -c  -> symbol _Z12cpp_make_strv
// A caller built with one ABI cannot link an object built with the other:
//   "undefined reference to `cpp_make_str[abi:cxx11]()`"
//
// Point: exposing std::string (or any STL type) at a binary boundary is unsafe
// unless BOTH sides use the exact same compiler + STL + flags. npu.h
// therefore uses const char* / POD, never std::string.

#include <string>

std::string cpp_make_str() { return "hi"; }
