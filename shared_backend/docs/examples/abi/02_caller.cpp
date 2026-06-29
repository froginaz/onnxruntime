// 02_caller.cpp — calls cpp_make_str(); link it against the OTHER-ABI object
// (see run.sh) to observe the ABI-mismatch link error.
#include <string>
std::string cpp_make_str();
int main() { return (int)cpp_make_str().size(); }
