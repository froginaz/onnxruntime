#!/usr/bin/env bash
# Runnable demos for the C/C++ ABI boundary rationale (npu_model.h C ABI vs
# npu_api.h C++). Requires g++ and nm (GCC/binutils). Linux/macOS.
set -u
cd "$(dirname "$0")"
CXX=${CXX:-g++}
line() { printf '\n========== %s ==========\n' "$1"; }

line "01  name mangling: C++ vs extern \"C\""
$CXX -std=c++17 -O2 -c 01_mangling.cpp -o /tmp/01.o
echo "-- mangled (nm) --";    nm /tmp/01.o | grep -E ' T ' | sort
echo "-- demangled (nm -C) --"; nm -C /tmp/01.o | grep -E ' T ' | sort

line "02  SAME source, different STL ABI -> different symbol -> link break"
$CXX -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -c 02_stl_abi_break.cpp -o /tmp/new.o
$CXX -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -c 02_stl_abi_break.cpp -o /tmp/old.o
echo "new ABI symbol: $(nm /tmp/new.o | grep make_str)"
echo "old ABI symbol: $(nm /tmp/old.o | grep make_str)"
echo "-- link new-ABI caller against old-ABI object (expected: FAIL) --"
$CXX -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 02_caller.cpp /tmp/old.o -o /tmp/app 2>&1 | head -3 || true
echo "(the undefined-reference error above is the ABI break)"

line "03  struct layout fragility"
$CXX -std=c++17 03_struct_layout.cpp -o /tmp/03 && /tmp/03

line "04  the C-ABI fix: struct_size negotiation (npu_model.h pattern)"
$CXX -std=c++17 04_struct_size_negotiation.cpp -o /tmp/04 && /tmp/04
