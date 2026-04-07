# ONNX Runtime 빌드 이슈 해결 기록 (XNNPACK EP, Linux x86_64)

## 환경

| 항목 | 버전 |
|------|------|
| OS | Linux x86_64 |
| GCC | 13.x |
| binutils (as) | 2.35 |
| ONNX Runtime | 1.26.0 |
| 목표 EP | XNNPACK |

## 배경

XNNPACK Execution Provider로 모델을 실행하기 위해 ONNX Runtime을 빌드하려 했으나, MLAS(Math Library for Accelerated Scenarios) 컴파일 단계에서 연속적으로 에러가 발생했다.

MLAS는 XNNPACK EP와 별개로 **모든 빌드에서 무조건 포함**되는 핵심 라이브러리이므로 비활성화할 수 없다. XNNPACK이 처리하지 않는 연산은 CPU EP(MLAS 기반)로 폴백된다.

---

## 에러 1: AVX-VNNI (`vpdpbusds`) 어셈블러 미지원

### 증상

```
/tmp/ccGLMAcC.s: Error: unsupported instruction 'vpdpbusds'
CMakeFiles/onnxruntime_mlas.dir/.../sqnbitgemm_kernel_avx2.cpp.o] Error 1
```

또는 `-mavxvnni` 전달 시:

```
avxvnniintrin.h:57: error: inlining failed in call to 'always_inline'
'_mm256_dpbusds_avx_epi32': target specific option mismatch
```

### 원인

`cmake/onnxruntime_mlas.cmake`에서 GCC 버전만으로 `-mavxvnni` 플래그 적용 여부를 판단했다.

```cmake
# 원본: GCC 버전만 체크
if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR CMAKE_CXX_COMPILER_VERSION VERSION_GREATER "11")
    set_source_files_properties(... PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c -mavxvnni")
```

GCC 13은 `-mavxvnni`를 인식하여 `vpdpbusds` 어셈블리를 생성하지만, binutils 2.35의 어셈블러(`as`)가 이 명령어를 모른다(binutils 2.36 이상 필요).

### AVX-VNNI란?

VNNI(Vector Neural Network Instructions)는 Intel이 도입한 정수 행렬 곱셈 가속 명령어 확장이다.

핵심 명령어 `vpdpbusds`는 uint8 × int8 곱셈 후 int32 누적을 **한 명령어**로 수행한다.

**Non-VNNI 경로** (2개 명령어):

```
Step 1: _mm256_maddubs_epi16  →  인접 2쌍 곱 → int16 × 16개
Step 2: _mm256_madd_epi16     →  인접 2쌍 합 → int32 × 8개
```

**VNNI 경로** (1개 명령어):

```
Step 1: _mm256_dpbusds_avx_epi32  →  4개씩 곱+합 → int32 × 8개 (포화 연산 포함)
```

VNNI는 명령어 수 절반, 추가 레지스터 불필요, 오버플로 안전이라는 장점이 있다.
4bit 양자화 모델(GPTQ, AWQ 등)의 GEMM 내부 루프에서 수십만 번 반복되므로 성능 차이가 크다.

두 가지 변형이 존재한다:

| | AVX-512 VNNI | AVX-VNNI |
|---|---|---|
| 레지스터 | 512-bit (zmm) | 256-bit (ymm) |
| 인코딩 | EVEX | VEX |
| 도입 CPU | Ice Lake (2019) | Alder Lake (2021) |
| GCC 플래그 | `-mavx512vnni` | `-mavxvnni` |

### 수정

**CMake** (`cmake/onnxruntime_mlas.cmake`):

GCC 버전 체크를 `check_cxx_source_compiles`로 교체하여 실제로 VNNI intrinsic을 컴파일+어셈블할 수 있는지 테스트한다.

```cmake
# 수정: 실제 컴파일+어셈블 테스트
set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma -mf16c -mavxvnni")
check_cxx_source_compiles("
  #include <immintrin.h>
  int main() {
    __m256i a = _mm256_setzero_si256();
    __m256i b = _mm256_setzero_si256();
    __m256i c = _mm256_dpbusds_avx_epi32(a, a, b);
    (void)c;
    return 0;
  }"
  HAS_AVXVNNI
)
unset(CMAKE_REQUIRED_FLAGS)
if(HAS_AVXVNNI)
  set_source_files_properties(... PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c -mavxvnni")
else()
  set_source_files_properties(... PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
  set_source_files_properties(... PROPERTIES COMPILE_DEFINITIONS "MLAS_AVXVNNI_UNSUPPORTED")
endif()
```

**C++ 헤더** (4개 파일):

VNNI 코드 가드에 `MLAS_AVXVNNI_UNSUPPORTED` 조건을 추가하여, VNNI 미지원 시 `if constexpr(vnni)` 분기 자체를 전처리기에서 제거한다.

```cpp
// 원본
#if !defined(__GNUC__) || (__GNUC__ > 10)

// 수정
#if (!defined(__GNUC__) || (__GNUC__ > 10)) && !defined(MLAS_AVXVNNI_UNSUPPORTED)
```

수정된 파일:
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2_int8_blklen16.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2_int8_blklen32.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2_int8_blklen64.h`
- `onnxruntime/core/mlas/lib/sqnbitgemm_m1_sym_kernel_avx2_int8_blklen32.h`

### 영향

양자화 GEMM은 non-VNNI 폴백 경로(`maddubs` + `madd`)로 정상 동작한다. VNNI 하드웨어가 있어도 non-VNNI 경로를 사용하므로 약간의 성능 저하가 있다.

---

## 에러 2: AVX-NE-CONVERT (`vcvtneeph2ps`) 어셈블러 미지원

### 증상

```
onnxruntime/core/mlas/lib/x86_64/cvtfp16Avx.S:
Error: no such instruction: 'vcvtneeph2ps ymm0, ymmword PTR [rdi]'
```

### 원인

`cvtfp16Avx.S`는 AVX-NE-CONVERT ISA를 사용한다. CMake에서 GCC >= 13.1이면 무조건 포함하지만, binutils 2.40 이상이 필요하다.

```cmake
# 원본: GCC 버전만 체크
if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 13.1 AND NOT(APPLE))
  set(mlas_platform_srcs_avx2 ${mlas_platform_srcs_avx2} .../cvtfp16Avx.S)
endif()
```

### 수정

**CMake** (`cmake/onnxruntime_mlas.cmake`):

`check_cxx_source_compiles`로 AVX-NE-CONVERT 지원 여부를 테스트하고, 지원 시 `MLAS_HAS_AVXNECONVERT` 매크로를 정의한다.

```cmake
set(CMAKE_REQUIRED_FLAGS "-mavxneconvert")
check_cxx_source_compiles("
  #include <immintrin.h>
  int main() {
    float f[8] = {};
    __m256 r = _mm256_cvtneeph_ps((__m256h*)f);
    (void)r;
    return 0;
  }"
  HAS_AVXNECONVERT
)
unset(CMAKE_REQUIRED_FLAGS)
if(HAS_AVXNECONVERT)
  # cvtfp16Avx.S 포함 + 매크로 정의
  target_compile_definitions(onnxruntime_mlas PRIVATE MLAS_HAS_AVXNECONVERT)
else()
  message(STATUS "Skipping cvtfp16Avx.S (AVX-NE-CONVERT not supported by assembler)")
endif()
```

**C++** (`onnxruntime/core/mlas/lib/platform.cpp`):

런타임 디스패치 가드를 `__GNUC__ >= 13`에서 `MLAS_HAS_AVXNECONVERT`로 변경한다.

```cpp
// 원본
#if (defined(_MSC_VER) && (_MSC_VER >= 1933)) || (defined(__GNUC__) && (__GNUC__ >= 13))

// 수정
#if (defined(_MSC_VER) && (_MSC_VER >= 1933)) || defined(MLAS_HAS_AVXNECONVERT)
```

### 영향

FP16→FP32 변환이 AVX2 커널(`vcvtph2ps`)로 폴백된다. 기능적 차이 없음.

---

## 에러 3: Windows x64 AVX-512 컴파일 플래그 누락

### 증상

Windows x64 빌드에서 AVX-512 intrinsic 관련 컴파일 에러.

### 원인

`sqnbitgemm_kernel_avx512.cpp`, `sqnbitgemm_kernel_avx512vnni.cpp`, `q4gemm_avx512.cpp`가 `target_sources`에 포함되어 있지만 `/arch:AVX512` 플래그가 설정되지 않았다.

### 수정

```cmake
set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
set_source_files_properties(${MLAS_SRC_DIR}/q4gemm_avx512.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX512")
```

---

## 근본 원인

모든 에러의 공통 원인: **컴파일러(GCC) 버전 ≠ 어셈블러(binutils) 버전**

ONNX Runtime의 CMake가 GCC 버전만으로 ISA 확장 지원 여부를 판단했으나, 실제로는 어셈블러도 해당 명령어를 인코딩할 수 있어야 한다. 이 두 컴포넌트는 독립적으로 설치/업데이트된다.

```
소스코드 → [GCC: C++ → 어셈블리] → [as(binutils): 어셈블리 → .o] → [CPU: 실행]
            GCC 13: OK              binutils 2.35: 실패!
```

### 어셈블러 vs CPU

어셈블러는 명령어 텍스트를 바이트코드로 변환하는 **번역기**이므로 CPU에 의존하지 않는다. 최신 binutils로 빌드하면 오래된 CPU에서도 동작한다. ONNX Runtime은 `platform.cpp`에서 CPUID를 통해 런타임에 CPU 기능을 감지하고, 해당 CPU가 지원하는 커널만 사용한다.

### 필요한 binutils 버전

| ISA 확장 | 최소 binutils | 명령어 예시 |
|----------|--------------|------------|
| AVX-512 기본 | 2.25 | `vfmadd231ps zmm` |
| AVX-VNNI | 2.36 | `vpdpbusds ymm` |
| AMX | 2.36 | `tdpbssd` |
| AVX-NE-CONVERT | 2.40 | `vcvtneeph2ps ymm` |

**권장: binutils 2.40 이상으로 업데이트하면 모든 에러가 해결된다.**

binutils 업데이트가 어려운 환경에서는 이 문서에 기술된 `check_cxx_source_compiles` 기반 수정을 적용하면 폴백 경로로 빌드할 수 있다.

---

## 수정 파일 목록

| 파일 | 수정 내용 |
|------|----------|
| `cmake/onnxruntime_mlas.cmake` | AVX-VNNI/AVX-NE-CONVERT 컴파일 테스트, Windows AVX-512 플래그 |
| `onnxruntime/core/mlas/lib/platform.cpp` | AVX-NE-CONVERT 디스패치 가드 |
| `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2_int8_blklen16.h` | VNNI 전처리기 가드 |
| `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2_int8_blklen32.h` | VNNI 전처리기 가드 |
| `onnxruntime/core/mlas/lib/sqnbitgemm_kernel_avx2_int8_blklen64.h` | VNNI 전처리기 가드 |
| `onnxruntime/core/mlas/lib/sqnbitgemm_m1_sym_kernel_avx2_int8_blklen32.h` | VNNI 전처리기 가드 |
