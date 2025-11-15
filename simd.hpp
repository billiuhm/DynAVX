#include <vector>
#include <cstddef>
#include <immintrin.h>
#include <iostream>

// Add `-msse -msse2 -msse3 -mssse3 -msse4 -msse4 -mavx512f` to compile command

class simd {
    int avx_ver = 0; // 0: scalar, 1: SSE2/SSE4.x, 2: AVX2, 3: AVX-512

public:
    simd() noexcept {
        detect_features();
    }

    void detect_features() noexcept {
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_cpu_init();

        if (__builtin_cpu_supports("avx512f")) {
            avx_ver = 3;
        } else if (__builtin_cpu_supports("avx2")) {
            avx_ver = 2;
        } else if (__builtin_cpu_supports("avx")) {
            avx_ver = 1; // treat plain AVX as SSE2-level for our simple ops
        } else if (__builtin_cpu_supports("sse") || __builtin_cpu_supports("sse2") || __builtin_cpu_supports("sse3") || __builtin_cpu_supports("sse4.1") || __builtin_cpu_supports("sse4.2")) {
            avx_ver = 1;
        } else {
            avx_ver = 0;
        }
    #else
        // For non-GCC/Clang compilers, you might need different detection logic
        avx_ver = 0; // default to scalar
    #endif
    }

    //==================== INT ADD ====================//
    void addi(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: addi_avx512(a, b, out); break;
            case 2: addi_avx2(a, b, out); break;
            case 1: addi_sse2(a, b, out); break;
            default: addi_scalar(a, b, out); break;
        }
    }

    static void addi_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] + b[i];
    }

    static void addi_sse2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
            __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
            __m128i vc = _mm_add_epi32(va, vb);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    static void addi_avx2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
            __m256i vc = _mm256_add_epi32(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    static void addi_avx512(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 16;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&a[i]));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&b[i]));
            __m512i vc = _mm512_add_epi32(va, vb);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    //==================== INT SUB ====================//
    void subi(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: subi_avx512(a, b, out); break;
            case 2: subi_avx2(a, b, out); break;
            case 1: subi_sse2(a, b, out); break;
            default: subi_scalar(a, b, out); break;
        }
    }

    static void subi_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] - b[i];
    }

    static void subi_sse2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
            __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
            __m128i vc = _mm_sub_epi32(va, vb);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    static void subi_avx2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
            __m256i vc = _mm256_sub_epi32(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    static void subi_avx512(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 16;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&a[i]));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&b[i]));
            __m512i vc = _mm512_sub_epi32(va, vb);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    //==================== INT MUL ====================//
    void muli(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: muli_avx512(a, b, out); break;
            case 2: muli_avx2(a, b, out); break;
            case 1: muli_sse(a, b, out); break;
            default: muli_scalar(a, b, out); break;
        }
    }

    static void muli_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] * b[i];
    }

    static void muli_sse(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        // uses _mm_mullo_epi32 (SSE4.1); your compile flags mention sse4.1
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
            __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
            __m128i vc = _mm_mullo_epi32(va, vb);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    static void muli_avx2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
            __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
            __m256i vc = _mm256_mullo_epi32(va, vb);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    static void muli_avx512(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        const size_t step = 16;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512i va = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&a[i]));
            __m512i vb = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(&b[i]));
            __m512i vc = _mm512_mullo_epi32(va, vb);
            _mm512_storeu_si512(reinterpret_cast<__m512i*>(&out[i]), vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    //==================== INT DIV ====================//
    void divi(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        // There is no standard SIMD integer divide intrinsic for 32-bit integers in SSE/AVX.
        // We therefore use scalar division to preserve exact integer semantics (truncation toward zero).
        divi_scalar(a, b, out);
    }

    static void divi_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i) {
            // avoid division by zero UB; mirror normal integer behavior (implementation-defined)
            // Here we simply perform the division; caller should ensure b[i] != 0 if needed.
            out[i] = a[i] / b[i];
        }
    }

    //==================== FLOAT ADD ====================//
    void addf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: addf_avx512(a, b, out); break;
            case 2: addf_avx2(a, b, out); break;
            case 1: addf_sse(a, b, out); break;
            default: addf_scalar(a, b, out); break;
        }
    }

    static void addf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] + b[i];
    }

    static void addf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            __m128 vc = _mm_add_ps(va, vb);
            _mm_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    static void addf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    static void addf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 16;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vc = _mm512_add_ps(va, vb);
            _mm512_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    //==================== FLOAT SUB ====================//
    void subf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: subf_avx512(a, b, out); break;
            case 2: subf_avx2(a, b, out); break;
            case 1: subf_sse(a, b, out); break;
            default: subf_scalar(a, b, out); break;
        }
    }

    static void subf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] - b[i];
    }

    static void subf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            __m128 vc = _mm_sub_ps(va, vb);
            _mm_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    static void subf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    static void subf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 16;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vc = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    //==================== FLOAT MUL ====================//
    void mulf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: mulf_avx512(a, b, out); break;
            case 2: mulf_avx2(a, b, out); break;
            case 1: mulf_sse(a, b, out); break;
            default: mulf_scalar(a, b, out); break;
        }
    }

    static void mulf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] * b[i];
    }

    static void mulf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            __m128 vc = _mm_mul_ps(va, vb);
            _mm_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    static void mulf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    static void mulf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 16;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vc = _mm512_mul_ps(va, vb);
            _mm512_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    //==================== FLOAT DIV ====================//
    void divf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: divf_avx512(a, b, out); break;
            case 2: divf_avx2(a, b, out); break;
            case 1: divf_sse(a, b, out); break;
            default: divf_scalar(a, b, out); break;
        }
    }

    static void divf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] / b[i];
    }

    static void divf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128 va = _mm_loadu_ps(&a[i]);
            __m128 vb = _mm_loadu_ps(&b[i]);
            __m128 vc = _mm_div_ps(va, vb);
            _mm_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] / b[i];
    }

    static void divf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256 va = _mm256_loadu_ps(&a[i]);
            __m256 vb = _mm256_loadu_ps(&b[i]);
            __m256 vc = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] / b[i];
    }

    static void divf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept {
        const size_t step = 16;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512 va = _mm512_loadu_ps(&a[i]);
            __m512 vb = _mm512_loadu_ps(&b[i]);
            __m512 vc = _mm512_div_ps(va, vb);
            _mm512_storeu_ps(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] / b[i];
    }

    //==================== DOUBLE ADD ====================//
    void addd(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: addd_avx512(a, b, out); break;
            case 2: addd_avx2(a, b, out); break;
            case 1: addd_sse2(a, b, out); break;
            default: addd_scalar(a, b, out); break;
        }
    }

    static void addd_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] + b[i];
    }

    static void addd_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 2;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128d va = _mm_loadu_pd(&a[i]);
            __m128d vb = _mm_loadu_pd(&b[i]);
            __m128d vc = _mm_add_pd(va, vb);
            _mm_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    static void addd_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            __m256d vc = _mm256_add_pd(va, vb);
            _mm256_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    static void addd_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __m512d vb = _mm512_loadu_pd(&b[i]);
            __m512d vc = _mm512_add_pd(va, vb);
            _mm512_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] + b[i];
    }

    //==================== DOUBLE SUB ====================//
    void subd(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: subd_avx512(a, b, out); break;
            case 2: subd_avx2(a, b, out); break;
            case 1: subd_sse2(a, b, out); break;
            default: subd_scalar(a, b, out); break;
        }
    }

    static void subd_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] - b[i];
    }

    static void subd_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 2;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128d va = _mm_loadu_pd(&a[i]);
            __m128d vb = _mm_loadu_pd(&b[i]);
            __m128d vc = _mm_sub_pd(va, vb);
            _mm_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    static void subd_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            __m256d vc = _mm256_sub_pd(va, vb);
            _mm256_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    static void subd_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __m512d vb = _mm512_loadu_pd(&b[i]);
            __m512d vc = _mm512_sub_pd(va, vb);
            _mm512_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] - b[i];
    }

    //==================== DOUBLE MUL ====================//
    void muld(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: muld_avx512(a, b, out); break;
            case 2: muld_avx2(a, b, out); break;
            case 1: muld_sse2(a, b, out); break;
            default: muld_scalar(a, b, out); break;
        }
    }

    static void muld_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] * b[i];
    }

    static void muld_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 2;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128d va = _mm_loadu_pd(&a[i]);
            __m128d vb = _mm_loadu_pd(&b[i]);
            __m128d vc = _mm_mul_pd(va, vb);
            _mm_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    static void muld_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            __m256d vc = _mm256_mul_pd(va, vb);
            _mm256_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    static void muld_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __m512d vb = _mm512_loadu_pd(&b[i]);
            __m512d vc = _mm512_mul_pd(va, vb);
            _mm512_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] * b[i];
    }

    //==================== DOUBLE DIV ====================//
    void divd(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept {
        size_t size = a.size();
        if (b.size() != size || out.size() != size) return;

        switch (avx_ver) {
            case 3: divd_avx512(a, b, out); break;
            case 2: divd_avx2(a, b, out); break;
            case 1: divd_sse2(a, b, out); break;
            default: divd_scalar(a, b, out); break;
        }
    }

    static void divd_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        for (size_t i = 0; i < a.size(); ++i)
            out[i] = a[i] / b[i];
    }

    static void divd_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 2;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m128d va = _mm_loadu_pd(&a[i]);
            __m128d vb = _mm_loadu_pd(&b[i]);
            __m128d vc = _mm_div_pd(va, vb);
            _mm_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] / b[i];
    }

    static void divd_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 4;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m256d va = _mm256_loadu_pd(&a[i]);
            __m256d vb = _mm256_loadu_pd(&b[i]);
            __m256d vc = _mm256_div_pd(va, vb);
            _mm256_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] / b[i];
    }

    static void divd_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept {
        const size_t step = 8;
        size_t i = 0;
        for (; i + step <= a.size(); i += step) {
            __m512d va = _mm512_loadu_pd(&a[i]);
            __m512d vb = _mm512_loadu_pd(&b[i]);
            __m512d vc = _mm512_div_pd(va, vb);
            _mm512_storeu_pd(&out[i], vc);
        }
        for (; i < a.size(); ++i) out[i] = a[i] / b[i];
    }

    //==================== META ====================//
    int version() const noexcept { return avx_ver; }
};
