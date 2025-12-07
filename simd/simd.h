#ifndef SIMD_H
#define SIMD_H

#include <vector>
#include <cstddef>

class simd {
    int avx_ver = 0; // 0: scalar, 1: SSE2/SSE4.x, 2: AVX2, 3: AVX-512

public:
    simd() noexcept;
    void detect_features() noexcept;

    // INT operations
    void addi(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept;
    static void addi_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void addi_sse2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void addi_avx2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void addi_avx512(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;

    void subi(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept;
    static void subi_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void subi_sse2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void subi_avx2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void subi_avx512(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;

    void muli(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept;
    static void muli_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void muli_sse(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void muli_avx2(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;
    static void muli_avx512(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;

    void divi(std::vector<int>& a, std::vector<int>& b, std::vector<int>& out) const noexcept;
    static void divi_scalar(const std::vector<int>& a, const std::vector<int>& b, std::vector<int>& out) noexcept;

    // FLOAT operations
    void addf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept;
    static void addf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void addf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void addf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void addf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;

    void subf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept;
    static void subf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void subf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void subf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void subf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;

    void mulf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept;
    static void mulf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void mulf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void mulf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void mulf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;

    void divf(std::vector<float>& a, std::vector<float>& b, std::vector<float>& out) const noexcept;
    static void divf_scalar(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void divf_sse(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void divf_avx2(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;
    static void divf_avx512(const std::vector<float>& a, const std::vector<float>& b, std::vector<float>& out) noexcept;

    // DOUBLE operations
    void addd(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept;
    static void addd_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void addd_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void addd_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void addd_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;

    void subd(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept;
    static void subd_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void subd_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void subd_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void subd_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;

    void muld(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept;
    static void muld_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void muld_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void muld_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void muld_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;

    void divd(std::vector<double>& a, std::vector<double>& b, std::vector<double>& out) const noexcept;
    static void divd_scalar(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void divd_sse2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void divd_avx2(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;
    static void divd_avx512(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& out) noexcept;

    //==================== DOT PRODUCT ====================//
    float dotf(const float* a, const float* b, size_t n);
    float dotf_scalar(const float* a, const float* b, size_t n);
    float dotf_sse(const float* a, const float* b, size_t n);
    float dotf_avx2(const float* a, const float* b, size_t n);
    float dotf_avx512(const float* a, const float* b, size_t n);

    //==================== META ====================//
    int version() const noexcept;
};

#endif