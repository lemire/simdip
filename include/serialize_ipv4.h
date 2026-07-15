#ifndef SERIALIZE_IPV4_H
#define SERIALIZE_IPV4_H

#include <x86intrin.h>

#include <cstdint>
#include <cstring>

size_t new_ipv4_to_string(uint32_t ip, char *out) { // out needs 16B of slack
  __m128i v  = _mm_cvtsi32_si128((int)ip);
  // each octet in both words of its dword: [b0,b0, b1,b1, b2,b2, b3,b3]
  __m128i vw = _mm_shuffle_epi8(v,
      _mm_setr_epi8(0,-1,0,-1, 1,-1,1,-1, 2,-1,2,-1, 3,-1,3,-1));
  __m128i q   = _mm_mulhi_epu16(vw, _mm_set1_epi32((6554 << 16) | 656)); // [b/100, b/10]
  __m128i t10 = _mm_mullo_epi16(q, _mm_set1_epi16(10));   // [10*(b/100), 10*(b/10)]
  __m128i ht  = _mm_sub_epi16(q, _mm_slli_epi32(t10, 16)); // [h, t]
  __m128i on  = _mm_sub_epi16(vw, t10);                    // odd words = ones digit
  // per lane: pick h, t, o, and a known-zero byte (high byte of h word)
  __m128i idx = _mm_setr_epi8(0,2,18,1, 4,6,22,5, 8,10,26,9, 12,14,30,13);
  __m128i chars = _mm_add_epi8(_mm_permutex2var_epi8(ht, idx, on),
      _mm_setr_epi8('0','0','0','.', '0','0','0','.',
                    '0','0','0','.', '0','0','0','.'));
  // mask: hundreds iff b>=100, tens iff b>=10, ones/dot always, kill pos 15
  __m128i vb = _mm_shuffle_epi8(v,
      _mm_setr_epi8(0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,-1));
  __mmask16 m = _mm_cmpge_epu8_mask(vb,
      _mm_setr_epi8(100,10,0,0, 100,10,0,0, 100,10,0,0, 100,10,0,1));
  _mm_storeu_si128((__m128i*)out, _mm_maskz_compress_epi8(m, chars));
  return _mm_popcnt_u32((unsigned)m);
}


/**
 * IPv4 dotted-decimal serialization (the inverse of parse_ipv4_avx512vl):
 * four octets in network order -> "a.b.c.d".
 *
 * Two SIMD paths are provided, both producing identical output:
 *  - serialize_ipv4_simd: digits from per-octet multiply-add reductions
 *    (Granlund-Montgomery), all four octets at once in 32-bit SSE lanes.
 *  - serialize_ipv4_ifma: digits from the AVX-512 IFMA method of
 *      Daniel Lemire, "Converting integers to decimal strings faster with
 *      AVX-512," 2022, and Jaël Champagne Gareau & Daniel Lemire,
 *      "Converting an Integer to a Decimal String in Under Two Nanoseconds,"
 *      arXiv:2604.26019 (reference code: fastfloat/int_serialization_benchmark).
 *
 * In both paths the variable-length result is compacted with a single
 * vpcompressb after a keep-mask drops the leading-zero digits.
 */

/**
 * Scalar reference serializer, used to validate the SIMD paths.
 *
 * @param octets Pointer to four bytes in network order (first octet first).
 * @param out    Output buffer of at least 16 bytes; NUL-terminated on return.
 * @return The number of characters written (excluding the NUL).
 */
static int serialize_ipv4_scalar(const uint8_t octets[4], char *out) {
    char *p = out;
    for (int i = 0; i < 4; i++) {
        unsigned x = octets[i];
        if (x >= 100) { *p++ = char('0' + x / 100); }
        if (x >= 10)  { *p++ = char('0' + (x / 10) % 10); }
        *p++ = char('0' + x % 10);
        if (i != 3) { *p++ = '.'; }
    }
    *p = '\0';
    return int(p - out);
}

// Shared tail: given D holding four 4-byte groups [hundreds, tens, ones, '.']
// (ASCII), and v holding the four octet values in 32-bit lanes, drop the
// leading-zero digits and the trailing '.', then compact in one vpcompressb.
static inline int serialize_ipv4_emit(__m128i D, __m128i v, char *out) {
    const __m128i mh = _mm_setr_epi8(0, -128, -128, -128, 4, -128, -128, -128,
                                     8, -128, -128, -128, 12, -128, -128, -128);
    const __m128i mt = _mm_setr_epi8(-128, 0, -128, -128, -128, 4, -128, -128,
                                     -128, 8, -128, -128, -128, 12, -128, -128);
    // Keep the ones digit and the separators always; the tens digit only when
    // x >= 10; the hundreds digit only when x >= 100; drop the final '.'.
    __m128i ge10  = _mm_cmpgt_epi32(v, _mm_set1_epi32(9));
    __m128i ge100 = _mm_cmpgt_epi32(v, _mm_set1_epi32(99));
    const __m128i ones_keep = _mm_setr_epi8(0, 0, -128, 0, 0, 0, -128, 0,
                                            0, 0, -128, 0, 0, 0, -128, 0);
    const __m128i dot_keep = _mm_setr_epi8(0, 0, 0, -128, 0, 0, 0, -128,
                                           0, 0, 0, -128, 0, 0, 0, 0);
    __m128i keep_bytes = _mm_or_si128(
        _mm_or_si128(ones_keep, dot_keep),
        _mm_or_si128(_mm_shuffle_epi8(ge100, mh), _mm_shuffle_epi8(ge10, mt)));
    __mmask16 keep = _mm_movepi8_mask(keep_bytes);

    __m128i packed = _mm_maskz_compress_epi8(keep, D);
    int len = _mm_popcnt_u32(keep);
    _mm_storeu_si128((__m128i *)out, packed);
    out[len] = '\0';
    return len;
}

/**
 * SIMD serializer using per-octet multiply-add digit extraction.
 *
 * @param octets Pointer to four bytes in network order (first octet first).
 * @param out    Output buffer of at least 16 bytes; NUL-terminated on return.
 * @return The number of characters written (excluding the NUL).
 */
static int serialize_ipv4_simd(const uint8_t octets[4], char *out) {
    uint32_t raw;
    std::memcpy(&raw, octets, 4);
    __m128i v = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw));  // [o0, o1, o2, o3]

    // Table-free digit extraction. For x in [0,255]:
    //   x / 100      == (x * 41)  >> 12
    //   (x % 100)/10 == (r * 205) >> 11   with r = x - 100*(x/100)
    __m128i q100 = _mm_srli_epi32(_mm_mullo_epi32(v, _mm_set1_epi32(41)), 12);
    __m128i r1   = _mm_sub_epi32(v, _mm_mullo_epi32(q100, _mm_set1_epi32(100)));
    __m128i q10  = _mm_srli_epi32(_mm_mullo_epi32(r1, _mm_set1_epi32(205)), 11);
    __m128i ones = _mm_sub_epi32(r1, _mm_mullo_epi32(q10, _mm_set1_epi32(10)));

    __m128i ascii0 = _mm_set1_epi32('0');
    __m128i H = _mm_add_epi32(q100, ascii0);
    __m128i T = _mm_add_epi32(q10, ascii0);
    __m128i O = _mm_add_epi32(ones, ascii0);

    // Scatter into per-octet 4-byte groups [hundreds, tens, ones, '.'].
    const __m128i mh = _mm_setr_epi8(0, -128, -128, -128, 4, -128, -128, -128,
                                     8, -128, -128, -128, 12, -128, -128, -128);
    const __m128i mt = _mm_setr_epi8(-128, 0, -128, -128, -128, 4, -128, -128,
                                     -128, 8, -128, -128, -128, 12, -128, -128);
    const __m128i mo = _mm_setr_epi8(-128, -128, 0, -128, -128, -128, 4, -128,
                                     -128, -128, 8, -128, -128, -128, 12, -128);
    const __m128i dots = _mm_setr_epi8(0, 0, 0, '.', 0, 0, 0, '.',
                                       0, 0, 0, '.', 0, 0, 0, '.');
    __m128i D = _mm_or_si128(
        _mm_or_si128(_mm_shuffle_epi8(H, mh), _mm_shuffle_epi8(T, mt)),
        _mm_or_si128(_mm_shuffle_epi8(O, mo), dots));

    return serialize_ipv4_emit(D, v, out);
}

#if defined(__AVX512IFMA__) && defined(__AVX512VBMI__)

// 8-digit decimal conversion via AVX-512 IFMA (Lemire 2022). Each of the 8
// 64-bit lanes holds one ASCII digit of n (n < 10^8), most significant first.
// A single IFMA pair, no scalar division and no cross-lane permute.
static inline __m512i to_string_avx512ifma_8digits(uint64_t n) {
    __m512i bcstq_l = _mm512_set1_epi64(n);
    constexpr uint64_t twoto52 = 0x10000000000000ULL;  // 2^52
    __m512i ifma_const = _mm512_setr_epi64(
        twoto52 / 100000000, twoto52 / 10000000, twoto52 / 1000000,
        twoto52 / 100000, twoto52 / 10000, twoto52 / 1000, twoto52 / 100,
        twoto52 / 10);
    __m512i zmmTen = _mm512_set1_epi64(10);
    __m512i asciiZero = _mm512_set1_epi64('0');
    __m512i lowbits = _mm512_madd52lo_epu64(ifma_const, bcstq_l, ifma_const);
    return _mm512_madd52hi_epu64(asciiZero, zmmTen, lowbits);
}

/**
 * SIMD serializer using two cheap 8-digit IFMA calls.
 *
 * Each octet pair is packed into a 6-digit integer (o0*1000 + o1 and
 * o2*1000 + o3) and converted with the lightweight 8-digit IFMA routine; the
 * two results are merged, spread into [hundreds, tens, ones, '.'] groups, and
 * compacted like the other paths. Avoids the 16-digit routine's scalar /10^8
 * and permutex2var.
 *
 * @param octets Pointer to four bytes in network order (first octet first).
 * @param out    Output buffer of at least 16 bytes; NUL-terminated on return.
 * @return The number of characters written (excluding the NUL).
 */
static int serialize_ipv4_ifma8(const uint8_t octets[4], char *out) {
    uint32_t raw;
    std::memcpy(&raw, octets, 4);
    __m128i v = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw));  // [o0, o1, o2, o3]

    uint64_t n01 = uint64_t(octets[0]) * 1000 + octets[1];  // 6 digits: 00 o0 o1
    uint64_t n23 = uint64_t(octets[2]) * 1000 + octets[3];  // 6 digits: 00 o2 o3
    __m128i r01 = _mm512_cvtusepi64_epi8(to_string_avx512ifma_8digits(n01));
    __m128i r23 = _mm512_cvtusepi64_epi8(to_string_avx512ifma_8digits(n23));
    // Merge: bytes 0-7 = n01 digits, bytes 8-15 = n23 digits.
    __m128i merged = _mm_unpacklo_epi64(r01, r23);

    // o0 = bytes 2,3,4 ; o1 = 5,6,7 ; o2 = 10,11,12 ; o3 = 13,14,15.
    const __m128i spread = _mm_setr_epi8(2, 3, 4, -128, 5, 6, 7, -128, 10, 11,
                                         12, -128, 13, 14, 15, -128);
    const __m128i dots = _mm_setr_epi8(0, 0, 0, '.', 0, 0, 0, '.',
                                       0, 0, 0, '.', 0, 0, 0, '.');
    __m128i D = _mm_or_si128(_mm_shuffle_epi8(merged, spread), dots);

    return serialize_ipv4_emit(D, v, out);
}

// 16-digit decimal conversion via AVX-512 IFMA (Lemire 2022). Returns the 16
// ASCII digits of n (n < 10^16), most significant byte first, zero-padded.
static inline __m128i to_string_avx512ifma(uint64_t n) {
    uint64_t n_15_08 = n / 100000000;
    uint64_t n_07_00 = n % 100000000;
    __m512i bcstq_h = _mm512_set1_epi64(n_15_08);
    __m512i bcstq_l = _mm512_set1_epi64(n_07_00);
    constexpr uint64_t twoto52 = 0x10000000000000ULL;  // 2^52
    __m512i ifma_const = _mm512_setr_epi64(
        twoto52 / 100000000, twoto52 / 10000000, twoto52 / 1000000,
        twoto52 / 100000, twoto52 / 10000, twoto52 / 1000, twoto52 / 100,
        twoto52 / 10);
    __m512i zmmTen = _mm512_set1_epi64(10);
    __m512i asciiZero = _mm512_set1_epi64('0');
    __m512i permb_const = _mm512_castsi128_si512(
        _mm_set_epi8(0x78, 0x70, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40, 0x38, 0x30,
                     0x28, 0x20, 0x18, 0x10, 0x08, 0x00));
    __m512i lowbits_h = _mm512_madd52lo_epu64(ifma_const, bcstq_h, ifma_const);
    __m512i lowbits_l = _mm512_madd52lo_epu64(ifma_const, bcstq_l, ifma_const);
    __m512i highbits_h = _mm512_madd52hi_epu64(asciiZero, zmmTen, lowbits_h);
    __m512i highbits_l = _mm512_madd52hi_epu64(asciiZero, zmmTen, lowbits_l);
    __m512i perm = _mm512_permutex2var_epi8(highbits_h, permb_const, highbits_l);
    return _mm512_castsi512_si128(perm);
}

/**
 * SIMD serializer using the AVX-512 IFMA digit method.
 *
 * The four octets are packed into a single 12-digit integer
 *   N = o0*10^9 + o1*10^6 + o2*10^3 + o3
 * (each octet occupies exactly three decimal positions, 000..255), converted
 * in one IFMA call, then spread into [hundreds, tens, ones, '.'] groups and
 * compacted exactly like serialize_ipv4_simd.
 *
 * @param octets Pointer to four bytes in network order (first octet first).
 * @param out    Output buffer of at least 16 bytes; NUL-terminated on return.
 * @return The number of characters written (excluding the NUL).
 */
static int serialize_ipv4_ifma(const uint8_t octets[4], char *out) {
    uint32_t raw;
    std::memcpy(&raw, octets, 4);
    __m128i v = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(raw));  // [o0, o1, o2, o3]

    uint64_t o0 = octets[0], o1 = octets[1], o2 = octets[2], o3 = octets[3];
    uint64_t N = (((o0 * 1000 + o1) * 1000) + o2) * 1000 + o3;  // 12 digits
    __m128i d16 = to_string_avx512ifma(N);  // [0..3]='0', [4..15]=our 12 digits

    // Spread the 12 digit bytes into per-octet 4-byte groups, then add dots.
    const __m128i spread = _mm_setr_epi8(4, 5, 6, -128, 7, 8, 9, -128, 10, 11,
                                         12, -128, 13, 14, 15, -128);
    const __m128i dots = _mm_setr_epi8(0, 0, 0, '.', 0, 0, 0, '.',
                                       0, 0, 0, '.', 0, 0, 0, '.');
    __m128i D = _mm_or_si128(_mm_shuffle_epi8(d16, spread), dots);

    return serialize_ipv4_emit(D, v, out);
}

#endif  // IFMA

// Canonical IPv4 serializer: the fastest available path (IFMA when present).
static inline int serialize_ipv4(const uint8_t octets[4], char *out) {
#if defined(__AVX512IFMA__) && defined(__AVX512VBMI__)
    return serialize_ipv4_ifma(octets, out);
#else
    return serialize_ipv4_simd(octets, out);
#endif
}

#endif
