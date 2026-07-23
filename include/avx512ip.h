#ifndef AVX512IP_H
#define AVX512IP_H

#include <x86intrin.h>

#include <cstddef>
#include <cstdint>

// Scalar ada/WHATWG fallback used by the IPv4 SIMD path for unusual-but-valid
// forms (octal/hex segments, fewer than four parts).
#include "ada_ip.h"

/**
 * AVX-512VL and AVX-512BW based parsing of IPv4 and IPv6 addresses respectively.
 * credit to Shreesh Adiga.
 *
 * Daniel Lemire, "Parsing IP addresses crazily fast," in Daniel Lemire's blog,
 * June 8, 2023, https://lemire.me/blog/2023/06/08/parsing-ip-addresses-crazily-fast/.
 */

/**
 * Parses an IPv4 address in dotted-decimal form (e.g., "192.168.0.1")
 * using an AVX-512VL SIMD path.
 *
 * @param input Pointer to the input string (NUL termination is not required).
 * @param len Length of the input string.
 * @param ptr Output pointer to a uint32_t receiving the IPv4 value in host order.
 * @return 1 on success, 0 if the format is invalid.
 */
static int parse_ipv4_avx512vl(const char *input, size_t len, uint32_t *ptr) {
    // Unusual-but-valid forms (octal/hex segments, fewer than four parts) are
    // too long for, or rejected by, the SIMD path; defer them to the fallback.
    if (len > 15) [[unlikely]] { return parse_ipv4_ada(input, len, ptr); }
    int error = 0;
    __mmask16 len_mask = (1 << len) - 1;
    __m128i dot = _mm_set1_epi8('.');
    __m128i str = _mm_mask_loadu_epi8(dot, len_mask, (const __m128i *)input);
    __mmask16 dots_bitvector = _mm_cmpeq_epu8_mask(str, dot);
    __m128i digits_vec = _mm_sub_epi8(str, _mm_set1_epi8('0'));
    // string should have only 3 dots
    error |= (_mm_popcnt_u32(dots_bitvector & len_mask) != 3);
    __mmask16 copy_mask = ~dots_bitvector;
    __m128i index_reg = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    // obtain the indices corresponding to location of the dots
    __m128i compressed_index = _mm_maskz_compress_epi8(dots_bitvector, index_reg);
    // 4 32-bit integers (zero-extended) a0, a1, a2, a3 where a_i indicates the location (1-16) of the i_th dot in input string
    __m128i dot_location_epi32 = _mm_cvtepu8_epi32(compressed_index);
    // 4 32-bit integers 0, a0, a1, a2
    __m128i dot_location_shifted = _mm_bslli_si128(dot_location_epi32, 4);
    // compute a0 - 0, a1 - a0, a2 - a1, a3 - a2 which gives number of digits between dots (including dot)
    __m128i difference = _mm_sub_epi32(dot_location_epi32, dot_location_shifted);
    // need to subtract 1 to exclude dot and get in-between digits count
    __m128i num_digits_between_dots = _mm_sub_epi32(difference, _mm_set1_epi32(1));
    // error if num_digits is not between 1 and 3
    // using unsigned ((x - 1) > 2) for error expression instead of two compares
    error |= _mm_cmpgt_epu8_mask(_mm_sub_epi32(num_digits_between_dots, _mm_set1_epi32(0x1)), _mm_set1_epi32(0x2));
    __m128i expand_mask_creation_register = _mm_setr_epi8(
            0,    0,    0,    0,
            0,    0,    0, 0xff,
            0,    0, 0xff, 0xff,
            0, 0xff, 0xff, 0xff
    );

    // _mm_permutevar_ps is required to index into the expand_mask_creation_register
    // _mm_movepi8_mask for each 32 bit integer will have the bitmask for zero padding necessary per octet.
    __mmask16 expand_mask = _mm_movepi8_mask(
            _mm_castps_si128(_mm_permutevar_ps(_mm_castsi128_ps(expand_mask_creation_register), num_digits_between_dots)));

    // get rid of '.' and compress digits
    __m128i compressed_vec = _mm_maskz_compress_epi8(copy_mask, digits_vec);
    // all entries must be less than 10
    error |= _mm_cmpgt_epu8_mask(compressed_vec, _mm_set1_epi8(0x9));
    // expand the compressed_vec to have all octets with 4 digits (zero padding)
    __m128i padded_digits_vec = _mm_maskz_expand_epi8(expand_mask, compressed_vec);

    // Error if the leading digit is 0 e.g. 02, 012 etc will be treated as error.
    // Extract the most significant digit from num_digits_between_dots and check if it is 0.
    // If there is only 1 digit then skip the check as zero is valid number
    __m128i shifted_vec = _mm_shuffle_epi8(padded_digits_vec,
            _mm_sub_epi32(
                _mm_setr_epi8(4, 0x0, 0x0, 0x0, 8, 0x0, 0x0, 0x0, 12, 0x0, 0x0, 0x0, 16, 0x0, 0x0, 0x0),
                num_digits_between_dots));
    __mmask8 more_than_1digit = _mm_cmpgt_epu32_mask(_mm_sub_epi32(num_digits_between_dots, _mm_set1_epi32(0x1)), _mm_setzero_si128());
    error |= _mm_mask_cmpeq_epu32_mask(more_than_1digit, _mm_setzero_si128(), shifted_vec);
    // multiply digits by 100, 10, 1 respectively and sum them to a 32 bit num
    __m128i res = _mm_dpbusd_epi32(_mm_setzero_si128(), padded_digits_vec, _mm_set1_epi32(0x010a6400));
    // error if any of the 4 octet is bigger than 255
    error |= _mm_cmpgt_epu32_mask(res, _mm_set1_epi32(0xff));

    if (!error) [[likely]] {
        // assemble the bytes and convert from network order to host order and write to memory
        _mm_storeu_si32(ptr, _mm_shuffle_epi8(res, _mm_setr_epi8(0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)));
        return 1;
    }

    // The SIMD path is strict: it rejects valid-but-unusual forms (octal, hex,
    // leading zeros, fewer than four parts). Before declaring failure, give the
    // permissive ada parser a chance to accept the address.
    return parse_ipv4_ada(input, len, ptr);
}

#endif